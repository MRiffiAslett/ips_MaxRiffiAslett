import sys
import math

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
from utils.utils import shuffle_batch, shuffle_instance
from architecture.transformer import Transformer, pos_enc_1d

class IPSNet(nn.Module):
    def get_conv_patch_enc(self, enc_type, pretrained, n_chan_in, n_res_blocks, freeze_weights):
        if enc_type == 'resnet18':
            res_net_fn = resnet18
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            out_dim = 512
        elif enc_type == 'resnet50':
            res_net_fn = resnet50
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            out_dim = 2048

        res_net = res_net_fn(weights=weights)

        if freeze_weights and pretrained:
            for param in res_net.parameters():
                param.requires_grad = False

        if n_chan_in == 1:
            res_net.conv1 = nn.Conv2d(n_chan_in, 64, kernel_size=7, stride=2, padding=3, bias=False)

        modules = list(res_net.children())[:-2]
        encoder = nn.Sequential(*modules)
        projection = nn.Linear(out_dim * 2 * 2, 128)
        return encoder, projection, 128

    def get_projector(self, n_chan_in, D):
        return nn.Sequential(
            nn.LayerNorm(n_chan_in, eps=1e-05, elementwise_affine=False),
            nn.Linear(n_chan_in, D),
            nn.BatchNorm1d(D),
            nn.ReLU()
        )

    def get_output_layers(self, tasks):
        D = self.D
        n_class = self.n_class

        output_layers = nn.ModuleDict()
        for task in tasks.values():
            if task['act_fn'] == 'softmax':
                act_fn = nn.Softmax(dim=-1)
            elif task['act_fn'] == 'sigmoid':
                act_fn = nn.Sigmoid()
            
            layers = [
                nn.Linear(D, n_class),
                act_fn
            ]
            output_layers[task['name']] = nn.Sequential(*layers)

        return output_layers

    def __init__(self, device, conf):
        super().__init__()

        self.device = device
        self.n_class = conf.n_class
        self.M = conf.M
        self.I = conf.I
        self.D = conf.D 
        self.use_pos = conf.use_pos
        self.tasks = conf.tasks
        self.shuffle = conf.shuffle
        self.shuffle_style = conf.shuffle_style
        self.is_image = conf.is_image
        self.mask_p = conf.mask_p
        self.mask_K = conf.mask_K

        freeze_weights = getattr(conf, 'freeze_weights', True)

        if self.is_image:
            self.encoder, self.projection, self.encoder_out_dim = self.get_conv_patch_enc(
                conf.enc_type, conf.pretrained, conf.n_chan_in, conf.n_res_blocks, freeze_weights
            )
        else:
            self.encoder = self.get_projector(conf.n_chan_in, self.D)
            self.projection = None

        self.transf = Transformer(conf.n_token, conf.H, self.encoder_out_dim, conf.D_k, conf.D_v,
            conf.D_inner, conf.attn_dropout, conf.dropout)

        if conf.use_pos:
            self.pos_enc = pos_enc_1d(conf.D, conf.N).unsqueeze(0).to(device)
        else:
            self.pos_enc = None
        
        self.output_layers = self.get_output_layers(conf.tasks)

    def do_shuffle(self, patches, pos_enc):
        shuffle_style = self.shuffle_style
        if shuffle_style == 'batch':
            patches, shuffle_idx = shuffle_batch(patches)
            if torch.is_tensor(pos_enc):
                pos_enc, _ = shuffle_batch(pos_enc, shuffle_idx)
        elif shuffle_style == 'instance':
            patches, shuffle_idx = shuffle_instance(patches, 1)
            if torch.is_tensor(pos_enc):
                pos_enc, _ = shuffle_instance(pos_enc, 1, shuffle_idx)
        
        return patches, pos_enc

    def score_and_select(self, emb, emb_pos, M, idx, mask_K, mask_p):
        D = emb.shape[2]
    
        emb_to_score = emb_pos if torch.is_tensor(emb_pos) else emb
    
        attn = self.transf.get_scores(emb_to_score)
        
        top_K_idx = torch.topk(attn, self.mask_K, dim=-1)[1]
                
        mask = (torch.rand(top_K_idx.shape, device=attn.device) < self.mask_p).float()
    
        attn.scatter_(1, top_K_idx, attn.gather(1, top_K_idx) * (1 - mask))
    
        top_idx = torch.topk(attn, M, dim=-1)[1]
    
        mem_emb = torch.gather(emb, 1, top_idx.unsqueeze(-1).expand(-1, -1, D))
        mem_idx = torch.gather(idx, 1, top_idx)
        
        return mem_emb, mem_idx

    def get_preds(self, embeddings):
        preds = {}
        for task in self.tasks.values():
            t_name, t_id = task['name'], task['id']
            layer = self.output_layers[t_name]

            emb = embeddings[:,t_id]
            preds[t_name] = layer(emb)            

        return preds

    @torch.no_grad()
    def ips(self, patches):
        M = self.M
        I = self.I
        D = self.encoder_out_dim
        device = self.device
        shuffle = self.shuffle
        use_pos = self.use_pos
        pos_enc = self.pos_enc
        patch_shape = patches.shape
        B, N = patch_shape[:2]
        mask_p = self.mask_p
        mask_K = self.mask_K

        if M >= N:
            pos_enc = pos_enc.expand(B, -1, -1) if use_pos else None
            return patches.to(device), pos_enc 

        if self.training:
            self.encoder.eval()
            self.transf.eval()

        if use_pos:
            pos_enc = pos_enc.expand(B, -1, -1)

        if shuffle:
            patches, pos_enc = self.do_shuffle(patches, pos_enc)

        init_patch = patches[:,:M].to(device)
        
        mem_emb = self.encoder(init_patch.reshape(-1, *patch_shape[2:]))
        mem_emb = mem_emb.view(B, M, -1)

        if self.projection:
            mem_emb = self.projection(mem_emb.view(B * M, -1)).view(B, M, -1)
        
        idx = torch.arange(N, dtype=torch.int64, device=device).unsqueeze(0).expand(B, -1)
        mem_idx = idx[:,:M]

        n_iter = math.ceil((N - M) / I)
        for i in range(n_iter):
            start_idx = i * I + M
            end_idx = min(start_idx + I, N)

            iter_patch = patches[:, start_idx:end_idx].to(device)
            iter_idx = idx[:, start_idx:end_idx]

            iter_emb = self.encoder(iter_patch.reshape(-1, *patch_shape[2:]))
            iter_emb = iter_emb.view(B, iter_patch.size(1), -1)
            
            if self.projection:
                iter_emb = self.projection(iter_emb.view(B * iter_emb.size(1), -1)).view(B, -1, 128)
            
            all_emb = torch.cat((mem_emb, iter_emb), dim=1)
            all_idx = torch.cat((mem_idx, iter_idx), dim=1)

            if use_pos:
                all_pos_enc = torch.gather(pos_enc, 1, all_idx.view(B, -1, 1).expand(-1, -1, D))
                all_emb_pos = all_emb + all_pos_enc
            else:
                all_emb_pos = None

            mem_emb, mem_idx = self.score_and_select(all_emb, all_emb_pos, M, all_idx, mask_K, mask_p)

        n_dim_expand = len(patch_shape) - 2
        mem_patch = torch.gather(patches, 1, 
            mem_idx.view(B, -1, *(1,)*n_dim_expand).expand(-1, -1, *patch_shape[2:]).to(patches.device)
        ).to(device)

        if use_pos:
            mem_pos = torch.gather(pos_enc, 1, mem_idx.unsqueeze(-1).expand(-1, -1, D))
        else:
            mem_pos = None

        if self.training:
            self.encoder.train()
            self.transf.train()
    
        return mem_patch, mem_pos

    def forward(self, mem_patch, mem_pos=None):
        patch_shape = mem_patch.shape
        B, M = patch_shape[:2]

        mem_emb = self.encoder(mem_patch.reshape(-1, *patch_shape[2:]))
        mem_emb = mem_emb.view(B, M, -1)
        
        if self.projection:
            mem_emb = self.projection(mem_emb.view(B * M, -1)).view(B, M, -1)

        if torch.is_tensor(mem_pos):
            mem_emb = mem_emb + mem_pos

        image_emb = self.transf(mem_emb)[0]
        branch_embeddings = self.transf(mem_emb)[1]

        preds = self.get_preds(image_emb)

        branch_preds = []
        for i in range(branch_embeddings.shape[1]):
            branch_preds.append(self.get_preds(branch_embeddings[:,i]))

        return preds, branch_preds

# Define your configuration and other necessary components here
