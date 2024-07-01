import sys
import math

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights

from utils.utils import shuffle_batch, shuffle_instance
from architecture.transformer import Transformer, pos_enc_1d

class IPSNet(nn.Module):
    """
    Net that runs all the main components:
    patch encoder, IPS, patch aggregator and classification head
    """

    def get_conv_patch_enc(self, enc_type, pretrained, n_chan_in, n_res_blocks):
        # Get architecture for patch encoder
        if enc_type == 'resnet18': 
            res_net_fn = resnet18
            weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        elif enc_type == 'resnet50':
            res_net_fn = resnet50
            # Resnet50 pretrained weights not used in experiments
            weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None        

        res_net = res_net_fn(weights=weights)

        if n_chan_in == 1:
            # Standard resnet uses 3 input channels
            res_net.conv1 = nn.Conv2d(n_chan_in, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Compose patch encoder
        layer_ls = []
        layer_ls.extend([
            res_net.conv1,
            res_net.bn1,
            res_net.relu,
            res_net.maxpool,
            res_net.layer1,
            res_net.layer2
        ])

        if n_res_blocks == 4:
            layer_ls.extend([
                res_net.layer3,
                res_net.layer4
            ])
        
        layer_ls.append(res_net.avgpool)

        return nn.Sequential(*layer_ls)

    def get_projector(self, n_chan_in, D):
        return nn.Sequential(
            nn.LayerNorm(n_chan_in, eps=1e-05, elementwise_affine=False),
            nn.Linear(n_chan_in, D),
            nn.BatchNorm1d(D),
            nn.ReLU()
        )   

    def get_output_layers(self, tasks):
        """
        Create an output layer for each task according to task definition
        """

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

        if self.is_image:
            self.encoder = self.get_conv_patch_enc(conf.enc_type, conf.pretrained,
                conf.n_chan_in, conf.n_res_blocks)
        else:
            self.encoder = self.get_projector(conf.n_chan_in, self.D)

        # Define the multi-head cross-attention transformer
        self.transf = Transformer(conf.n_token, conf.H, conf.D, conf.D_k, conf.D_v,
            conf.D_inner, conf.attn_dropout, conf.dropout)

        # Optionally use standard 1d sinusoidal positional encoding
        if conf.use_pos:
            self.pos_enc = pos_enc_1d(conf.D, conf.N).unsqueeze(0).to(device)
        else:
            self.pos_enc = None
        
        # Define an output layer for each task
        self.output_layers = self.get_output_layers(conf.tasks)

    def do_shuffle(self, patches, pos_enc):
        """
        Shuffles patches and pos_enc so that patches that have an equivalent score
        are sampled uniformly
        """

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

    def score_and_select(self, emb, emb_pos, M, idx):
        """ 
        Scores embeddings and selects the top-M embeddings
        """
        D = emb.shape[2]

        emb_to_score = emb_pos if torch.is_tensor(emb_pos) else emb

        # Obtain scores from transformer
        attn = self.transf.get_scores(emb_to_score) # (B, M+I)

        # Get indixes of top-scoring patches
        top_idx = torch.topk(attn, M, dim = -1)[1] # (B, M)
        
        # Update memory buffers
        # Note: Scoring is based on `emb_to_score`, selection is based on `emb`
        mem_emb = torch.gather(emb, 1, top_idx.unsqueeze(-1).expand(-1,-1,D))
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

    # IPS runs in no-gradient mode
    @torch.no_grad()
    def ips(self, patches):
        """ Iterative Patch Selection """

        # Get useful variables
        M = self.M
        I = self.I
        D = self.D  
        device = self.device
        shuffle = self.shuffle
        use_pos = self.use_pos
        pos_enc = self.pos_enc
        patch_shape = patches.shape
        B, N = patch_shape[:2]

        # Shortcut: IPS not required when memory is larger than total number of patches
        if M >= N:
            # Batchify pos enc
            pos_enc = pos_enc.expand(B, -1, -1) if use_pos else None
            return patches.to(device), pos_enc 

        # IPS runs in evaluation mode
        if self.training:
            self.encoder.eval()
            self.transf.eval()

        # Batchify positional encoding
        if use_pos:
            pos_enc = pos_enc.expand(B, -1, -1)

        # Shuffle patches (i.e., randomize when patches obtain identical scores)
        if shuffle:
            patches, pos_enc = self.do_shuffle(patches, pos_enc)

        # Init memory buffer
        # Put patches onto GPU in case it is not there yet (lazy loading).
        # `to` will return self in case patches are located on GPU already (eager loading)
        init_patch = patches[:,:M].to(device) 
        
        ## Embed
        mem_emb = self.encoder(init_patch.reshape(-1, *patch_shape[2:]))
        mem_emb = mem_emb.view(B, M, -1)
        
        # Init memory indixes in order to select patches at the end of IPS.
        idx = torch.arange(N, dtype=torch.int64, device=device).unsqueeze(0).expand(B, -1)
        mem_idx = idx[:,:M]

        # Apply IPS for `n_iter` iterations
        n_iter = math.ceil((N - M) / I)
        for i in range(n_iter):
            # Get next patches
            start_idx = i * I + M
            end_idx = min(start_idx + I, N)

            iter_patch = patches[:, start_idx:end_idx].to(device)
            iter_idx = idx[:, start_idx:end_idx]

            # Embed
            iter_emb = self.encoder(iter_patch.reshape(-1, *patch_shape[2:]))
            iter_emb = iter_emb.view(B, -1, D)
            
            # Concatenate with memory buffer
            all_emb = torch.cat((mem_emb, iter_emb), dim=1)
            all_idx = torch.cat((mem_idx, iter_idx), dim=1)
            # When using positional encoding, also apply it during patch selection
            if use_pos:
                all_pos_enc = torch.gather(pos_enc, 1, all_idx.view(B, -1, 1).expand(-1, -1, D))
                all_emb_pos = all_emb + all_pos_enc
            else:
                all_emb_pos = None

            # Select Top-M patches according to cross-attention scores
            mem_emb, mem_idx = self.score_and_select(all_emb, all_emb_pos, M, all_idx)

        # Select patches
        n_dim_expand = len(patch_shape) - 2
        mem_patch = torch.gather(patches, 1, 
            mem_idx.view(B, -1, *(1,)*n_dim_expand).expand(-1, -1, *patch_shape[2:]).to(patches.device)
        ).to(device)

        if use_pos:
            mem_pos = torch.gather(pos_enc, 1, mem_idx.unsqueeze(-1).expand(-1, -1, D))
        else:
            mem_pos = None

        # Set components back to training mode
        # Although components of `self` that are relevant for IPS have been set to eval mode,
        # self is still in training mode at training time, i.e., we can use it here.
        if self.training:
            self.encoder.train()
            self.transf.train()
    
        # Return selected patch and corresponding positional embeddings
        return mem_patch, mem_pos

    def forward(self, mem_patch, mem_pos=None):
        """
        After M patches have been selected during IPS, encode and aggregate them.
        The aggregated embedding is input to a classification head.
        """

        patch_shape = mem_patch.shape
        B, M = patch_shape[:2]

        mem_emb = self.encoder(mem_patch.reshape(-1, *patch_shape[2:]))
        mem_emb = mem_emb.view(B, M, -1)        

        if torch.is_tensor(mem_pos):
            mem_emb = mem_emb + mem_pos

        # Seperate main embeddings and image embeddings
        image_emb = self.transf(mem_emb)[0]

        branch_embeddings = self.transf(mem_emb)[1]

        preds = self.get_preds(image_emb)

        branch_preds = []

        for i in range(branch_embeddings.shape[1]):
          branch_preds.append(self.get_preds(branch_embeddings[:,i]))

        # if isinstance(preds, dict):
        #     for key, value in preds.items():
                # print(f'Prediction key: {key}, value shape: {value.shape if hasattr(value, "shape") else "Not a tensor"}')
                # Prediction key: majority, value shape: torch.Size([16, 10])
                # Prediction key: max, value shape: torch.Size([16, 10])
                # Prediction key: top, value shape: torch.Size([16, 10])
                # Prediction key: multi, value shape: torch.Size([16, 10])
        # else:
            #print(f'Prediction shape main: {preds.shape}')

        # Similarly for branch_preds[0] if it might be a dictionary
        # if isinstance(branch_preds[0], dict):
        #     for key, value in branch_preds[0].items():
                # print(f'Branch prediction key: {key}, value shape: {value.shape if hasattr(value, "shape") else "Not a tensor"}')
                # Branch prediction key: majority, value shape: torch.Size([16, 10])
                # Branch prediction key: max, value shape: torch.Size([16, 10])
                # Branch prediction key: top, value shape: torch.Size([16, 10])
                # Branch prediction key: multi, value shape: torch.Size([16, 10])
        #else:
            # print(f'Prediction shape main: {branch_preds[0].shape}')

            
        return preds, branch_preds
        
    def compute_diversity_loss(self):
        """
        Compute the diversity loss using the attention maps from the transformer.
        """
        return self.transf.compute_diversity_loss()
