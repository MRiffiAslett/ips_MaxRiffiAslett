# transformer.py
import math

import torch
from torch import nn
import torch.nn.functional as F

def pos_enc_1d(D, len_seq):
    if D % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(D))
    pe = torch.zeros(len_seq, D)
    position = torch.arange(0, len_seq).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, D, 2, dtype=torch.float) *
                         -(math.log(10000.0) / D)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

class ScaledDotProductAttention(nn.Module):
    ''' Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def compute_attn(self, q, k):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        attn = self.dropout(torch.softmax(attn, dim=-1))
        return attn

    def forward(self, q, k, v):
        attn = self.compute_attn(q, k)
        output = torch.matmul(attn, v)
        return output

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, n_token, H, D, D_k, D_v, attn_dropout=0.1, dropout=0.1):
        super().__init__()
        self.n_token = n_token
        self.H = H
        self.D_k = D_k
        self.D_v = D_v

        self.q = nn.Parameter(torch.empty((1, n_token, D)))
        q_init_val = math.sqrt(1 / D_k)
        nn.init.uniform_(self.q, a=-q_init_val, b=q_init_val)

        self.q_w = nn.Linear(D, H * D_k, bias=False)
        self.k_w = nn.Linear(D, H * D_k, bias=False)
        self.v_w = nn.Linear(D, H * D_v, bias=False)
        self.fc = nn.Linear(H * D_v, D, bias=False)
        self.fb = nn.Linear(D_v, D, bias=False)

        self.attention = ScaledDotProductAttention(temperature=D_k ** 0.5, attn_dropout=attn_dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(D, eps=1e-6)

    def get_attn(self, x):
        D_k, H, n_token = self.D_k, self.H, self.n_token
        B, len_seq = x.shape[:2]

        q = self.q_w(self.q).view(1, n_token, H, D_k)
        k = self.k_w(x).view(B, len_seq, H, D_k)

        q, k = q.transpose(1, 2), k.transpose(1, 2)

        attn = self.attention.compute_attn(q, k)

        return attn

    def forward(self, x):
        D_k, D_v, H, n_token = self.D_k, self.D_v, self.H, self.n_token
        B, len_seq = x.shape[:2]

        q = self.q_w(self.q).view(1, n_token, H, D_k) # (1, 4, 8, 16)
        k = self.k_w(x).view(B, len_seq, H, D_k) # (16, 100, 8, 16)
        v = self.v_w(x).view(B, len_seq, H, D_v) # (16, 100, 8, 16)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # q: (1, 8, 4, 16)
        # k: (16, 8, 100, 16)
        # v: (16, 8, 100, 16)
        # print(f"q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")

        x = self.attention(q, k, v)
        # x: (16, 8, 4, 16)
        # print(f"Output of attention mechanism: {x.shape}")

        # Store attention maps for diversity loss
        self.attn_maps = x

        x_br = x.transpose(1, 2).contiguous().view(B, n_token, -1)
        # x_br: (16, 4, 128)
        # print(f"Transposed attention output: {x_br.shape}")

        # Apply fc layer (aggregates heads)
        x = self.dropout(self.fc(x_br))
        # x: (16, 4, 128)
        x += self.q
        x = self.layer_norm(x)
        # x: (16, 4, 128)
        # print(f"Output after fc: {x.shape}")

        # Apply fb layer to each head separately
        attention_branches = []
        for i in range(H):
            head_output = x_br[:, :, i * D_v:(i + 1) * D_v]
            # head_output: (16, 4, 16)
            branch_output = self.fb(head_output)
            # branch_output: (16, 4, 128)
            attention_branches.append(branch_output)

        attention_branches = torch.stack(attention_branches, dim=1)
        # attention_branches: (16, 8, 4, 128)
        # print(f"Attention branches output: {attention_branches.shape}")

        return x, attention_branches

class MLP(nn.Module):
    ''' MLP consisting of two feed-forward layers '''

    def __init__(self, D, D_inner, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(D, D_inner)
        self.w_2 = nn.Linear(D_inner, D)
        self.layer_norm = nn.LayerNorm(D, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(torch.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x

class Transformer(nn.Module):
    """ Cross-attention based transformer module """

    def __init__(self, n_token, H, D, D_k, D_v, D_inner, attn_dropout=0.1, dropout=0.1):
        super().__init__()
        self.crs_attn = MultiHeadCrossAttention(n_token, H, D, D_k, D_v, attn_dropout=attn_dropout, dropout=dropout)
        self.mlp = MLP(D, D_inner, dropout=dropout)
        self.n_branches = H  # Use H as the number of branches
        self.attn_maps = None  # Initialize attn_maps attribute

    def get_scores(self, x):
        attn = self.crs_attn.get_attn(x)
        return attn.mean(dim=1).transpose(1, 2).mean(-1)

    def forward(self, x):
        # Main forward pass
        attn_output, attention_branches = self.crs_attn(x)  # Unpack the tuple
        self.attn_maps = self.crs_attn.attn_maps  # Store attn_maps in the Transformer class
        main_output = self.mlp(attn_output)
        # print(f'main attention output shape: {main_output.shape}')
        # Pass attention branches through the same MLP
        branch_outputs = []
        for i in range(self.n_branches):
            branch_output = self.mlp(attention_branches[:, i])
            # print(f'Branch nb {i} shape: {branch_output.shape}')
            # Branch nb 2 shape: torch.Size([16, 4, 128])
            # Branch nb 3 shape: torch.Size([16, 4, 128])
            # Branch nb 4 shape: torch.Size([16, 4, 128])
            # Branch nb 5 shape: torch.Size([16, 4, 128])
            # Branch nb 6 shape: torch.Size([16, 4, 128])
            # Branch nb 7 shape: torch.Size([16, 4, 128])
            branch_outputs.append(branch_output)

        branch_outputs = torch.stack(branch_outputs, dim=1)  # Stack along a new dimension

        return main_output, branch_outputs

    def compute_diversity_loss(self):
        return compute_diversity_loss(self.attn_maps)
