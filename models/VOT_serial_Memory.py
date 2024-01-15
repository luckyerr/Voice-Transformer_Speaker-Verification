# -*- encoding: utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from torch.nn import Parameter
from IPython import embed
from models import encoder
from models.bidfsmn import BiDfsmnLayer
MIN_NUM_PATCHES = 16

class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, bottleneck_dim):
        super(AttentiveStatsPool, self).__init__()
        # Use Conv1d with stride == 1 rather than Linear, then we don't need to transpose inputs.
        self.linear1 = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1)  # equals W and b in the paper
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)  # equals V and k in the paper

    def forward(self, x):
        # DON'T use ReLU here! In experiments, I find ReLU hard to converge.
        alpha = torch.tanh(self.linear1(x))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * x ** 2, dim=2) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5
        self.model = BiDfsmnLayer(
                 128,
                 301,
                 3,
                 4,
                 dilation=1,
                 dropout=0.2)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
       #v torch.Size([64, 8, 301, 32])
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max
        #embed()
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        v_=self.model(v.reshape(-1,301,32)).reshape(v.shape[0],-1,301,32)
        out=out+v_
        

        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)

        return out
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            #embed()
            x = ff(x)
        return x
class VOT_serial(nn.Module):
     def __init__(self, *, image_size, patch_size, ac_patch_size,
                          pad, dim, depth, heads, mlp_dim, pool = 'cls', channels = 1, dim_head = 64, dropout = 0.1, emb_dropout = 0.):
         super().__init__()
         assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
         num_patches = (image_size // patch_size) ** 2
         patch_dim = 112
         assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
         assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

         self.patch_size = patch_size
         self.soft_split = nn.Unfold(kernel_size=(ac_patch_size, ac_patch_size), stride=(self.patch_size, self.patch_size), padding=(pad, pad))


         self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
         self.patch_to_embedding = nn.Linear(patch_dim, dim)
         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
         self.dropout = nn.Dropout(emb_dropout)

         #self.transformer1 = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
         self.transformer1 = Transformer(dim, 20, 8, 32, 2048, 0.15)
         self.transformer2 = Transformer(dim, 12, 4, 32, 1024, 0.15)
         #self.transformer3 = Transformer(dim, 6, 2, 32, 1024, 0.15)
         self.pool = pool
         self.to_latent = nn.Identity()
         self.mlp_head = nn.Sequential(
             nn.LayerNorm(256),
         )
         self.asp = AttentiveStatsPool(256, 128)
         self.asp_bn = nn.BatchNorm1d(512)
         
     def forward(self, x,  mask = None):
         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
         with torch.cuda.amp.autocast(enabled=False):
             x=x.squeeze(1).to("cuda")
             x = self.patch_to_embedding(x)
             b, n, _ = x.shape
             cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
             x = torch.cat((cls_tokens, x), dim=1)
             #x += self.pos_embedding[:, :(n + 1)]
             x = self.dropout(x)
             
             x = self.transformer1(x, mask)
             x = self.transformer2(x, mask)
             #x = self.transformer3(x, mask)
            
             

             x = x.mean(dim = 1)

             emb = self.mlp_head(x) 
             #emb = emb.unsqueeze(-1)
             #emb = emb.transpose(1,2).to("cuda")
             emb=emb.unsqueeze(2)
             result = self.asp(emb)
             result= self.asp_bn(result)

         return result




def MainModel(**kwargs):

    model =VOT_serial(
                         image_size=112,
                         patch_size=8,
                         ac_patch_size=12,
                         pad = 4,
                         dim=256,
                         depth=12,
                         heads=8,
                         mlp_dim=2048,
                         dropout=0.15,
                         emb_dropout=0.1
                     )
    return model
