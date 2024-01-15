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
                 2,
                 3,
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
class transformer(nn.Module):
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


         self.pos_embedding = nn.Parameter(torch.randn(1, 2100, dim))
         self.patch_to_embedding = nn.Linear(patch_dim, dim)
         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
         self.alpha = nn.Parameter(torch.FloatTensor(1),requires_grad=True)
        
         self.gama = nn.Parameter(torch.FloatTensor(1),requires_grad=True)
         self.dropout = nn.Dropout(emb_dropout)

         self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
         #self.transformer = Transformer(dim, 12, 8, 64, 2048, 0.15)
        
         self.pool = pool
         self.to_latent = nn.Identity()
         self.mlp_head = nn.Sequential(
             nn.LayerNorm(512),
         )
        
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
             x = self.transformer(x, mask)
             #print(x.shape)

             x = x.mean(dim = 1)

             emb = self.mlp_head(x) 
             #print(x.shape)
         return x




def MainModel(**kwargs):

    model =transformer(
                         image_size=112,
                         patch_size=8,
                         ac_patch_size=12,
                         pad = 4,
                         dim=512,
                         depth=12,
                         heads=8,
                         mlp_dim=2048,
                         dropout=0.1,
                         emb_dropout=0
                     )
    return model
