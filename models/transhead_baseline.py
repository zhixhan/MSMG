import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from collections import OrderedDict

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.dat import DAttention
from utils.mask import get_mask_box, exchange_token
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, spacial_size=14, chunk_attn=False):
        super().__init__()
        self.norm1 = norm_layer(dim) #if not chunk_attn else nn.GroupNorm(2, dim)
    
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class HATHead(nn.Module):

    def __init__(self, feature_names=["feat_res2", "feat_res3", "feat_res4"], in_channels=[256, 512, 1024], depth=2, num_heads=8, embed_dim=384, sr_ratios=[2, 1, 1], spacial_size=14):
        super(HATHead, self).__init__()
        self.proj = nn.ModuleDict([
            ['feat_res2', nn.Conv2d(256, embed_dim, kernel_size=1, stride=1, bias=False)], 
            ['feat_res3', nn.Conv2d(512, embed_dim, kernel_size=1, stride=1, bias=False)],
            ['feat_res4', nn.Conv2d(1024, embed_dim, kernel_size=1, stride=1, bias=False)]
            ])
        self.output = nn.ModuleDict([
            ['feat_res2', nn.Conv2d(embed_dim, 512, kernel_size=1, stride=1, bias=False)], 
            ['feat_res3', nn.Conv2d(embed_dim, 1024, kernel_size=1, stride=1, bias=False)],
            ['feat_res4', nn.Conv2d(embed_dim, 2048, kernel_size=1, stride=1, bias=False)]
            ])
        self.blocks = nn.ModuleList([Block(
            dim=embed_dim, num_heads=num_heads, sr_ratio=sr_ratios[j], spacial_size=spacial_size[0])
        for j in range(depth)])
        self.apply(self._init_weights)
        self.embed_dim = embed_dim


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    
    def forward(self, x):
        feats, feats_location = [], []
        for res, feat in x.items():
            feat = self.proj[res](feat)
            B, C, H, W = feat.size()
            original = feat.clone()
            feat = feat.view(B, self.embed_dim, -1).permute(0, 2, 1)
            for block in self.blocks:
                feat = block(feat, H, W)
            feat = feat.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            feat += original
            feat = self.output[res](feat)
            feats.append(["trans_"+res, F.adaptive_max_pool2d(feat, 1)])
            feats_location.append(["location_"+res, feat])
        return OrderedDict(feats), feat
 

def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


