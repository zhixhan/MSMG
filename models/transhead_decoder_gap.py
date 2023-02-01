import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from collections import OrderedDict
from copy import deepcopy

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


class CrossScaleAttention(nn.Module):
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
        

    def forward(self, x, H=None, W=None, cross_kv=None):
       
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        pre_kv = cross_kv if cross_kv is not None else x
        if self.sr_ratio > 1:
            pre_kv_ = pre_kv.permute(0, 2, 1).reshape(B, C, H, W)
            pre_kv_ = self.sr(pre_kv_).reshape(B, C, -1).permute(0, 2, 1)
            pre_kv_ = self.norm(pre_kv_)
            kv = self.kv(pre_kv_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(pre_kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

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
        self.chunk_attn = chunk_attn
        
        if chunk_attn:
            self.self_attn1 = CrossScaleAttention(dim // 2, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
            self.self_attn2 = CrossScaleAttention(dim // 2, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio )
            self.cross_attn1 = CrossScaleAttention(dim // 2, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio )
            self.cross_attn2 = CrossScaleAttention(dim // 2, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio )
            #self.attn = CrossScaleAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio )
            self.norm1_s1 = norm_layer(dim // 2)
            self.norm1_s2 = norm_layer(dim // 2)
            self.norm1_c1 = norm_layer(dim // 2)
            self.norm1_c2 = norm_layer(dim // 2)
        else:
            self.norm1 = norm_layer(dim)
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W, cross_kv=None):
        if not self.chunk_attn:
            x = x + self.drop_path(self.attn(self.norm1(x), H, W))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x_pre, x_cur = torch.chunk(x, 2, dim=-1)
            # self attn
            x_pre = x_pre + self.drop_path(self.self_attn1(self.norm1_s1(x_pre), H, W)) 
            x_cur = x_cur + self.drop_path(self.self_attn2(self.norm1_s2(x_cur), H, W))
            # cross attn
            x_pre_cross, x_cur_cross = x_pre.clone(), x_cur.clone()
            #x_pre = x_pre + self.drop_path(self.cross_attn1(self.norm1_c1(x_pre), H, W, cross_kv=x_cur_cross))
            x_cur = x_cur + self.drop_path(self.cross_attn2(self.norm1_c2(x_cur), H, W, cross_kv=x_pre_cross))
            x = torch.cat([x_pre, x_cur], dim=-1)
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class DecoderBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, spacial_size=14, chunk_attn=False):
        super().__init__()
        self.norm0 = norm_layer(dim) 
        self.norm1 = norm_layer(dim)
        
        self.attn = CrossScaleAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.cross_attn = CrossScaleAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.querys_num = 8
        self.part_querys = nn.Embedding(self.querys_num, dim)

    def forward(self, x, H, W):
        B, C, H, W = x.size()
        x = x.view(B, C, -1).permute(0, 2, 1)
        part_querys = self.part_querys.weight.unsqueeze(0).expand(B, self.querys_num, C)
        part_querys = part_querys + self.drop_path(self.norm0(self.attn(part_querys)))
        querys = torch.chunk(part_querys, self.querys_num, dim=1)
        
        part_feats = [self.cross_attn(querys[0], cross_kv=x)] # B, 1, Dim x B, 196, Dim
        for q, kv in zip(querys[1:], torch.chunk(x, 7, dim=1)):
            part_feats.append(self.cross_attn(q, cross_kv=kv)) # B, 1, Dim x B, 28, Dim
        part_feats = torch.cat(part_feats, dim=1)
        x = part_querys + self.drop_path(self.norm1(part_feats))
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x


class HATHead(nn.Module):

    def __init__(self, feature_names=["feat_res2", "feat_res3", "feat_res4"], in_channels=[256, 512, 1024], depth=2, num_heads=8, embed_dim=384, sr_ratios=[1, 1, 1], spacial_size=14, part_feats=7):
        super(HATHead, self).__init__()
        self.proj = nn.ModuleDict([
            ['feat_res2', nn.Conv2d(256, embed_dim , kernel_size=3, stride=3, bias=False)],
            ['feat_res3', nn.Conv2d(512, embed_dim // 2, kernel_size=2, stride=2, bias=False)],
            ['feat_res4', nn.Conv2d(1024, embed_dim // 2, kernel_size=1, stride=1, bias=False)]
            ])
        self.bridge = nn.ModuleDict([
            ['feat_res2', nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1, stride=1, bias=False)],
            ['feat_res3', nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1, stride=1, bias=False)],
            ])
        self.output = nn.ModuleDict([
            ['feat_res4', nn.Conv2d(embed_dim, 2048, kernel_size=1, stride=1, bias=False)],
            ['part_feat4', nn.Linear(embed_dim, 2048)]
            ])
        
        #assert len(sr_ratios) == depth, " Error spacial reduction size"
        self.blocks = nn.ModuleList([Block(
            dim=embed_dim, num_heads=num_heads, sr_ratio=sr_ratios[j], spacial_size=spacial_size[-1])
            for j in range(depth)])
        #self.blocks2 = nn.ModuleList([Block(
        #    dim=embed_dim, num_heads=num_heads, sr_ratio=sr_ratios[j], spacial_size=spacial_size[-1])
        #    for j in range(depth)])
        self.apply(self._init_weights)
        self.embed_dim = embed_dim
        self.part_feats = part_feats
        self.decoder = DecoderBlock(dim=embed_dim, num_heads=num_heads, sr_ratio=1, spacial_size=spacial_size[-1])

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
        proj_feats = [(res, self.proj[res](feat)) for res, feat in x.items()]
        
        res2, feat2 = proj_feats[0]
        B, C, H, W = feat2.size()
        original2 = feat2.clone()
        feat2 = feat2.view(B, self.embed_dim, -1).permute(0, 2, 1)
        for block in self.blocks:
            feat2 = block(feat2, H, W)
        feat2 = feat2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        feat2 += original2
        feat2 = self.bridge[res2](feat2)
        
        res3, feat3 = proj_feats[1]
        feat3 = torch.cat([feat2, feat3], dim=1)
        B, C, H, W = feat3.size()
        original3 = feat3.clone()
        feat3 = feat3.view(B, self.embed_dim, -1).permute(0, 2, 1)
        for block in self.blocks:
            feat3 = block(feat3, H, W)
        feat3 = feat3.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        feat3 += original3
        feat3 = self.bridge[res3](feat3)
       
        res4, feat4 = proj_feats[2]
        feat4 = torch.cat([feat3, feat4], dim=1)
        B, C, H, W = feat4.size()
        original4 = feat4.clone()
        feat4 = feat4.view(B, self.embed_dim, -1).permute(0, 2, 1)
        for block in self.blocks:
            feat4 = block(feat4, H, W)
        feat4 = feat4.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        feat4 += original4
        
        part_feat4 = self.decoder(feat4.clone(), H, W)
        part_feat4 = self.output['part_feat4'](part_feat4)
        part_feat4 = torch.chunk(part_feat4, self.part_feats + 1, dim=1)
        feats.append(["trans_"+res4, part_feat4[0].squeeze(1)])
        feats_location.append(["location_"+res4, part_feat4[0].squeeze(1)])
        for i, pf in enumerate(part_feat4[1:]):
            feats.append(["part_trans_"+str(i), pf.squeeze(1)])
        
        return OrderedDict(feats), OrderedDict(feats_location)
    
    '''
    def forward(self, x):
        feats = []
        proj_feats = [(res, self.proj[res](feat)) for res, feat in x.items()]
        
        res2, feat2 = proj_feats[0]
        if self.token_mixup:
            feature_mask = self.get_mask_box(feat2)
        B, C, H, W = feat2.size()
        #original2 = feat2.clone()
        feat2 = feat2.view(B, self.embed_dim, -1).permute(0, 2, 1)
        for block in self.blocks:
            if self.token_mixup:
                feat2 = self.exchange_token(feat2, feature_mask)
            feat2 = block(feat2, H, W)
        feat2 = feat2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        #feat2 += original2
        feat2 = self.output[res2](feat2)
        
        # cross attn
        res3, feat3 = proj_feats[1]
        feat3 = torch.cat([feat2, feat3], dim=1)
        B, C, H, W = feat3.size()
        #original3 = feat3.clone()
        feat3 = feat3.view(B, self.embed_dim, -1).permute(0, 2, 1)
        for block in self.blocks:
            if self.token_mixup:
                feat3 = self.exchange_token(feat3, feature_mask)
            feat3 = block(feat3, H, W)
        feat3 = feat3.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        #feat3 += original3
        feat3 = self.output[res3](feat3)
       
       
        # self attn
        res4, feat4 = proj_feats[2]
        feat4 = torch.cat([feat3, feat4], dim=1)
        B, C, H, W = feat4.size()
        #original4 = feat4.clone()
        feat4 = feat4.view(B, self.embed_dim, -1).permute(0, 2, 1)
        for block in self.blocks:
            if self.token_mixup:
                feat4 = self.exchange_token(feat4, feature_mask)
            feat4 = block(feat4, H, W)
        feat4 = feat4.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        #feat4 += original4
        feat4 = self.output[res4](feat4)
        feat4 = F.adaptive_max_pool2d(feat4, 1)
        feats.append(["trans_"+res4, feat4])
        return OrderedDict(feats)
    '''  

def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


