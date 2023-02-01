import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from timm.models.layers import to_2tuple, trunc_normal_
    

class PDAttention(nn.Module):

    def __init__(
        self, n_heads, n_head_channels, n_groups=1,
        attn_drop=0., proj_drop=0., stride=1, use_pe=False, dwc_pe=False,
        no_off=False, fixed_pe=False
    ):

        super().__init__()
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, 3, stride, 1, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )

        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_offset = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )


        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

            
    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device), 
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device)
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1)
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1) # B * g H W 2
        
        return ref

    def forward_attn_single(self, q, x):
        B, NQ, C = q.size()
        assert NQ == 1
        q = q.permute(0, 2, 1).reshape(B, C, 1, 1)
        x = x.reshape(B, C, 1, -1)

        q = self.proj_q(q)
        q = q.reshape(B * self.n_heads, self.n_head_channels, 1)
        k = self.proj_k(x).reshape(B * self.n_heads, self.n_head_channels, -1)
        v = self.proj_v(x).reshape(B * self.n_heads, self.n_head_channels, -1)
        
        attn = torch.einsum('b c m, b c n -> b m n', q, k) # B * h, HW, Ns
        attn = attn.mul(self.scale)
        
        
        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)
        
        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        
        out = out.reshape(B, C, 1, 1)
        
        y = self.proj_drop(self.proj_out(out))
        return y.reshape(B, C, -1).permute(0, 2, 1) # B, 1, C

    def forward(self, q, x, H, W, ra_type='gar', offset_range_factor=None):
        B, HW, C = x.size()
        B, NQ, C = q.size()
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        dtype, device = x.dtype, x.device
        
        offset_proj = self.proj_offset(x)
        offset_proj = einops.rearrange(offset_proj, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(offset_proj) # B * g 2 Hg Wg
        
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk
        
        if offset_range_factor is not None:
            offset_range = torch.tensor([offset_range_factor[0] / Hk, offset_range_factor[1] / Wk], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range)
        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)
            
        if self.no_off:
            offset = offset.fill(0.0)
            
        if offset_range_factor is not None:
            pos = offset + reference
        else:
            pos = (offset + reference).tanh()
        
        
        x_sampled = F.grid_sample(
            input=x.reshape(B * self.n_groups, self.n_group_channels, H, W), 
            grid=pos[..., (1, 0)], # y, x -> x, y
            mode='bilinear', align_corners=True) # B * g, Cg, Hg, Wg
        if ra_type == 'gar':
            qs = torch.chunk(q, NQ, dim=-2)
            y = [self.forward_attn_single(qs[0], x_sampled)]
            for _q, _x in zip(qs[1:], torch.split(x_sampled, 2, dim=-2)):
                y.append(self.forward_attn_single(_q, _x))
        elif ra_type == 'ggar':
            qs = torch.chunk(q, NQ, dim=-2)
            y = [self.forward_attn_single(qs[0], x_sampled)]
            y.append(self.forward_attn_single(qs[1], x_sampled))
            for _q, _x in zip(qs[1:], torch.chunk(x_sampled, NQ-1, dim=-2)):
                y.append(self.forward_attn_single(_q, _x))
        elif ra_type == 'g':
            qs = torch.chunk(q, NQ, dim=-2)
            y = []
            for _q in qs:
                y.append(self.forward_attn_single(_q, x_sampled))
        elif ra_type == 'r':
            qs = torch.chunk(q, NQ, dim=-2)
            y = []
            for _q, _x in zip(qs, torch.chunk(x_sampled, NQ, dim=-2)):
                y.append(self.forward_attn_single(_q, _x))
        elif ra_type == 'garr':
            qs = torch.chunk(q, NQ, dim=-2)
            y = [self.forward_attn_single(qs[0], x_sampled)]
            for _q, _x in zip(qs[1:3], torch.chunk(x_sampled, 2, dim=-2)):
                y.append(self.forward_attn_single(_q, _x))
            for _q, _x in zip(qs[3:], torch.chunk(x_sampled, NQ-3, dim=-2)):
                y.append(self.forward_attn_single(_q, _x))
        else:
            raise Exception("Unknown part deformable attention.")
        return y

class LayerNormProxy(nn.Module):
    
    def __init__(self, dim):
        
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')
