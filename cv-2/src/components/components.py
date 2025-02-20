import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, t):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        # Add time embedding (broadcast to spatial dims)
        t_emb = self.time_mlp(t)[:, :, None, None]
        h = h + t_emb
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        q = self.q(x_norm).reshape(b, c, -1).permute(0, 2, 1)  # shape: (b, hw, c)
        k = self.k(x_norm).reshape(b, c, -1)  # shape: (b, c, hw)
        v = self.v(x_norm).reshape(b, c, -1).permute(0, 2, 1)  # shape: (b, hw, c)
        attn = torch.bmm(q, k) * (c ** -0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.bmm(attn, v).permute(0, 2, 1).reshape(b, c, h, w)
        out = self.proj_out(out)
        return x + out


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, num_res_blocks, use_attn=False):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResBlock(in_channels if i == 0 else out_channels, out_channels, time_emb_dim)
            for i in range(num_res_blocks)
        ])
        self.use_attn = use_attn
        if use_attn:
            self.attn = AttentionBlock(out_channels)
        self.downsample = Downsample(out_channels)

    def forward(self, x, t):
        skip_connections = []
        for block in self.res_blocks:
            x = block(x, t)
            skip_connections.append(x)
        if self.use_attn:
            x = self.attn(x)
        x_down = self.downsample(x)
        return x_down, skip_connections


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, num_res_blocks, use_attn=False):
        super().__init__()
        self.upsample = Upsample(in_channels)
        self.res_blocks = nn.ModuleList([
            ResBlock(in_channels + out_channels if i == 0 else out_channels, out_channels, time_emb_dim)
            for i in range(num_res_blocks)
        ])
        self.use_attn = use_attn
        if use_attn:
            self.attn = AttentionBlock(out_channels)

    def forward(self, x, skip_connections, t):
        x = self.upsample(x)
        skip = skip_connections.pop()
        x = torch.cat([x, skip], dim=1)
        for block in self.res_blocks:
            x = block(x, t)
        if self.use_attn:
            x = self.attn(x)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, channels, context_dim, heads=4, dropout=0.0):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=heads, dropout=dropout, batch_first=True)
        self.k_proj = nn.Linear(context_dim, channels)
        self.v_proj = nn.Linear(context_dim, channels)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x, context):
        B, C, H, W = x.shape
        x_norm = self.norm(x)
        # Flatten spatially: (B, H*W, C)
        q = self.q_proj(x_norm).reshape(B, C, -1).permute(0, 2, 1)
        # If context is a single token per sample, unsqueeze to (B, 1, context_dim)
        if context.dim() == 2:
            context = context.unsqueeze(1)
        k = self.k_proj(context)
        v = self.v_proj(context)
        attn_out, _ = self.mha(q, k, v)
        attn_out = attn_out.permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.out_proj(attn_out)
