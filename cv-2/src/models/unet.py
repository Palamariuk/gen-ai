from ..components.components import *
import torch.nn as nn


class UNet(nn.Module):
    def __init__(
            self,
            in_channels=3,
            out_channels=3,
            base_channels=64,
            channel_mults=(1, 2, 4, 8),
            num_res_blocks=2,
            time_emb_dim=256,
            attn_resolutions=None
    ):
        super().__init__()
        if attn_resolutions is None:
            attn_resolutions = []

        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        self.initial_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        in_ch = base_channels
        self.down_skip_connections = []
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            use_attn = (i in attn_resolutions)
            self.down_blocks.append(DownBlock(in_ch, out_ch, time_emb_dim, num_res_blocks, use_attn))
            in_ch = out_ch

        self.mid_block1 = ResBlock(in_ch, in_ch, time_emb_dim)
        self.mid_attn = AttentionBlock(in_ch)
        self.mid_block2 = ResBlock(in_ch, in_ch, time_emb_dim)

        self.up_blocks = nn.ModuleList()
        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult
            use_attn = ((len(channel_mults) - 1 - i) in attn_resolutions)
            self.up_blocks.append(UpBlock(in_ch, out_ch, time_emb_dim, num_res_blocks, use_attn))
            in_ch = out_ch

        self.final_norm = nn.GroupNorm(32, in_ch)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        t = t.to(x.device)
        t_emb = self.time_embedding(t)
        x = self.initial_conv(x)

        skip_connections_all = []
        for block in self.down_blocks:
            x, skip_connections = block(x, t_emb)
            skip_connections_all.append(skip_connections)

        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        for block in self.up_blocks:
            # Pop skip connections in reverse order
            skip_connections = skip_connections_all.pop()
            x = block(x, skip_connections, t_emb)

        x = self.final_norm(x)
        x = self.final_act(x)
        return self.final_conv(x)



class UNetWithCFG(nn.Module):
    def __init__(
            self,
            in_channels=3,
            out_channels=3,
            base_channels=64,
            channel_mults=(1, 2, 4, 8),
            num_res_blocks=2,
            time_emb_dim=256,
            num_classes=10,
            cond_dim=32,
            cond_drop_prob=0.1,
            attn_resolutions=None
    ):
        super().__init__()
        if attn_resolutions is None:
            attn_resolutions = []
        self.cond_dim = cond_dim
        self.cond_drop_prob = cond_drop_prob

        # Embedding for class labels.
        self.class_embedding = nn.Embedding(num_classes, cond_dim)
        # Learned null embedding for the unconditional (dropped) condition.
        self.null_embed = nn.Parameter(torch.zeros(cond_dim))

        # Time embedding network.
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # Adjust initial convolution to accept additional cond_dim channels.
        self.initial_conv = nn.Conv2d(in_channels + cond_dim, base_channels, kernel_size=3, padding=1)

        # Downsampling path.
        self.down_blocks = nn.ModuleList()
        in_ch = base_channels
        self.down_skip_connections = []
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            use_attn = (i in attn_resolutions)
            self.down_blocks.append(DownBlock(in_ch, out_ch, time_emb_dim, num_res_blocks, use_attn))
            in_ch = out_ch

        # Bottleneck.
        self.mid_block1 = ResBlock(in_ch, in_ch, time_emb_dim)
        self.mid_attn = AttentionBlock(in_ch)
        self.mid_block2 = ResBlock(in_ch, in_ch, time_emb_dim)
        # Insert cross-attention to condition on the class embedding.
        self.cross_attn_mid = CrossAttentionBlock(in_ch, context_dim=cond_dim)

        # Upsampling path.
        self.up_blocks = nn.ModuleList()
        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult
            use_attn = ((len(channel_mults) - 1 - i) in attn_resolutions)
            self.up_blocks.append(UpBlock(in_ch, out_ch, time_emb_dim, num_res_blocks, use_attn))
            in_ch = out_ch

        self.final_norm = nn.GroupNorm(32, in_ch)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1)

    def forward(self, x, t, cond=None):
        B, _, H, W = x.shape
        # Classifier-free guidance: randomly drop the class condition during training.
        if self.training:
            if cond is not None:
                drop_mask = torch.rand(B, device=x.device) < self.cond_drop_prob
                cond_emb = self.class_embedding(cond)
                null_emb = self.null_embed.unsqueeze(0).expand(B, -1)
                cond_emb = torch.where(drop_mask.unsqueeze(1), null_emb, cond_emb)
            else:
                cond_emb = self.null_embed.unsqueeze(0).expand(B, -1)
        else:
            # At inference, if no class label is provided, use the unconditional embedding.
            cond_emb = self.null_embed.unsqueeze(0).expand(B, -1) if cond is None else self.class_embedding(
                cond)

        # Create a spatial conditioning map by expanding the class embedding.
        cond_map = cond_emb.unsqueeze(-1).unsqueeze(-1).expand(B, self.cond_dim, H, W)
        # Concatenate along the channel dimension.
        x = torch.cat([x, cond_map], dim=1)

        # Compute time embedding.
        t_emb = self.time_embedding(t.to(x.device))
        x = self.initial_conv(x)

        skip_connections_all = []
        for block in self.down_blocks:
            x, skips = block(x, t_emb)
            skip_connections_all.append(skips)

        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        # Inject class conditioning via cross-attention.
        x = self.cross_attn_mid(x, cond_emb)

        for block in self.up_blocks:
            skips = skip_connections_all.pop()
            x = block(x, skips, t_emb)

        x = self.final_norm(x)
        x = self.final_act(x)
        return self.final_conv(x)
