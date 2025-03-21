
import torch
from torch import nn, einsum
from einops import rearrange
import torch.nn.functional as F


class PreNorm(nn.Module):
    """Layer normalization before processing"""

    def __init__(self, embed_dim, processor):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.processor = processor

    def forward(self, tensor, **params):
        return self.processor(self.layer_norm(tensor), **params)


class FeedForward(nn.Module):
    """Multilayer perceptron with GELU activation"""

    def __init__(self, dim, hidden_dim, dropout=0.):
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


class MultiHeadAttention(nn.Module):
    """Scaled dot-product attention mechanism"""

    def __init__(self, embed_dim, num_heads=8, head_dim=64, p_drop=0.):
        super().__init__()
        inner_dim = head_dim * num_heads
        self.needs_projection = not (num_heads == 1 and head_dim == embed_dim)

        self.num_heads = num_heads
        self.scaling_factor = head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.qkv_proj = nn.Linear(embed_dim, inner_dim * 3, bias=False)

        self.output_proj = nn.Sequential(
            nn.Linear(inner_dim, embed_dim),
            nn.Dropout(p_drop)
        ) if self.needs_projection else nn.Identity()

    def forward(self, input_tensor):
        batch_size, seq_len, _ = input_tensor.shape
        qkv = self.qkv_proj(input_tensor).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)

        similarity = einsum('b h i d, b h j d -> b h i j', q, k) * self.scaling_factor
        attention_weights = self.softmax(similarity)

        output = einsum('b h i j, b h j d -> b h i d', attention_weights, v)
        output = rearrange(output, 'b h n d -> b n (h d)')
        return self.output_proj(output)


class AttentionStack(nn.Module):
    """Stack of transformer layers"""

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, MultiHeadAttention(dim, num_heads=heads, head_dim=dim_head, p_drop=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class DSConv(nn.Module):
    """Depthwise separable convolution"""

    def __init__(self, in_ch, out_ch, kernel_size, padding=0, depth_mult=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch,
            in_ch * depth_mult,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_ch
        )
        self.pointwise = nn.Conv2d(in_ch * depth_mult, out_ch, kernel_size=1)

    def forward(self, feature_map):
        return self.pointwise(self.depthwise(feature_map))


class DualDSConv(nn.Module):
    """Double depthwise separable convolution"""

    def __init__(self, in_ch, out_ch, mid_ch=None, depth_mult=1):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.two_stage_conv = nn.Sequential(
            DSConv(in_ch, mid_ch, 3, padding=1, depth_mult=depth_mult),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            DSConv(mid_ch, out_ch, 3, padding=1, depth_mult=depth_mult),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, feature_map):
        return self.two_stage_conv(feature_map)


class DownsampleDS(nn.Module):
    """Downsampling with maxpool and dual convolution"""

    def __init__(self, in_ch, out_ch, depth_mult=1):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DualDSConv(in_ch, out_ch, depth_mult=depth_mult)
        )

    def forward(self, feature_map):
        return self.down_conv(feature_map)


class UpsampleDS(nn.Module):
    """Upsampling with bilinear interpolation"""

    def __init__(self, in_ch, out_ch, bilinear=True, depth_mult=1):
        super().__init__()
        if bilinear:
            self.upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv_block = DualDSConv(
                in_ch,
                out_ch,
                mid_ch=in_ch // 2,
                depth_mult=depth_mult
            )
        else:
            self.upscale = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv_block = DualDSConv(in_ch, out_ch, depth_mult=depth_mult)

    def forward(self, main_feat, skip_feat):
        main_feat = self.upscale(main_feat)
        h_diff = skip_feat.size()[2] - main_feat.size()[2]
        w_diff = skip_feat.size()[3] - main_feat.size()[3]
        main_feat = F.pad(main_feat, [w_diff // 2, w_diff - w_diff // 2,
                                      h_diff // 2, h_diff - h_diff // 2])
        merged = torch.cat([skip_feat, main_feat], dim=1)
        return self.conv_block(merged)


class OutputProjection(nn.Module):
    """Final 1x1 convolution for output"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.final_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, feature_map):
        return self.final_conv(feature_map)


class TrafficUNet(nn.Module):
    """Hybrid CNN-Transformer architecture for segmentation"""

    def __init__(self, in_ch, out_ch, depth_mult=2, bilinear=True, hidden = 512):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.bilinear = bilinear

        # Encoder path
        self.init_conv = DualDSConv(in_ch, hidden//8, depth_mult=depth_mult)
        self.down1 = DownsampleDS(hidden//8, hidden//4, depth_mult)
        self.down2 = DownsampleDS(hidden//4, hidden//2, depth_mult)
        self.down3 = DownsampleDS(hidden//2, hidden, depth_mult)
        self.down4 = DownsampleDS(hidden, hidden*2 // (2 if bilinear else 1), depth_mult)

        # Transformer blocks
        self.transformer1 = AttentionStack(hidden//8, 2, 2, 2, 64)
        self.transformer2 = AttentionStack(hidden//4, 2, 2, 2, 64)
        self.transformer3 = AttentionStack(hidden//2, 2, 2, 2, 64)
        self.transformer4 = AttentionStack(hidden, 2, 2, 2, 64)
        self.transformer5 = AttentionStack(hidden, 2, 2, 2, 64)

        # Decoder path
        # self.up21 = UpsampleDS(768, 512 // (2 if bilinear else 1), bilinear, depth_mult)
        self.up1 = UpsampleDS(hidden*2, hidden // (2 if bilinear else 1), bilinear, depth_mult)
        self.up2 = UpsampleDS(hidden, hidden//2 // (2 if bilinear else 1), bilinear, depth_mult)
        self.up3 = UpsampleDS(hidden//2, hidden//4 // (2 if bilinear else 1), bilinear, depth_mult)
        self.up4 = UpsampleDS(hidden//4, hidden//8, bilinear, depth_mult)
        self.final_proj = OutputProjection(hidden//8, out_ch)

    def forward(self, x):
        # Encoder processing
        x1 = self.init_conv(x)
        x1_t = self._apply_transformer(x1, self.transformer1)

        x2 = self.down1(x1)
        x2_t = self._apply_transformer(x2, self.transformer2)

        x3 = self.down2(x2)
        x3_t = self._apply_transformer(x3, self.transformer3)

        x4 = self.down3(x3)
        x4_t = self._apply_transformer(x4, self.transformer4)

        x5 = self.down4(x4)
        x5_t = self._apply_transformer(x5, self.transformer5)

        # Decoder processing
        # d = self.up21(x4Att, x3Att)  #for taxicq/bikedc
        d = self.up1(x5_t, x4_t)
        d = self.up2(d, x3_t)
        d = self.up3(d, x2_t)
        d = self.up4(d, x1_t)
        return self.final_proj(d)

    def _apply_transformer(self, feat, transformer):
        """Reshape features for transformer processing"""
        batch, ch, h, w = feat.shape
        reshaped = feat.view(batch, h * w, ch)
        processed = transformer(reshaped)
        return processed.view(batch, ch, h, w)
