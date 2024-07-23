import math
import torch
import torch.nn as nn
import torch.fft as fft

class CrossAttentionBlock(nn.Module):
    def __init__(self, channels):
        super(CrossAttentionBlock, self).__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(32, channels)
        self.conv = nn.Conv2d(channels * 2, channels, 1)  # Single convolution layer
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x, cond):
        B, C, H, W = x.shape

        x = self.norm(x)
        cond = self.norm(cond)

        # Concatenate x and cond
        combined = torch.cat([x, cond], dim=1)  # B, 2C, H, W

        # Single convolution
        combined = self.conv(combined)  # B, C, H, W

        # Simplified attention mechanism
        attn = torch.sigmoid(combined.mean(dim=1, keepdim=True))  # B, 1, H, W

        out = attn * combined

        out = self.proj(out)

        return out + x

class FFTAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super(FFTAttention, self).__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        # Apply FFT to the input
        x_fft = fft.fft2(x)
        x_fft = torch.view_as_real(x_fft)
        
        # Reshape for attention
        b, c, h, w = x.shape
        x_fft = x_fft.view(b, c, -1)
        
        qkv = self.qkv(self.norm(x_fft))
        qkv = qkv.view(b * self.num_heads, -1, qkv.shape[2])
        
        h = self._apply_attention(qkv)
        h = h.view(b, -1, h.shape[-1])
        h = self.proj_out(h)
        
        # Reshape back and apply inverse FFT
        h = h.view(b, c, h, w)
        h = torch.view_as_complex(h)
        h = fft.ifft2(h)
        
        return h

    def _apply_attention(self, qkv):
        ch = qkv.shape[1] // 3
        q, k, v = torch.split(qkv, ch, dim=1)
        scale = 1 / (ch ** 0.25)
        weight = torch.einsum('bct,bcs->bts', q * scale, k * scale)
        weight = torch.softmax(weight, dim=-1)
        return torch.einsum('bts,bcs->bct', weight, v)

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Imported from https://github.com/openai/improved-diffusion.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x, _):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = torch.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        return torch.einsum("bts,bcs->bct", weight, v)
    
class SimpleAttentionBlock(nn.Module):
    def __init__(self, channels):
        super(SimpleAttentionBlock, self).__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(32, channels)
        self.qkv_conv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x, _):
        B, C, H, W = x.shape
        normed = self.norm(x)
        qkv = self.qkv_conv(normed)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        q = q.view(B, C, -1)  # B, C, HW
        k = k.view(B, C, -1)  # B, C, HW
        v = v.view(B, C, -1)  # B, C, HW

        attn_scores = torch.bmm(q.permute(0, 2, 1), k) / (C ** 0.5)  # B, HW, HW
        attn = torch.softmax(attn_scores, dim=-1)

        out = torch.bmm(v, attn.permute(0, 2, 1))  # B, C, HW
        out = out.view(B, C, H, W)
        out = self.proj_conv(out)

        return x + out