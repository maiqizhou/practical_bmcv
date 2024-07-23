import math

import torch
import torch.nn as nn
from torch.nn import functional as F

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.

    """

    def __init__(self, channels, use_conv=True, out_channel=None):
        super().__init__()
        self.channels = channels
        self.out_channel = out_channel or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channel, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """

    def __init__(self, channels, use_conv=True, out_channel=None):
        super().__init__()
        self.channels = channels
        self.out_channel = out_channel or channels
        self.use_conv = use_conv
        stride = 2
        if use_conv:
            self.op = nn.Conv2d(
                self.channels, self.out_channel, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channel
            self.op = nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)
    
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
        self.norm = groupNorm(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x, *_):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

def groupNorm(channels):
    return nn.GroupNorm(min(32, channels), channels)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=0.1, time_emb_dim=None, cond_dim=None):
        super(ResBlock, self).__init__()
        self.layers_in = nn.Sequential(
            groupNorm(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            groupNorm(out_channels),)
        
        if time_emb_dim is not None:
            self.time_embd = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels * 2),
            )

        if cond_dim is not None:
            self.cond_embd = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cond_dim, out_channels),
            )

        self.layers_out = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
        )

        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, time_embd=None, cond=None):
        h = self.layers_in(x)
        if time_embd is not None:
            # Scale and shift
            embd = self.time_embd(time_embd)
            embd = embd.unsqueeze(-1).unsqueeze(-1)
            assert embd.shape == (h.shape[0], h.shape[1] * 2, 1, 1)
            embd = embd.chunk(2, dim=1)
            h = h * (1 + embd[0]) + embd[1]
        if cond is not None:
            # Scale
            cond = self.cond_embd(cond)
            cond = cond.unsqueeze(-1).unsqueeze(-1)
            assert cond.shape == (h.shape[0], h.shape[1], 1, 1)
            h = h * (1 + cond)
        h = self.layers_out(h)

        return h + self.shortcut(x)
    
class TimeEmbedding(nn.Module):
    def __init__(self, base_dim, time_embd_dim, device):
        super(TimeEmbedding, self).__init__()
        self.base_dim = base_dim
        self.device = device
        self.mlp = nn.Sequential(
            nn.Linear(base_dim, time_embd_dim),
            nn.ReLU(inplace=True),
            nn.Linear(time_embd_dim, time_embd_dim))

    def sinusoidal_embd(self, t):
        half_dim = self.base_dim // 2
        emb = torch.log(torch.tensor(10000.0, device=self.device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=self.device) * -emb)
        t = t.unsqueeze(1) if t.dim() == 1 else t
        emb = t * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=1)
        return emb
    
    def forward(self, t):
        emb = self.sinusoidal_embd(t)
        emb = self.mlp(emb)
        return emb