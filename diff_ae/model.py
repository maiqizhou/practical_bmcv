import torch
import torch.nn as nn

import torch.nn.functional as F

from diff_ae.blocks import Upsample, Downsample, TimeEmbedding, ResBlock, AttentionBlock, groupNorm

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, device, time_emb_dim=512, cond_dim=512, num_channels=[64, 64, 128, 128, 256, 256, 512, 512], dropout=0.1): 
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.cond_dim = cond_dim
        self.num_channels = num_channels
        self.dropout = dropout
        self.device = device

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.time_embedding = TimeEmbedding(num_channels[0], time_emb_dim, device)
        self.first = nn.Conv2d(in_channels, num_channels[0], kernel_size=3, padding=1)

        for i in range(1, len(num_channels)):
            ch_in = num_channels[i-1]
            ch_out = num_channels[i]
            self.encoders.append(ResBlock(ch_in, ch_out, dropout=dropout, time_emb_dim=time_emb_dim, cond_dim=cond_dim))
            self.decoders.insert(0, ResBlock(2 * ch_out, ch_in, dropout=dropout, time_emb_dim=time_emb_dim, cond_dim=cond_dim))

        self.attn = AttentionBlock(num_channels[-1])

        self.final = nn.Conv2d(num_channels[0], out_channels, kernel_size=3, padding=1)
        # self.tanh = nn.Tanh()

        self.to(device)

    def forward(self, x, t=None, cond=None):
        h = x
        time_embd = self.time_embedding(t)
        enc_outs = []
        h = self.first(h)
        for encoder in self.encoders:
            h = encoder(h, time_embd, cond)
            enc_outs.append(h)
            h = F.max_pool2d(h, 2)

        h = self.attn(h, None)

        for decoder in self.decoders:
            h = F.interpolate(h, scale_factor=2, mode='bilinear', align_corners=False)
            h = torch.cat([h, enc_outs.pop()], dim=1)
            h = decoder(h, time_embd, cond)

        h = self.final(h)

        return h
    
class SemanticEncoder(nn.Module):
    def __init__(self, in_channels, z_dim, device, use_time_embd=True, num_channels=[64, 64, 128, 128, 256, 256, 512, 512], dropout=0.1):
        super(SemanticEncoder, self).__init__()
        self.in_channels = in_channels
        self.z_dim = z_dim
        self.num_channels = num_channels
        self.dropout = dropout
        self.device = device
        self.time_embed_dim = num_channels[0] * 4
        if use_time_embd:
            self.time_embedding = TimeEmbedding(num_channels[0], self.time_embed_dim, device)

        self.encoders = nn.ModuleList()
        self.first = nn.Conv2d(in_channels, num_channels[0], kernel_size=3, padding=1)

        for i in range(1, len(num_channels)):
            in_channels = num_channels[i-1]
            self.encoders.append(ResBlock(in_channels, num_channels[i], dropout=dropout, time_emb_dim=self.time_embed_dim))

        self.bottleneck = ResBlock(num_channels[-1], num_channels[-1], dropout=dropout, time_emb_dim=self.time_embed_dim)

        self.out = nn.Sequential(
            groupNorm(num_channels[-1]),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(num_channels[-1], z_dim, kernel_size=1),
            nn.Flatten())

        self.attn = AttentionBlock(num_channels[-1])

        self.to(device)

    def forward(self, x, t=None):
        h = x
        h = self.first(h)
        if t is not None:
            time_embd = self.time_embedding(t)
        for encoder in self.encoders:
            h = encoder(h, time_embd)
            h = F.max_pool2d(h, 2)

        h = self.attn(h, None)
        h = self.bottleneck(h, time_embd=time_embd)
        h = self.out(h)

        return h
        
    
def calculate_parameter_size(model):
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    
    param_size_bytes = total_params * 4
    param_size_kb = param_size_bytes / 1024
    param_size_mb = param_size_kb / 1024

    print(f"Total Parameters: {total_params}")
    print(f"Total Parameter Size: {param_size_bytes} bytes")
    print(f"Total Parameter Size: {param_size_kb:.2f} KB")
    print(f"Total Parameter Size: {param_size_mb:.2f} MB")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = UNet(in_channels=3, out_channels=3, device=device)
    encoder = SemanticEncoder(in_channels=3, z_dim=512, device=device)
    # print(model)
    # calculate_parameter_size(model)

    # Test forward pass
    x = torch.randn(1, 3, 256, 256).to(device)
    t = torch.randn(1, 1).to(device)
    cond = torch.randn(1, 3, 256, 256).to(device)
    cond_latent = encoder(cond, t)
    out = model(x, t, cond_latent)
    print(out.shape)
    print(calculate_parameter_size(model))