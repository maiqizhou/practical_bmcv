import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

from .attention import AttentionBlock, CrossAttentionBlock

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None, dropout=0.1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels))
        self.time_emb = nn.Sequential(
            nn.ReLU(),
            nn.Linear(time_emb_dim * 4, out_channels)) if time_emb_dim else nn.Identity()

    def forward(self, x, emb=None):
        out = self.relu(self.norm1(self.conv1(x)))
        if emb is not None:
            time_emb = self.time_emb(emb)
            time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)
            out = out + time_emb
        out = self.dropout(out)
        out = self.relu(self.norm2(self.conv2(out)))
        return out + self.shortcut(x)

class TimeEmbedding(nn.Module):
    def __init__(self, emb_dim, device):
        super(TimeEmbedding, self).__init__()
        self.emb_dim = emb_dim
        self.device = device
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim * 4, emb_dim * 4))

    def sinusoidal_embd(self, t):
        half_dim = self.emb_dim // 2
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
    
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, device, time_emb_dim=256, cond_dim=3, num_channels=[64, 128, 256, 512], dropout=0.1): 
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.cond_dim = cond_dim
        self.num_channels = num_channels
        self.device = device
        
        self.time_embedding = TimeEmbedding(time_emb_dim, device)

        self.cond_encoder = nn.ModuleList()

        self.down_blocks = nn.ModuleList()
        in_ch = in_channels + cond_dim
        for out_ch in num_channels:
            self.down_blocks.append(ResBlock(in_ch, out_ch, time_emb_dim, dropout))
            self.cond_encoder.append(ResBlock(in_ch, out_ch, None, dropout))
            if out_ch != num_channels[-1]:
                self.cond_encoder.append(CrossAttentionBlock(out_ch))
            in_ch = out_ch
        
        self.bottleneck = nn.ModuleList()
        self.bottleneck.append(ResBlock(num_channels[-1], num_channels[-1], time_emb_dim, dropout))
        self.bottleneck.append(AttentionBlock(num_channels[-1]))
        self.bottleneck.append(ResBlock(num_channels[-1], num_channels[-1], time_emb_dim, dropout))

        self.up_blocks = nn.ModuleList()
        in_ch = num_channels[-1]
        for out_ch in reversed(num_channels):
            self.up_blocks.append(ResBlock(in_ch + out_ch, out_ch, time_emb_dim, dropout))
            in_ch = out_ch

        self.final_conv = nn.Conv2d(num_channels[0], out_channels, kernel_size=1)

        self.to(device)


    def forward(self, x, t, cond):
        t_emb = self.time_embedding(t)

        if self.cond_dim > 0:
            x = torch.cat([x, cond], dim=1)

        down_outputs = []
        out = x
        cond_out = cond
        for i, block in enumerate(self.down_blocks):
            out = block(out, t_emb)
            down_outputs.append(out)
            out = F.max_pool2d(out, kernel_size=2)

            cond_out = self.cond_encoder[2 * i](cond_out) # Encoder Block
            cond_out = F.max_pool2d(cond_out, kernel_size=2)
            
            if i != len(self.down_blocks) - 1:
                cond_out = self.cond_encoder[2 * i + 1](cond_out, out) # Cross Attention

        
        out = out + cond_out

        bottleneck = self.bottleneck[0](out, t_emb)
        bottleneck = self.bottleneck[1](bottleneck, None)
        bottleneck = self.bottleneck[2](bottleneck, t_emb)

        for block in self.up_blocks:
            bottleneck = F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=True)
            bottleneck = torch.cat([bottleneck, down_outputs.pop()], dim=1)
            bottleneck = block(bottleneck, t_emb)

        out = self.final_conv(bottleneck)
        return out

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
    model = UNet(in_channels=3, out_channels=3, device=device, time_emb_dim=128, num_channels=[64, 128, 256, 512])
    print(model)
    calculate_parameter_size(model)