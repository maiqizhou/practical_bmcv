import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.profiler import profile, record_function, ProfilerActivity

class TimeEmbedding(nn.Module):
    def __init__(self, emb_dim, device):
        super(TimeEmbedding, self).__init__()
        self.emb_dim = emb_dim
        self.device = device

    def forward(self, t):
        half_dim = self.emb_dim // 2
        emb = torch.log(torch.tensor(10000.0, device=self.device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=self.device) * -emb)
        t = t.unsqueeze(1) if t.dim() == 1 else t
        emb = t * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=1)
        return emb

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.in_channels = in_channels
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out
    
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h x y d', h=self.heads), qkv)

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum('b h x y d, b h x y e -> b h d e', k, v)

        out = torch.einsum('b h d e, b h x y d -> b h x y e', context, q)
        out = rearrange(out, 'b h x y d -> b (h d) x y', h=self.heads)
        return self.to_out(out)
    
class CondBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CondBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.relu(self.norm2(self.conv2(out)))
        return out

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super(Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_emb = nn.Linear(time_emb_dim, out_channels)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, t, cond):
        t_emb = self.time_emb(t)
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)
        out = self.relu(self.norm1(self.conv1(x)) + t_emb)
        out = self.relu(self.norm2(self.conv2(out)) + t_emb)
        return out
    
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, device, time_emb_dim=128, cond_dim=3, num_channels=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.cond_dim = cond_dim
        self.num_channels = num_channels
        self.device = device
        
        self.time_embedding = TimeEmbedding(time_emb_dim, device)

        self.down_blocks = nn.ModuleList()
        in_ch = in_channels + cond_dim
        for out_ch in num_channels[:-1]:
            self.down_blocks.append(Block(in_ch, out_ch, time_emb_dim))
            in_ch = out_ch
        
        self.bottleneck = Block(num_channels[-2], num_channels[-1], time_emb_dim)

        self.up_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        in_ch = num_channels[-1]
        for out_ch in reversed(num_channels[:-1]):
            self.up_blocks.append(Block(in_ch + out_ch, out_ch, time_emb_dim))
            self.attn_blocks.append(LinearAttention(out_ch))
            in_ch = out_ch

        self.final_conv = nn.Conv2d(num_channels[0], out_channels, kernel_size=1)

        self.to(device)


    def forward(self, x, t, cond):
        t_emb = self.time_embedding(t)

        x = torch.cat([x, cond], dim=1)

        down_outputs = []
        out = x
        for down_block in self.down_blocks:
            out = down_block(out, t_emb, None)
            down_outputs.append(out)
            out = F.max_pool2d(out, kernel_size=2)

        bottleneck = self.bottleneck(F.max_pool2d(down_outputs[-1], kernel_size=2), t_emb, None)

        for i in range(len(self.up_blocks)):
            bottleneck = F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=True)
            bottleneck = torch.cat([bottleneck, down_outputs[-(i+1)]], dim=1)
            bottleneck = self.up_blocks[i](bottleneck, t_emb, None)
            if i < len(self.attn_blocks):
                bottleneck = self.attn_blocks[i](bottleneck)

        out = self.final_conv(bottleneck)
        return out
    

def test_unet_performance(model, device, input_shape=(1, 3, 128, 128), cond_shape=(1, 3, 128, 128), time_steps=10):
    x = torch.randn(input_shape, device=device)
    cond = torch.randn(cond_shape, device=device)
    t = torch.tensor([time_steps], device=device)

    for _ in range(10):
        _ = model(x, t, cond)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            # Encoding
            t_emb = model.time_embedding(t)
            x_concat = torch.cat([x, cond], dim=1)
            down_outputs = []
            out = x_concat
            for i, down_block in enumerate(model.down_blocks):
                with record_function(f"down_block_{i}"):
                    out = down_block(out, t_emb, None)
                    down_outputs.append(out)
                    out = F.max_pool2d(out, kernel_size=2)

            with record_function("bottleneck"):
                bottleneck = model.bottleneck(F.max_pool2d(down_outputs[-1], kernel_size=2), t_emb, None)

            for i in range(len(model.up_blocks)):
                with record_function(f"up_block_{i}"):
                    bottleneck = F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=True)
                    bottleneck = torch.cat([bottleneck, down_outputs[-(i+1)]], dim=1)
                    bottleneck = model.up_blocks[i](bottleneck, t_emb, None)
                    if i < len(model.attn_blocks):
                        bottleneck = model.attn_blocks[i](bottleneck)

            with record_function("final_conv"):
                out = model.final_conv(bottleneck)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

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
    model = UNet(in_channels=3, out_channels=3, device=device, time_emb_dim=128, num_channels=[64, 128, 256, 512])
    
    calculate_parameter_size(model)