import torch
import torch.nn as nn

def downsample(in_channels, out_channels, apply_batchnorm=True):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
    if apply_batchnorm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)

def upsample(in_channels, out_channels, apply_dropout=False):
    layers = [
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if apply_dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.down1 = downsample(3, 64, apply_batchnorm=False)  # (256, 256, 3) -> (128, 128, 64)
        self.down2 = downsample(64, 128)  # (128, 128, 64) -> (64, 64, 128)
        self.down3 = downsample(128, 256)  # (64, 64, 128) -> (32, 32, 256)
        self.down4 = downsample(256, 512)  # (32, 32, 256) -> (16, 16, 512)
        self.down5 = downsample(512, 512)  # (16, 16, 512) -> (8, 8, 512)
        self.down6 = downsample(512, 512)  # (8, 8, 512) -> (4, 4, 512)
        self.down7 = downsample(512, 512)  # (4, 4, 512) -> (2, 2, 512)
        self.down8 = downsample(512, 512)  # (2, 2, 512) -> (1, 1, 512)

        self.up1 = upsample(512, 512, apply_dropout=True)  # (1, 1, 512) -> (2, 2, 1024)
        self.up2 = upsample(1024, 512, apply_dropout=True)  # (2, 2, 1024) -> (4, 4, 1024)
        self.up3 = upsample(1024, 512, apply_dropout=True)  # (4, 4, 1024) -> (8, 8, 1024)
        self.up4 = upsample(1024, 512)  # (8, 8, 1024) -> (16, 16, 1024)
        self.up5 = upsample(1024, 256)  # (16, 16, 1024) -> (32, 32, 512)
        self.up6 = upsample(512, 128)  # (32, 32, 512) -> (64, 64, 256)
        self.up7 = upsample(256, 64)  # (64, 64, 256) -> (128, 128, 128)
        self.up8 = upsample(128, 128)  # (128, 128, 128) -> (256, 256, 128)
        self.final = nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1)  # (256, 256, 128) -> (256, 256, 3)
        self.tanh = nn.Tanh()

    def forward(self, x):
      # print(f"Input: {x.shape}")
      d1 = self.down1(x)
      # print(f"After down1: {d1.shape}")
      d2 = self.down2(d1)
      # print(f"After down2: {d2.shape}")
      d3 = self.down3(d2)
      # print(f"After down3: {d3.shape}")
      d4 = self.down4(d3)
      # print(f"After down4: {d4.shape}")
      d5 = self.down5(d4)
      # print(f"After down5: {d5.shape}")
      d6 = self.down6(d5)
      # print(f"After down6: {d6.shape}")
      d7 = self.down7(d6)
      # print(f"After down7: {d7.shape}")
      d8 = self.down8(d7)
      # print(f"After down8: {d8.shape}")

      u1 = self.up1(d8)
      u1 = torch.cat([u1, d7], dim=1)
      # print(f"After up1 and concat: {u1.shape}")
      u2 = self.up2(u1)
      u2 = torch.cat([u2, d6], dim=1)
      # print(f"After up2 and concat: {u2.shape}")
      u3 = self.up3(u2)
      u3 = torch.cat([u3, d5], dim=1)
      # print(f"After up3 and concat: {u3.shape}")
      u4 = self.up4(u3)
      u4 = torch.cat([u4, d4], dim=1)
      # print(f"After up4 and concat: {u4.shape}")
      u5 = self.up5(u4)
      u5 = torch.cat([u5, d3], dim=1)
      # print(f"After up5 and concat: {u5.shape}")
      u6 = self.up6(u5)
      u6 = torch.cat([u6, d2], dim=1)
      # print(f"After up6 and concat: {u6.shape}")
      u7 = self.up7(u6)
      u7 = torch.cat([u7, d1], dim=1)
      # print(f"After up7 and concat: {u7.shape}")
      u8 = self.up8(u7)
      # print(f"After up8: {u8.shape}")

      output = self.final(u8)
      # print(f"Output: {output.shape}")
      return self.tanh(output)



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv2 = downsample(64, 128)
        self.conv3 = downsample(128, 256)
        self.conv4 = downsample(256, 512)

        self.final = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, inp, tar):
      # print(f"Input: {inp.shape}, Target: {tar.shape}")
      x = torch.cat([inp, tar], dim=1)
      # print(f"After concat: {x.shape}")
      x = self.conv1(x)
      # print(f"After conv1: {x.shape}")
      x = self.conv2(x)
      # print(f"After conv2: {x.shape}")
      x = self.conv3(x)
      # print(f"After conv3: {x.shape}")
      x = self.conv4(x)
      # print(f"After conv4: {x.shape}")
      x = self.final(x)
      # print(f"Output: {x.shape}")
      return x

# if __name__ == "__main__":
#     G = Generator()
#     print('test Generator')
#     input_tensor = torch.randn(10, 3, 256, 256)  # Batch size of 10, 3 channels (RGB), 256x256 image
#     output_tensor = G(input_tensor)
#     print(output_tensor.shape)

#     D = Discriminator()
#     print('test Discriminator')
#     input_tensor = torch.randn(10, 3, 256, 256)
#     target_tensor = torch.randn(10, 3, 256, 256)
#     output_tensor = D(input_tensor, target_tensor)
#     print(output_tensor.shape)
