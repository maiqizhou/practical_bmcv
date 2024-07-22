import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms

def downsample(in_channels, out_channels, apply_batchnorm=True):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
    if apply_batchnorm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)

def upsample(in_channels, out_channels, apply_dropout=True):
    layers = [
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if apply_dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.relu = nn.ReLU(inplace=True)
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Encoder
        self.down1 = downsample(3, 64, apply_batchnorm=False)  # (256, 256, 3) -> (128, 128, 64)
        self.down2 = downsample(64, 128)  # (128, 128, 64) -> (64, 64, 128)
        self.down3 = downsample(128, 256)  # (64, 64, 128) -> (32, 32, 256)
        self.down4 = downsample(256, 512)  # (32, 32, 256) -> (16, 16, 512)
        self.down5 = downsample(512, 512)  # (16, 16, 512) -> (8, 8, 512)
        self.down6 = downsample(512, 512)  # (8, 8, 512) -> (4, 4, 512)
        self.down7 = downsample(512, 512)  # (4, 4, 512) -> (2, 2, 512)
        self.down8 = nn.Sequential(  
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Attention blocks
        self.att7 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.att6 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.att4 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.att3 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.att2 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.att1 = Attention_block(F_g=64, F_l=64, F_int=32)
        


        # Decoder
        self.up1 = upsample(512, 512, apply_dropout=True)  # (1, 1, 512) -> (2, 2, 1024)        
        self.up2 = upsample(1024, 512, apply_dropout=True)  # (2, 2, 1024) -> (4, 4, 1024)
        self.up3 = upsample(1024, 512, apply_dropout=True)  # (4, 4, 1024) -> (8, 8, 1024)
        self.up4 = upsample(1024, 512)  # (8, 8, 1024) -> (16, 16, 1024)
        self.up5 = upsample(1024, 256)  # (16, 16, 1024) -> (32, 32, 512)
        self.up6 = upsample(512, 128)  # (32, 32, 512) -> (64, 64, 256)
        self.up7 = upsample(256, 64)  # (64, 64, 256) -> (128, 128, 128)
        self.up8 = upsample(128, 64)  # (128, 128, 128) -> (256, 256, 64)
        self.final = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)  # (256, 256, 64) -> (256, 256, 3)
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
        d7 = self.att7(g=u1, x=d7)
        # print(f"After att7: {d7.shape}")
        u1 = torch.cat([u1, d7], dim=1)
        # print(f"After up1 and concat: {u1.shape}")

        u2 = self.up2(u1)
        d6 = self.att6(g=u2, x=d6)
        # print(f"After att6: {d6.shape}")
        u2 = torch.cat([u2, d6], dim=1)
        # print(f"After up2 and concat: {u2.shape}")

        u3 = self.up3(u2)
        d5 = self.att5(g=u3, x=d5)
        # print(f"After att5: {d5.shape}")
        u3 = torch.cat([u3, d5], dim=1)
        # print(f"After up3 and concat: {u3.shape}")

        u4 = self.up4(u3)
        d4 = self.att4(g=u4, x=d4)
        # print(f"After att4: {d4.shape}")
        u4 = torch.cat([u4, d4], dim=1)
        # print(f"After up4 and concat: {u4.shape}")

        u5 = self.up5(u4)
        d3 = self.att3(g=u5, x=d3)
        # print(f"After att3: {d3.shape}")
        u5 = torch.cat([u5, d3], dim=1)
        # print(f"After up5 and concat: {u5.shape}")

        u6 = self.up6(u5)
        d2 = self.att2(g=u6, x=d2)
        # print(f"After att2: {d2.shape}")
        u6 = torch.cat([u6, d2], dim=1)
        # print(f"After up6 and concat: {u6.shape}")

        u7 = self.up7(u6)
        d1 = self.att1(g=u7, x=d1)
        # print(f"After att1: {d1.shape}")
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

# class MultiscaleDiscriminator(nn.Module):
#     def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, num_D=3):
#         super(MultiscaleDiscriminator, self).__init__()
#         self.num_D = num_D
#         self.n_layers = n_layers

#         for i in range(num_D):
#             netD = self.create_single_discriminator(input_nc, ndf, n_layers, norm_layer)
#             setattr(self, 'layer' + str(i), netD)

#         self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

#     def create_single_discriminator(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
#         layers = [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
#                   nn.LeakyReLU(0.2, inplace=True)]
#         nf_mult = 1
#         nf_mult_prev = 1
#         for n in range(1, n_layers):
#             nf_mult_prev = nf_mult
#             nf_mult = min(2**n, 8)
#             layers += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
#                                  kernel_size=4, stride=2, padding=1, bias=False),
#                        norm_layer(ndf * nf_mult),
#                        nn.LeakyReLU(0.2, inplace=True)]

#         nf_mult_prev = nf_mult
#         nf_mult = min(2**n_layers, 8)
#         layers += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
#                              kernel_size=4, stride=1, padding=1, bias=False),
#                    norm_layer(ndf * nf_mult),
#                    nn.LeakyReLU(0.2, inplace=True)]

#         layers += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1, bias=False)]
#         return nn.Sequential(*layers)

#     def forward(self, input):
#         results = []
#         for i in range(self.num_D):
#             model = getattr(self, 'layer' + str(i))
#             out = model(input)
#             # print(f"Output of D{i + 1}: {out.shape}")
#             results.append(out)
#             if i != (self.num_D - 1):
#                 input = self.downsample(input)
#         return results

if __name__ == "__main__":
    input_nc = 3
    output_nc = 3
    generator = Generator()
    print('Testing Generator')
    input_tensor = torch.randn(1, 3, 256, 256)  # Batch size of 1, 3 channels (RGB), 256x256 image
    output_tensor = generator(input_tensor)
    print(output_tensor.shape)

    discriminator = Discriminator()
    # discriminator = MultiscaleDiscriminator(input_nc + output_nc)
    print('Testing Discriminator')
    input_tensor = torch.randn(1, 3, 256, 256)
    target_tensor = torch.randn(1, 3, 256, 256)
    output_tensor = discriminator(input_tensor, target_tensor)
    for i, out in enumerate(output_tensor):
        print(f"Output of D{i + 1}: {out.shape}")
