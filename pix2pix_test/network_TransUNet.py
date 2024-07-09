import torch
import torch.nn as nn
import torchvision.models as models
from vit_pytorch import ViT

class TransUnetEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(TransUnetEncoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.layer1 = nn.Sequential(*list(resnet.children())[:4])  # 64
        self.layer2 = nn.Sequential(*list(resnet.children())[4])   # 256
        self.layer3 = nn.Sequential(*list(resnet.children())[5])   # 512
        self.layer4 = nn.Sequential(*list(resnet.children())[6])   # 1024
        
        self.Vit = ViT(
            # image_size=256,
            image_size=16,  
            patch_size=1,
            # patch_size=32,
            num_classes=1024,
            dim=1024,  
            depth=10,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
            channels=1024
        )

        self.conv_after_vit = nn.Conv2d(1024, 512, kernel_size=1)

    def forward(self, x):
        x1 = self.layer1(x)
        # print(f"x1: {x1.shape}")  # Debug
        x2 = self.layer2(x1)
        # print(f"x2: {x2.shape}")  # Debug
        x3 = self.layer3(x2)
        # print(f"x3: {x3.shape}")  # Debug
        x4 = self.layer4(x3)
        # print(f"x4: {x4.shape}")  # Debug
        
        x4 = self.Vit(x4)
        # print(f"x4 after ViT: {x4.shape}")  # Debug
        
        batch_size = x4.size(0)
        x4 = x4.view(batch_size, 1024, 1, 1)
        # print(f"x4 reshaped: {x4.shape}")  # Debug

        x4 = self.conv_after_vit(x4)
        # print(f"x4 after conv: {x4.shape}")  # Debug
        
        x4 = nn.functional.interpolate(x4, size=(32, 32), mode='bilinear', align_corners=True)
        # print(f"x4 upsampled: {x4.shape}")  # Debug
        
        return [x1, x2, x3, x4]

class TransUnetDecoder(nn.Module):
    def __init__(self, out_channels=64, **kwargs):
        super(TransUnetDecoder, self).__init__()
        self.decoder1 = nn.Sequential(
            nn.Conv2d(out_channels // 4, out_channels // 4, 3, padding=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU()
        )
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(out_channels, out_channels // 4, 3, padding=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU()
        )

        self.decoder2 = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.decoder3 = nn.Sequential(
            nn.Conv2d(out_channels * 6, out_channels * 2, 3, padding=1),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU()
        )
        self.upsample3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(out_channels * 4, out_channels * 2, 3, padding=1),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU()
        )

        self.decoder4 = nn.Sequential(
            nn.Conv2d(out_channels * 16, out_channels * 4, 3, padding=1),
            nn.BatchNorm2d(out_channels * 4),
            nn.ReLU()
        )
        self.upsample4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(out_channels * 8, out_channels * 4, 3, padding=1),
            nn.BatchNorm2d(out_channels * 4),
            nn.ReLU()
        )

    def forward(self, inputs):
        x1, x2, x3, x4 = inputs
        # print(f"x4 before reshape: {x4.shape}")

        x4 = nn.functional.interpolate(x4, size=x3.shape[2:], mode='bilinear', align_corners=True)
        # print(f"x4 upsampled to match x3: {x4.shape}")

        x = self.decoder4(torch.cat([x4, x3], dim=1))
        # print(f"x after decoder4: {x.shape}")

        x = self.upsample3(x)
        x = torch.cat([x, x2], dim=1)
        # print(f"x before decoder3: {x.shape}")
        x = self.decoder3(x)
        # print(f"x after decoder3: {x.shape}")

        x = self.upsample2(x)
        x1 = nn.functional.interpolate(x1, size=x.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, x1], dim=1)
        # print(f"x before decoder2: {x.shape}")
        x = self.decoder2(x)
        # print(f"x after decoder2: {x.shape}")

        x = self.upsample1(x)
        x = self.decoder1(x)
        # print(f"x after decoder1: {x.shape}")

        return x

class Generator(nn.Module):
    def __init__(self, num_classes=3, **kwargs):
        super(Generator, self).__init__()
        self.TransUnetEncoder = TransUnetEncoder()
        self.TransUnetDecoder = TransUnetDecoder()
        self.cls_head = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        x = self.TransUnetEncoder(x)
        x = self.TransUnetDecoder(x)
        x = self.cls_head(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv2 = self.downsample(64, 128)
        self.conv3 = self.downsample(128, 256)
        self.conv4 = self.downsample(256, 512)

        self.final = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def downsample(self, in_channels, out_channels, apply_batchnorm=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if apply_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

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

if __name__ == "__main__":
    generator = Generator()
    print('Testing Generator')
    input_tensor = torch.randn(1, 3, 256, 256)  
    output_tensor = generator(input_tensor)
    print(output_tensor.shape)

    discriminator = Discriminator()
    print('Testing Discriminator')
    input_tensor = torch.randn(1, 3, 256, 256)
    target_tensor = torch.randn(1, 3, 256, 256)
    output_tensor = discriminator(input_tensor, target_tensor)
    print(output_tensor.shape)
