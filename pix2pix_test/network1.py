import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm2d(dim)
        ]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Encoder
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(3, 64, kernel_size=7, padding=0, bias=False),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(True)]
        
        # Adding 9 ResNet blocks
        for _ in range(9):
            model.append(ResNetBlock(64))
        
        # Decoder
        model.extend([
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7, padding=0),
            nn.Tanh()
        ])
        self.model = nn.Sequential(*model)

    # def forward(self, x):
    #     return self.model(x)
    def forward(self, x):
        print("Input size:", x.size())
        x = self.model[0](x)  # ReflectionPad2d
        x = self.model[1](x)  # Conv2d
        print("After initial conv:", x.size())
        for i in range(2, 11):  # ResNet blocks
            x = self.model[i](x)
            print(f"After ResNet block {i-1}:", x.size())
        for i in range(11, len(self.model)):  # Decoder and output layers
            x = self.model[i](x)
            print(f"After layer {i}:", x.size())
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        model = [
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        ]
        self.model = nn.Sequential(*model)

    # def forward(self, x):
    #     return self.model(x)
    def forward(self, x):
        print("Input size:", x.size())
        for i, layer in enumerate(self.model):
            x = layer(x)
            print(f"After layer {i}:", x.size())
        return x

if __name__ == '__main__':

    netG = Generator()
    netD = Discriminator()

    # Print model summaries
    # print(netG)
    # print(netD)

    # Test Generator
    test_input = torch.randn(1, 3, 1024, 1024)  
    generated_image = netG(test_input)
    print("Generated Image Size:", generated_image.size())

    # Test Discriminator
    discriminator_result = netD(generated_image)
    print("Discriminator Result:", discriminator_result)
