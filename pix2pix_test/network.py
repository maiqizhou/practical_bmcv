import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, in_features):
        super(ResNetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_features)
        )

    def forward(self, x):
        output = x + self.conv_block(x)
        print(f'ResNetBlock output size: {output.size()}')
        return output

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=7, padding=3),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.Sequential(*[ResNetBlock(64) for _ in range(9)])
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.output_layer = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # x = self.input_layer(x)
        # print(f'Input layer output size: {x.size()}')
        # x = self.res_blocks(x)
        # x = self.deconv_layers(x)
        # print(f'Deconv layers output size: {x.size()}')
        # x = self.output_layer(x)
        # print(f'Output layer size: {x.size()}')
        # return x
        print("Input size:", x.size())
        x = self.input_layer(x)
        print("After input layer:", x.size())
        for i, block in enumerate(self.res_blocks, start=1):
            x = block(x)
            print(f"After ResNet block {i}:", x.size())
        x = self.deconv_layers(x)
        print("After deconv layers:", x.size())
        x = self.output_layer(x)
        print("Output layer size:", x.size())
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = self.model(x)
        # print(f'Discriminator output size: {x.size()}')
        # return x
        print("Discriminator Input size:", x.size())
        for i, layer in enumerate(self.model):
            x = layer(x)
            print(f"After layer {i}:", x.size())
        return x

if __name__ == "__main__":
    netG = Generator()
    netD = Discriminator()

    test_input = torch.randn(1, 4, 1024, 1024)  
    generated_image = netG(test_input)
    print("Generated Image Size:", generated_image.size())

    # Test Discriminator
    discriminator_result = netD(generated_image)
    print("Discriminator Result:", discriminator_result)
