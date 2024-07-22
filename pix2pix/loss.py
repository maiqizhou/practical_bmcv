import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Vgg19(nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda() if torch.cuda.is_available() else Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

def generator_loss(disc_generated_output, gen_output, target, lambda_L1=100):
    gan_loss = 0
    for disc_generated_output in disc_generated_output:
        gan_loss += F.binary_cross_entropy_with_logits(disc_generated_output, torch.ones_like(disc_generated_output))
    gan_loss /= len(disc_generated_output)
    l1_loss = F.l1_loss(gen_output, target)
    total_gen_loss = gan_loss + (lambda_L1 * l1_loss)
    return total_gen_loss

def generator_loss_with_vgg(fake_output, fake_images, real_images, vgg_loss_fn):
    gan_loss = F.binary_cross_entropy_with_logits(fake_output, torch.ones_like(fake_output))
    l1_loss = F.l1_loss(fake_images, real_images)
    vgg_loss = vgg_loss_fn(fake_images, real_images)
    total_gen_loss = gan_loss + (100 * l1_loss) + (10 * vgg_loss)
    return total_gen_loss

# def feature_matching_loss(disc_real_outputs, disc_generated_outputs):
#     fm_loss = 0
#     for disc_real_output, disc_generated_output in zip(disc_real_outputs, disc_generated_outputs):
#         for real_feat, fake_feat in zip(disc_real_output, disc_generated_output):
#             fm_loss += torch.mean(torch.abs(real_feat - fake_feat))
#     return fm_loss

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    real_loss = 0
    generated_loss = 0
    for disc_real_output, disc_generated_output in zip(disc_real_outputs, disc_generated_outputs):
        real_loss += torch.mean(F.binary_cross_entropy_with_logits(disc_real_output, torch.ones_like(disc_real_output)))
        generated_loss += torch.mean(F.binary_cross_entropy_with_logits(disc_generated_output, torch.zeros_like(disc_generated_output)))
    total_disc_loss = (real_loss + generated_loss) * 0.5
    return total_disc_loss
