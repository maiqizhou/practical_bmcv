import torch
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn

# class VGG19FeatureExtractor(nn.Module):
#     def __init__(self, layers):
#         super(VGG19FeatureExtractor, self).__init__()
#         vgg19 = models.vgg19(pretrained=True).features
#         self.chosen_layers = [vgg19[i] for i in layers]
#         self.model = nn.Sequential(*self.chosen_layers).eval()

#         for param in self.model.parameters():
#             param.requires_grad = False

#     def forward(self, x):
#         features = []
#         for layer in self.model:
#             x = layer(x)
#             features.append(x)
#         return features

# vgg_layers = [3, 8, 17]  
# vgg19_extractor = VGG19FeatureExtractor(vgg_layers).to('cuda' if torch.cuda.is_available() else 'cpu')

# # new add
# def perceptual_loss(vgg_extractor, gen_images, real_images):
#     gen_features = vgg_extractor(gen_images)
#     real_features = vgg_extractor(real_images)

#     loss = 0.0
#     for gf, rf in zip(gen_features, real_features):
#         loss += F.l1_loss(gf, rf)
#     return loss

def generator_loss(disc_generated_output, gen_output, target, lambda_L1=100):
    gan_loss = F.binary_cross_entropy_with_logits(disc_generated_output, torch.ones_like(disc_generated_output))
    l1_loss = F.l1_loss(gen_output, target)
    total_gen_loss = gan_loss + (lambda_L1 * l1_loss)
    return total_gen_loss

# def generator_loss(disc_generated_output, gen_output, target, lambda_L1=100, lambda_perceptual=1, vgg_extractor=None):
#     gan_loss = F.binary_cross_entropy_with_logits(disc_generated_output, torch.ones_like(disc_generated_output))
#     l1_loss = F.l1_loss(gen_output, target)
#     total_gen_loss = gan_loss + (lambda_L1 * l1_loss)
    
#     if vgg_extractor is not None:
#         perc_loss = perceptual_loss(vgg_extractor, gen_output, target)
#         total_gen_loss += (lambda_perceptual * perc_loss)
    
#     return total_gen_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = torch.mean(torch.nn.functional.binary_cross_entropy_with_logits(disc_real_output, torch.ones_like(disc_real_output)))
    generated_loss = torch.mean(torch.nn.functional.binary_cross_entropy_with_logits(disc_generated_output, torch.zeros_like(disc_generated_output)))
    total_disc_loss = (real_loss + generated_loss) * 0.5
    return total_disc_loss
