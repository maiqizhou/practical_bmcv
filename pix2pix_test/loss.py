import torch
import torch.nn as nn
import torch.nn.functional as F

def generator_loss(outputs_fake):
    
    gen_loss = F.binary_cross_entropy(outputs_fake, torch.ones_like(outputs_fake))
    return gen_loss

def generator_loss_with_l1(outputs_fake, generated_images, real_images, l1_weight=100):
    
    bce_loss = F.binary_cross_entropy(outputs_fake, torch.ones_like(outputs_fake))
    l1_loss = F.l1_loss(generated_images, real_images)

    total_loss = bce_loss + (l1_weight * l1_loss)
    return total_loss

def discriminator_loss(outputs_real, outputs_fake):
    
    real_loss = F.binary_cross_entropy(outputs_real, torch.ones_like(outputs_real))
    fake_loss = F.binary_cross_entropy(outputs_fake, torch.zeros_like(outputs_fake))

    total_loss = (real_loss + fake_loss) / 2
    return total_loss
