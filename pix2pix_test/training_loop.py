import torch
from loss import discriminator_loss, generator_loss_with_l1

def train_one_epoch(generator, discriminator, gen_optimizer, disc_optimizer, dataloader, device):
    generator.train()
    discriminator.train()

    for real_images in dataloader:
        real_images = real_images.to(device)
        noise = torch.randn(real_images.size(0), 100, 1, 1, device=device)  
        # Train Discriminator
        fake_images = generator(noise)
        real_outputs = discriminator(real_images)
        fake_outputs = discriminator(fake_images.detach()) 

        disc_loss = discriminator_loss(real_outputs, fake_outputs)
        disc_optimizer.zero_grad()
        disc_loss.backward()
        disc_optimizer.step()

        # Train Generator
        output = discriminator(fake_images)
        gen_loss = generator_loss_with_l1(output, fake_images, real_images)
        gen_optimizer.zero_grad()
        gen_loss.backward()
        gen_optimizer.step()


    return disc_loss.item(), gen_loss.item()
