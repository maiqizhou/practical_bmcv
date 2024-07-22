import os
import torch
import sys
from torch import optim
import tqdm
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from custom_dataset import HEIHCDataset
# from loss import generator_loss, discriminator_loss
from loss import discriminator_loss, VGGLoss, generator_loss_with_vgg
import network

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# Hyperparameters
num_epochs = 100
batch_size = 10
lr = 0.0002
patience = 10  # Early stopping patience

generator = network.Generator().to(device)
discriminator = network.Discriminator().to(device)

# Optimizers
gen_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
disc_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

vgg_loss_fn = VGGLoss().to(device)

def print_tensor_stats(tensor, name):
    print(f"{name}:")
    print(f" - Shape: {tensor.shape}")
    print(f" - Max: {tensor.max().item()}")
    print(f" - Min: {tensor.min().item()}")
    print(f" - Mean: {tensor.mean().item()}")
    print(f" - Std: {tensor.std().item()}")

def train(generator, discriminator, gen_optimizer, disc_optimizer, train_loader, test_loader, epochs, device, resume=None, save_interval=100, patience=20):
    os.makedirs('snapshots/checkpoints', exist_ok=True)
    os.makedirs('snapshots/samples', exist_ok=True)
    log_file = 'snapshots/training_log.txt'

    start_epoch = 0
    if resume:
        try:
            checkpoint = torch.load(resume)
            generator.load_state_dict(checkpoint['generator'])
            discriminator.load_state_dict(checkpoint['discriminator'])
            start_epoch = checkpoint['epoch']
            print(f"Resuming training from epoch {start_epoch}")
            file_mode = 'a'
        except FileNotFoundError:
            print("Checkpoint not found, starting training from scratch.")
            file_mode = 'w'
    else:
        print("Starting training from scratch.")
        file_mode = 'w'

    with open(log_file, file_mode) as f:
        f.write("Training Log\n")

    best_g_loss = float('inf')
    early_stop_counter = 0
    vgg_loss_fn = VGGLoss().to(device)

    for epoch in range(start_epoch, epochs):
        generator.train()
        discriminator.train()

        for i, (he_img, ihc_img) in enumerate(tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')):
            he_img, ihc_img = he_img.to(device), ihc_img.to(device)

            # Print input images stats
            # print_tensor_stats(he_img, "HE Image")
            # print_tensor_stats(ihc_img, "IHC Image")

            # Train Discriminator
            fake_ihc = generator(he_img)
            # fake_ihc = (fake_ihc + 1) / 2

            # Print generated image stats
            # print_tensor_stats(fake_ihc, "Generated IHC Image")
            
            real_output = discriminator(he_img, ihc_img)
            fake_output = discriminator(he_img, fake_ihc.detach())

            # Print discriminator outputs stats
            # print_tensor_stats(real_output, "Discriminator Real Output")
            # print_tensor_stats(fake_output, "Discriminator Fake Output")
            
            d_loss = discriminator_loss(real_output, fake_output)

            disc_optimizer.zero_grad()
            d_loss.backward()
            disc_optimizer.step()

            # Train Generator
            fake_output = discriminator(he_img, fake_ihc)
            g_loss = generator_loss_with_vgg(fake_output, fake_ihc, ihc_img, vgg_loss_fn)
            # g_loss = generator_loss(fake_output, fake_ihc, ihc_img)

            # Feature Matching Loss
            # Feature Matching Loss
            # real_output_detached = [ro.detach() for ro in real_output]  # Detach to avoid backprop through discriminator
            # fake_output_detached = [fo.detach() for fo in fake_output]  # Detach to avoid backprop through discriminator
            # fm_loss = feature_matching_loss(real_output_detached, fake_output_detached)
            # g_loss_total = g_loss + fm_loss * 10  # Adjust the weight for feature matching loss
            
            g_loss_total = g_loss

            gen_optimizer.zero_grad()
            g_loss_total.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)

            gen_optimizer.step()

            if i % save_interval == 0:
                with open(log_file, 'a') as f:
                    log_msg = f"Epoch: {epoch+1}, Step: {i}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}\n"
                    f.write(log_msg)
                
                save_image(fake_ihc.data[:25], f'snapshots/samples/fake_epoch_{epoch+1}_step_{i}.png', nrow=5, normalize=True)
                save_image(ihc_img.data[:25], f'snapshots/samples/real_epoch_{epoch+1}_step_{i}.png', nrow=5, normalize=True)

        generator.eval()
        test_d_loss = 0.0
        test_g_loss = 0.0
        with torch.no_grad():
            for he_img, ihc_img in tqdm.tqdm(test_loader, desc=f'Testing Epoch {epoch+1}/{epochs}'):
                he_img, ihc_img = he_img.to(device), ihc_img.to(device)

                fake_ihc = generator(he_img)
                # fake_ihc = (fake_ihc + 1) / 2

                real_output = discriminator(he_img, ihc_img)
                fake_output = discriminator(he_img, fake_ihc)

                d_loss = discriminator_loss(real_output, fake_output)
                # g_loss = generator_loss(fake_output, fake_ihc, ihc_img)
                g_loss = generator_loss_with_vgg(fake_output, fake_ihc, ihc_img, vgg_loss_fn)

                test_d_loss += d_loss.item()
                test_g_loss += g_loss.item()

        test_d_loss /= len(test_loader)
        test_g_loss /= len(test_loader)

        with open(log_file, 'a') as f:
            log_msg = f"Epoch: {epoch+1}, Test D Loss: {test_d_loss:.4f}, Test G Loss: {test_g_loss:.4f}\n"
            f.write(log_msg)

        # Save the best model
        if test_g_loss < best_g_loss:
            best_g_loss = test_g_loss
            early_stop_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict()
            }, 'snapshots/checkpoints/best_model.pth')
        else:
            early_stop_counter += 1

        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict()
        }, f'snapshots/checkpoints/ckpt_epoch_{epoch+1}.pth')

        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print("Training completed.")

if __name__ == "__main__":
    pass
