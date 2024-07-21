import os
import torch
import sys
from torch import optim
import tqdm
from torchvision.utils import save_image
from custom_dataset import HEIHCDataset
from loss import generator_loss, discriminator_loss
import network

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Hyperparameters
num_epochs = 100
batch_size = 10
lr = 0.0001
patience = 10  # Early stopping patience

generator = network.Generator().to(device)
discriminator = network.Discriminator().to(device)

# Optimizers
gen_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
disc_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

def train(generator, discriminator, gen_optimizer, disc_optimizer, train_loader, test_loader, epochs, device, resume=None, save_interval=100, patience=10):
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

    for epoch in range(start_epoch, epochs):
        generator.train()
        discriminator.train()

        for i, (he_img, ihc_img) in enumerate(tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')):
            he_img, ihc_img = he_img.to(device), ihc_img.to(device)

            # Train Discriminator
            fake_ihc = generator(he_img)
            real_output = discriminator(he_img, ihc_img)
            fake_output = discriminator(he_img, fake_ihc.detach())
            
            d_loss = discriminator_loss(real_output, fake_output)

            disc_optimizer.zero_grad()
            d_loss.backward()
            disc_optimizer.step()

            # Train Generator
            fake_output = discriminator(he_img, fake_ihc)
            g_loss = generator_loss(fake_output, fake_ihc, ihc_img)

            gen_optimizer.zero_grad()
            g_loss.backward()
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
                real_output = discriminator(he_img, ihc_img)
                fake_output = discriminator(he_img, fake_ihc)

                d_loss = discriminator_loss(real_output, fake_output)
                g_loss = generator_loss(fake_output, fake_ihc, ihc_img)

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