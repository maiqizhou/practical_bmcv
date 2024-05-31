import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from training_loop import train_one_epoch
from network import Generator, Discriminator

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    num_epochs = 50
    batch_size = 64
    learning_rate = 0.0002

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    gen_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # Data
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  
    ])
    dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        disc_loss, gen_loss = train_one_epoch(generator, discriminator, gen_optimizer, disc_optimizer, dataloader, device)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss D: {disc_loss:.4f}, Loss G: {gen_loss:.4f}')

if __name__ == "__main__":
    main()
