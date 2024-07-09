import argparse
import torch
import os
import sys
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from custom_dataset import HEIHCDataset
from training_loop import train
from network import Generator, Discriminator

def main():
    parser = argparse.ArgumentParser(description="Train a GAN model on paired HE and IHC images.")
    parser.add_argument('--he_dir', type=str, required=True, help='Directory containing HE images')
    parser.add_argument('--ihc_dir', type=str, required=True, help='Directory containing IHC images')
    parser.add_argument('--he_test_dir', type=str, required=True, help='Directory containing HE test images')
    parser.add_argument('--ihc_test_dir', type=str, required=True, help='Directory containing IHC test images')
    parser.add_argument('--batch_size', type=int, default=64, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--resume', type=str, default=None, help='Path to latest checkpoint (if resuming)')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = HEIHCDataset(args.he_dir, args.ihc_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = HEIHCDataset(args.he_test_dir, args.ihc_test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    train(generator, discriminator, gen_optimizer, disc_optimizer, train_loader, test_loader, args.epochs, device, args.resume)

if __name__ == '__main__':
    main()
