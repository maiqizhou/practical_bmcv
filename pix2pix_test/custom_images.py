import os
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image

class HEIHCDataset(Dataset):
    def __init__(self, he_dir, ihc_dir, transform=None):
        self.he_dir = he_dir
        self.ihc_dir = ihc_dir
        self.transform = transform
        self.he_images = [f for f in os.listdir(he_dir) if f.endswith('.png')]
        self.ihc_images = [f for f in os.listdir(ihc_dir) if f.endswith('.png')]

        self.check_files()

    def __len__(self):
        return len(self.he_images)

    def __getitem__(self, idx):
        image_name = self.he_images[idx]
        he_image_path = os.path.join(self.he_dir, image_name)
        ihc_image_path = os.path.join(self.ihc_dir, image_name)

        try:
            # Use PIL to open images and convert to RGB
            he_image = Image.open(he_image_path).convert('RGB')
            ihc_image = Image.open(ihc_image_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading images: {e}")

        if self.transform:
            he_image = self.transform(he_image)
            ihc_image = self.transform(ihc_image)

        return he_image, ihc_image
    
    def check_files(self):
        missing_he = [img for img in self.he_images if img not in self.ihc_images]
        missing_ihc = [img for img in self.ihc_images if img not in self.he_images]
        if missing_he or missing_ihc:
            raise ValueError(f"Missing files - HE: {missing_he}, IHC: {missing_ihc}")

transform = Compose([
    Resize((256, 256)),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Create Dataset and DataLoader
# he_dir = './Data/BCI_filtered/train/HE/'
# ihc_dir = './Data/BCI_filtered/train/IHC/'
he_dir = '/content/drive/MyDrive/Data/BCI_origin/HE/train'
ihc_dir = '/content/drive/MyDrive/Data/BCI_origin/IHC/train'
dataset = HEIHCDataset(he_dir, ihc_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
