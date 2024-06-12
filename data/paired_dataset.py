import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class PairedDataset(Dataset):
    def __init__(self, he_dir, ihc_dir, transform=None):
        self.he_dir = he_dir
        self.ihc_dir = ihc_dir
        self.transform = transform
        self.image_names = os.listdir(he_dir)
        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        he_path = os.path.join(self.he_dir, img_name)
        ihc_path = os.path.join(self.ihc_dir, img_name)
        
        he_image = Image.open(he_path).convert('RGB')
        ihc_image = Image.open(ihc_path).convert('RGB')
        
        if self.transform:
            he_image = self.transform(he_image)
            ihc_image = self.transform(ihc_image)
            
        return he_image, ihc_image