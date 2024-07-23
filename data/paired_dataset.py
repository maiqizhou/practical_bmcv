import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.RandomCrop(128),
    transforms.ToTensor()
])

class PairedDataset(Dataset):
    def __init__(self, he_dir, ihc_dir, transform=None, patch_size=None):
        self.he_dir = he_dir
        self.ihc_dir = ihc_dir
        self.transform = transform
        self.image_names = os.listdir(he_dir)
        self.patch_size = patch_size
        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        he_path = os.path.join(self.he_dir, img_name)
        ihc_path = os.path.join(self.ihc_dir, img_name)
        
        he_image = Image.open(he_path).convert('RGB')
        ihc_image = Image.open(ihc_path).convert('RGB')

        if self.patch_size:
            he_image, ihc_image = self.random_crop((he_image, ihc_image), self.patch_size)
        
        if self.transform:
            he_image = self.transform(he_image)
            ihc_image = self.transform(ihc_image)
            
        return he_image, ihc_image
    
    @staticmethod
    def random_crop(image_pair, patch_size):
        he_image, ihc_image = image_pair
        assert patch_size < min(he_image.size), 'Patch size should be less than the minimum dimension of the image'
        w, h = he_image.size
        th, tw = patch_size, patch_size
        if w == tw and h == th:
            return he_image, ihc_image
        i = np.random.randint(0, h - th)
        j = np.random.randint(0, w - tw)
        return he_image.crop((j, i, j + tw, i + th)), ihc_image.crop((j, i, j + tw, i + th))
    
class PairWarper(Dataset):
    def __init__(self, dataset1, dataset2, transform=None, transform2=None):
        assert len(dataset1) == len(dataset2), 'Datasets should have the same length'
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.transform = transform
        self.transform2 = transform2
        
    def __len__(self):
        return len(self.dataset1)
    
    def __getitem__(self, idx):
        item1 = self.dataset1[idx][0]
        item2 = self.dataset2[idx][0]

        if self.transform:
            item1 = self.transform(item1)
            if self.transform2:
                item2 = self.transform2(item2)
            else:
                item2 = self.transform(item2)

        return item1, item2