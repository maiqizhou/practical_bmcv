import os
import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision
from torchvision import transforms

from data.paired_dataset import PairedDataset, PairWarper

def add_text_to_image(image_tensor, text, position=(10, 10), font_path=None, font_size=30, color=(0, 0, 0)) -> torch.Tensor:
    """
    Add text to the input image tensor for labeling.
    """
    image_pil = transforms.ToPILImage()(image_tensor)
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.load_default() if font_path is None else ImageFont.truetype(font_path, font_size)
    draw.text(position, text, color, font=font)
    image_tensor = transforms.ToTensor()(image_pil)
    return image_tensor

def get_transforms(resize=None, normalize=True) -> transforms.Compose:
    """
    Get the image transforms for the input image.
    """
    transforms_list = []
    if resize:
        transforms_list.append(transforms.Resize(resize))
    transforms_list.append(transforms.ToTensor())
    if normalize:
        transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(transforms_list)

def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize the input tensor back to the range [0, 1].
    """
    return (tensor + 1) / 2

def build_dataloader(batch_size, resize=None, patch_size=None, normalize=True):
    transform1 = get_transforms(normalize=normalize, resize=resize)
    transform2 = get_transforms(normalize=normalize, resize=resize)

    train_dataset = PairedDataset('data/BCI_dataset_512/HE/train', 'data/BCI_dataset_512/IHC/train', transform=transform1, patch_size=patch_size)
    test_dataset = PairedDataset('data/BCI_dataset_512/HE/test', 'data/BCI_dataset_512/IHC/test', transform=transform2, patch_size=patch_size)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader

def build_CIFAR10_dataloader(batch_size, resize=None, normalize=True):
    transform = get_transforms(resize=resize, normalize=normalize)
    transform2 = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                            torchvision.transforms.Grayscale(num_output_channels=3)])
    cifar10 = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True)
    cifar10 = PairWarper(cifar10, cifar10, transform=transform, transform2=transform2)
    train_size = int(0.8 * len(cifar10))
    test_size = len(cifar10) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(cifar10, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader

def resize_image(file, root, path_dest, size):
    try:
        image = Image.open(os.path.join(root, file))
        image = image.resize(size)
        image.save(os.path.join(path_dest, file))
    except Exception as e:
        print(f"Error processing file {file}: {e}")

def resize_path(path_target, path_dest, size, num_workers=8):
    """
    Resize the images in the input path to the given size and keep their original names.
    """
    from PIL import Image
    import concurrent.futures

    if not os.path.exists(path_dest):
        os.makedirs(path_dest)
    
    files_to_process = []
    for root, _, files in os.walk(path_target):
        for file in files:
            files_to_process.append((file, root))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for file, root in files_to_process:
            futures.append(executor.submit(resize_image, file, root, path_dest, size))
        
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error occurred: {e}")

def build_log_info(loss, epoch, time):
    loss_info = ''
    for key in loss:
        loss_info += f'{key}: {loss[key]:.4f} '

if __name__ == '__main__':
    # Test
    # image_tensor = torch.zeros(3, 256, 256)
    # text = 'Hello, World!'
    # image_tensor_with_text = add_text_to_image(image_tensor, text)
    # image_with_text = transforms.ToPILImage()(image_tensor_with_text)
    # image_with_text.show()
    # image_with_text.save('image_with_text.png')
    # resize_path('data/BCI_dataset/HE/test', 'data/BCI_dataset_512/HE/test', (512, 512))
    resize_path('data/BCI_dataset/IHC/train', 'data/BCI_dataset_512/IHC/train', (512, 512))
    resize_path('data/BCI_dataset/IHC/test', 'data/BCI_dataset_512/IHC/test', (512, 512))