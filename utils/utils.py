import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

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

def get_transforms(resize=256, normalize=True) -> transforms.Compose:
    """
    Get the image transforms for the input image.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((resize, resize)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if normalize else None
    ])

def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize the input tensor back to the range [0, 1].
    """
    return (tensor + 1) / 2

if __name__ == '__main__':
    # Test
    image_tensor = torch.zeros(3, 256, 256)
    text = 'Hello, World!'
    image_tensor_with_text = add_text_to_image(image_tensor, text)
    image_with_text = transforms.ToPILImage()(image_tensor_with_text)
    image_with_text.show()
    image_with_text.save('image_with_text.png')