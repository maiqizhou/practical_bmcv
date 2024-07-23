import sys
import torch
import base64
from io import BytesIO
from PIL import Image
sys.path.append('pix2pix')
from pix2pix.network_attention import Generator #import generator from pix network
from torchvision.utils import save_image
from torchvision import transforms

def convert_he_to_ihc(he_image_path):
    print("Loading HE image...")
    he_image = Image.open(he_image_path)

    print("Loading Pix2Pix model...")
    # load pix2pix model
    device = torch.device("cpu")  # Set default device to CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    checkpoint_path = '/Users/maggie/practical_bmcv/best_model.pth'
    generator = load_pix2pix_model(checkpoint_path)
    
    # Convert the PIL image to a tensor
    he_image_tensor = load_image(he_image_path).to(device)
    
    print("Generating IHC image...")
    ihc_image = generate_ihc_image(generator, he_image_tensor, device)
    
    print("Converting image to Base64...")
    buffered = BytesIO()

    # Save the PIL Image to the BytesIO object
    ihc_image.save(buffered, format="PNG")

    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str




def load_pix2pix_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        print("Loading checkpoint...")
        checkpoint = torch.load(model_path, map_location=device)
    except FileNotFoundError:
        print('Invalid checkpoint path')
        exit(1)
    
    print("Loading model state...")
    model = Generator()
    model.load_state_dict(checkpoint['generator'])
    model = model.to(device)
    model.eval()
    return model

def load_image(image_path, resize_to=(256, 256)):
    print("Loading and transforming image...")
    transform = transforms.Compose([
        transforms.Resize(resize_to),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

def generate_ihc_image(generator, he_image_tensor, device):
    print("Generating image...")
    with torch.no_grad():
        generated_image = generator(he_image_tensor)

    print("Post-processing generated image...")
    generated_image = (generated_image * 0.5) + 0.5  # Denormalize to [0, 1]
    generated_image = generated_image.squeeze().cpu().numpy().transpose(1, 2, 0)
    generated_image = (generated_image * 255).astype('uint8')
    generated_image_pil = Image.fromarray(generated_image)
    return generated_image_pil

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_script.py <he_image_path>")
        sys.exit(1)
    
    he_image_path = sys.argv[1]
    # he_image_path = '/Users/maggie/Downloads/BCI-main/PyramidPix2pix/BCI_dataset/HE/train/00500_train_1+.png'
    print("Starting conversion process...")
    sys.stdout.flush()
    ihc_image_base64 = convert_he_to_ihc(he_image_path)
    print("Conversion process completed.")
    sys.stdout.flush()
    print(ihc_image_base64, end="")
