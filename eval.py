import os

import numpy as np
from tqdm import tqdm
import lpips
import torch
from torchvision import transforms
from torchvision.utils import save_image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from PIL import Image

from data.paired_dataset import PairedDataset

def load_named_model(model_name, model_path):
    """
    Load the model.
    
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # try:
    #     checkpoint = torch.load(model_path, map_location=device)
    # except FileNotFoundError:
    #     print('Invalid checkpoint path')
    #     exit(1)
    
    if model_name == 'diff_ae':
        from diff_ae.diffusion import build_model
        # TODO: Config loading
        model = build_model(device, 512, 200)
        model.load(model_path)
    else:
        raise NotImplementedError(f"Model not implemented")
    return model

def load_image(image_path, resize_to=(256, 256), patch_size=None):
    """
    Load the image and resize.
    
    parameter:
        image_path
    return:
        image
    """
    transform_list = [
        transforms.Resize(resize_to),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]
    if patch_size:
        transform_list.insert(0, transforms.RandomCrop(patch_size))

    transform = transforms.Compose(transform_list)

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
 
    return image

def save_generated_image(tensor, path):
    """Save the generated image to the specified path."""
    tensor = tensor.clone().detach().cpu()
    tensor = (tensor * 0.5) + 0.5  # Denormalize to [0, 1]
    save_image(tensor, path)

def evaluate(generated_ihc_image_path, real_ihc_image_path):
    """
    Evaluate the score of generated images from our model.
    
    parameter:
        generated_ihc_image_path: the path of generated IHC images,
        real_ihc_image_path: the path of real IHC images,
    return:
        LPIPS score
        PSNR score
        SSIM score
    """
    # Load LPIPS model for perceptual similarity
    lpips_model = lpips.LPIPS(net='alex')

    # Load images
    generated_ihc_image = load_image(generated_ihc_image_path)
    real_ihc_image = load_image(real_ihc_image_path)

    # Move to GPU if available
    if torch.cuda.is_available():
        lpips_model = lpips_model.cuda()
        generated_ihc_image = generated_ihc_image.cuda()
        real_ihc_image = real_ihc_image.cuda()

    # Calculate LPIPS
    lpips_score = lpips_model(generated_ihc_image, real_ihc_image)
    lpips_score = lpips_score.item()  # Convert to a float
    # print(f"LPIPS: {lpips_score}")

    # Move tensors to CPU and convert for PSNR and SSIM
    generated_ihc_image_np = generated_ihc_image.squeeze().cpu().permute(1, 2, 0).numpy()
    real_ihc_image_np = real_ihc_image.squeeze().cpu().permute(1, 2, 0).numpy()

    # Calculate PSNR
    psnr_score = compare_psnr(real_ihc_image_np, generated_ihc_image_np, data_range=1)
    # print(f"PSNR: {psnr_score}")
    # print(real_ihc_image_np.shape, generated_ihc_image_np.shape)
    # Calculate SSIM
    ssim_score = compare_ssim(real_ihc_image_np, generated_ihc_image_np, multichannel=True, data_range=1.0, channel_axis=-1)
    # print(f"SSIM: {ssim_score}")

    return lpips_score, psnr_score, ssim_score

def generate_and_evaluate(model, he_image_path, ihc_image_path, generated_image_path, device):
    """Generate image using the model and evaluate it."""
    # Load input image
    he_image = load_image(he_image_path).to(device)

    # Generate image
    generated_image, _ = model.sample(he_image, 200)

    # Save generated image
    save_generated_image(generated_image, generated_image_path)

    # Evaluate the generated image
    lpips_score, psnr_score, ssim_score = evaluate(generated_image_path, ihc_image_path)
    return lpips_score, psnr_score, ssim_score

def evaluate_dataset(generator, he_dir, ihc_dir, generated_dir, device):
    """Evaluate the model on a dataset."""
    he_images = [f for f in os.listdir(he_dir) if f.endswith('.png')]
    ihc_images = [f for f in os.listdir(ihc_dir) if f.endswith('.png')]

    lpips_scores, psnr_scores, ssim_scores = [], [], []

    for image_name in tqdm(he_images):
        he_image_path = os.path.join(he_dir, image_name)
        ihc_image_path = os.path.join(ihc_dir, image_name)
        generated_image_path = os.path.join(generated_dir, image_name)

        if image_name in ihc_images:
            lpips_score, psnr_score, ssim_score = generate_and_evaluate(
                generator, he_image_path, ihc_image_path, generated_image_path, device)
            lpips_scores.append(lpips_score)
            psnr_scores.append(psnr_score)
            ssim_scores.append(ssim_score)
        else:
            print(f"Skipping {image_name}, corresponding IHC image not found.")

    avg_lpips = np.mean(lpips_scores)
    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)

    return avg_lpips, avg_psnr, avg_ssim

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # load model
    checkpoint_path = 'saved_models/diffae_final_2_best.pt'
    model = load_named_model('diff_ae', checkpoint_path)

    # test dataset
    test_HE_image_path = '/home/karl/practical_bmcv/data/BCI_dataset_512/HE/test'
    test_IHC_image_path = '/home/karl/practical_bmcv/data/BCI_dataset_512/IHC/test'
    
    save_generated_dir = '/home/karl/practical_bmcv/data/generated_diffusion'
    os.makedirs(save_generated_dir, exist_ok=True)

    avg_lpips, avg_psnr, avg_ssim = evaluate_dataset(model, test_HE_image_path, test_IHC_image_path, save_generated_dir, device)
    print(f"Average LPIPS: {avg_lpips}, Average PSNR: {avg_psnr}, Average SSIM: {avg_ssim}")