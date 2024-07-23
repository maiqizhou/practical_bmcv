import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

import sys
sys.path.append('/home/karl/practical_bmcv/')
from diff_ae.model import UNet, SemanticEncoder

class DiffusionModel:
    def __init__(self, model, encoder, timesteps=1000):
        # Weight decay of 0.01
        self.optimizer = optim.Adam(model.parameters(), lr=2e-5)
        # self.scaler = GradScaler()

        self.model = model
        self.encoder = encoder
        self.timesteps = timesteps
        self.betas = self._linear_beta_schedule(1e-4, 0.02, timesteps)
        assert self.betas.dim() == 1
        assert (self.betas > 0).all() and (self.betas < 1).all()
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(device=self.model.device)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars).to(device=self.model.device)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars).to(device=self.model.device)
        self.alpha_bars_prev = F.pad(self.alpha_bars[:-1], (1, 0), value=1.0)


    def parameters(self):
        return self.model.parameters()
    
    def _linear_beta_schedule(self, start, end, timesteps):
        return torch.linspace(start, end, timesteps, device=self.model.device)
    
    def _cosine_beta_schedule(self, start, end, timesteps):
        return start + 0.5 * (end - start) * (1 + torch.cos(torch.linspace(0, 3.14159, timesteps, device=self.model.device)))
    
    def q_sample(self, x, t, noise=None, use_ddim=True):
        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        assert sqrt_alpha_bar_t.shape == (x.shape[0], 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        assert sqrt_one_minus_alpha_bar_t.shape == (x.shape[0], 1, 1, 1)

        if noise is None:
            noise = torch.randn_like(x, device=self.model.device)

        x_noisy = sqrt_alpha_bar_t * x + sqrt_one_minus_alpha_bar_t * noise

        return x_noisy
    
    @torch.no_grad()
    def p_sample_ddim(self, x, t, cond, eta=0.0):
        alpha_t = self.alphas[t].unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        alpha_bar_t = self.alpha_bars[t].unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        alpha_bar_t_prev = self.alpha_bars_prev[t].unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        assert x.dim() == alpha_bar_t.dim() == alpha_bar_t_prev.dim() == 4

        sigma = eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev))

        noise_pred = self.model(x, t, cond)
        x_0_pred = self.predict_x_0(x, t, noise_pred)
        # Normalize the output by scaling it to [-1, 1]
        # x_0_pred = torch.clamp(x_0_pred, -1, 1)

        mean_pred = torch.sqrt(alpha_bar_t_prev) * x_0_pred + torch.sqrt(1 - alpha_bar_t_prev - sigma ** 2) * noise_pred
        noise = torch.randn_like(x, device=self.model.device)
        if t > 0:
            x_prev = mean_pred + sigma * noise
        else:
            x_prev = mean_pred

        return x_prev, x_0_pred

    @torch.no_grad()
    def sample(self, cond, num_steps=None, debug=False, use_cond=True, eta=0.0):
        sample_list = []
        x_0_list = []
        shape = cond.size()
        cond = cond.to(self.model.device)
        if num_steps is None:
            num_steps = self.timesteps
        x_t = torch.randn(shape, device=self.model.device) # TODO: Sample from partially noisy image
        for t in reversed(range(num_steps)):
            t = torch.tensor(t, device=self.model.device)
            if use_cond:
                z_cond = self.encoder(cond, t) if use_cond else None
                x_t, x_0 = self.p_sample_ddim(x_t, t, z_cond, eta)
            else:
                # x_t, x_0 = self.p_sample_ddim(x_t, t, None, eta)
                pass

            if debug and t % (num_steps // 10) == 0:
                if sample_list:
                    if torch.allclose(x_t, sample_list[-1]):
                        print('Sample is the same')
                    else:
                        # print(torch.min(x_t), torch.max(x_t), torch.mean(x_t))
                        print(torch.min(x_0), torch.max(x_0), torch.mean(x_0))
                        outrange = torch.logical_or(x_0 < -1, x_0 > 1)
                        print(torch.sum(outrange) / outrange.numel())
                        # print(torch.mean(torch.abs(x_t - sample_list[-1])))
                        pass
                sample_list.append(x_t)
                x_0_list.append(x_0)
        return (x_t, x_0) if not debug else (sample_list, x_0_list)
    
    def predict_x_0(self, x, t, noise_pred):
        if t.dim() == 0:
            t = t.unsqueeze(0)
        alpha_bar_t = self.alpha_bars[t].unsqueeze(1).unsqueeze(2).unsqueeze(3)

        x_0 = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        return torch.clamp(x_0, -1, 1)
    
    def compute_loss(self, data, debug=False, use_ddim=True):
        cond, x_start = data
        cond = cond.to(self.model.device)
        x_start = x_start.to(self.model.device)
        t = torch.randint(0, self.timesteps, (x_start.size(0),), device=self.model.device)
        z_cond = self.encoder(cond, t)
        noise = torch.randn_like(x_start, device=self.model.device)
        x_noisy = self.q_sample(x_start, t, noise, use_ddim=use_ddim)
        noise_pred = self.model(x_noisy, t, cond=z_cond)
        x_0_pred = self.predict_x_0(x_noisy, t, noise_pred)

        # loss_reconstruction = F.mse_loss(x_0_pred, x_start)
        loss_diffusion = F.mse_loss(noise_pred, noise)

        loss = {
            # 'recon': loss_reconstruction,
            'diff': loss_diffusion
        }

        if debug: # Visualize the forward process
            return loss, x_noisy, x_0_pred, t
        else:
            return loss
    
    def train(self, data):
        self.model.train()

        self.optimizer.zero_grad()

        loss = self.compute_loss(data)

        loss_term = 0
        for key in loss:
            loss_term += loss[key]

        loss_term.backward()
        self.optimizer.step()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        return loss

    def eval(self, data, debug=False):
        self.model.eval()
        with torch.no_grad():
            if debug:
                loss, noisy_x, x_0_pred, sampled_t = self.compute_loss(data, debug)
                return loss, noisy_x[0], x_0_pred[0], sampled_t[0]
            else:
                loss = self.compute_loss(data)
                return loss
    
    def visualize(self, data, debug=True):
        cond, x_start = data
        cond = cond.to(self.model.device)
        x_start = x_start.to(self.model.device)
        x_recon, _ = self.sample(cond, None, debug)
        data_img = torch.cat((cond, x_start), dim=0)
        return data_img, x_recon

    def save(self, path):
        torch.save((self.model.state_dict(), self.encoder.state_dict()), path)

    def load(self, path):
        model_state, encoder_state = torch.load(path)
        self.model.load_state_dict(model_state)
        self.encoder.load_state_dict(encoder_state)

def build_model(device, cond_dim, timesteps):
    denoisingModel = UNet(3, 3, device=device, cond_dim=cond_dim)
    semanticEncoder = SemanticEncoder(3, z_dim=cond_dim, device=device)
    model = DiffusionModel(denoisingModel, semanticEncoder, timesteps)
    return model
        
if __name__ == '__main__':
    print('Testing diffusion model')

    diffusion = build_model('cuda', 512, 200)
    diffusion.load('saved_models/diffae_final_2_best.pt')
    diffusion.model.eval()

    from PIL import Image
    import torchvision.transforms as transforms
    import torchvision

    testHE = Image.open('data/BCI_dataset/HE/train/00006_train_2+.png')
    testIHC = Image.open('data/BCI_dataset/IHC/train/00006_train_2+.png')
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize((128, 128)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    def random_crop(image_pair, patch_size):
        import numpy as np
        he_image, ihc_image = image_pair
        assert patch_size < min(he_image.size), 'Patch size should be less than the minimum dimension of the image'
        w, h = he_image.size
        th, tw = patch_size, patch_size
        if w == tw and h == th:
            return he_image, ihc_image
        i = np.random.randint(0, h - th)
        j = np.random.randint(0, w - tw)
        return he_image.crop((j, i, j + tw, i + th)), ihc_image.crop((j, i, j + tw, i + th))
    testHE, testIHC = random_crop((testHE, testIHC), 128)
    
    testHE = transform(testHE).unsqueeze(0).to('cuda')
    testIHC = transform(testIHC).unsqueeze(0).to('cuda')
    data = (testHE, testIHC)
    # import sys
    # sys.path.append('/home/karl/practical_bmcv/')
    # from utils.utils import *
    # cifar10, _ = build_CIFAR10_dataloader(batch_size=2, normalize=True)
    # data = next(iter(cifar10))
    data = (data[0][0].unsqueeze(0), data[1][0].unsqueeze(0))
    cond = data[0]
    x_recons, x_0s = diffusion.sample(cond, 200, debug=True, eta=0)

    # testHE = testHE * 0.5 + 0.5
    # testIHC = testIHC * 0.5 + 0.5
    # batch[0] = batch[0] * 0.5 + 0.5
    x_recons = [x * 0.5 + 0.5 for x in x_recons]
    # x_recons = [torch.clamp(x, 0, 1) for x in x_recons]
    x_0s = [x * 0.5 + 0.5 for x in x_0s]
    # x_0s = [torch.clamp(x, 0, 1) for x in x_0s]
    # x_recons = torch.cat(x_recons)
    result = torch.cat([(cond * 0.5 + 0.5).to('cuda'), (data[1] * 0.5 + 0.5).to('cuda'), x_recons[-1], x_0s[-1]], dim=0)
    recons = torch.cat(x_recons, dim=0)
    x_0s = torch.cat(x_0s, dim=0)
    torchvision.utils.save_image(result, 'result.png', nrow=4)
    torchvision.utils.save_image(recons, 'recons.png', nrow=4)
    torchvision.utils.save_image(x_0s, 'x_0s.png', nrow=4)