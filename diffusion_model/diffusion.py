import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from diffusion_model.model import UNet
# from model import UNet

class DiffusionModel:
    def __init__(self, model, timesteps=2000):
        self.optimizer = optim.Adam(model.parameters(), lr=2e-5)
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.1, patience=10)
        self.scaler = GradScaler()

        self.model = model
        self.timesteps = timesteps
        self.betas = self._cosine_beta_schedule(1e-6, 0.01, timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim = 0).to(device=self.model.device)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars).to(device=self.model.device)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars).to(device=self.model.device)
        self.alpha_bars_prev = F.pad(self.alpha_bars[:-1], (1, 0), value=1.0)
        self.posterior_variance = self.betas * (1. - self.alpha_bars_prev) / (1. - self.alpha_bars)
        self.posterior_log_variance = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.eta = 0.0

    def parameters(self):
        return self.model.parameters()
    
    def _linear_beta_schedule(self, start, end, timesteps):
        return torch.linspace(start, end, timesteps, device=self.model.device)
    
    def _cosine_beta_schedule(self, start, end, timesteps):
        return start + 0.5 * (end - start) * (1 + torch.cos(torch.linspace(0, 3.14159, timesteps, device=self.model.device)))
    
    def q_sample(self, x, t, cond = None):
        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        assert sqrt_alpha_bar_t.shape == (x.shape[0], 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        assert sqrt_one_minus_alpha_bar_t.shape == (x.shape[0], 1, 1, 1)

        if cond is not None: # Predict noise form model
            noise = self.model(x, t, cond)
        else:
            noise = torch.randn_like(x, device=self.model.device)

        x_noisy = sqrt_alpha_bar_t * x + sqrt_one_minus_alpha_bar_t * noise

        return noise, x_noisy
    
    @torch.no_grad()
    def p_sample_ddim(self, x, t, t_prev, cond):
        alpha_bar_t = self.alpha_bars[t]
        alpha_bar_t_prev = self.alpha_bars[t_prev] if t_prev >= 0 else self.alpha_bars[0]
        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t].unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        sqrt_alpha_bar_t_prev = self.sqrt_alpha_bars[t_prev].unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(3)

        noise_pred = self.model(x, t, cond)
        mean_pred = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)

        x_prev = torch.sqrt(alpha_bar_t_prev) * mean_pred + torch.sqrt(1 - alpha_bar_t_prev) * noise_pred
        if self.eta > 0:
            noise = torch.randn_like(x, device=self.model.device)
            sigma_t = self.eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev))
            x_prev = x_prev + sigma_t * noise
        return x_prev
    
    @torch.no_grad()
    def p_sample_ddpm(self, x, t, cond):
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t].unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        self.model.eval()
        noise_pred = self.model(x, t, cond)
        mean = 1 / torch.sqrt(self.alphas[t]) * (x - self.betas[t] / sqrt_one_minus_alpha_bar_t * noise_pred)
        if t == 0:
            return mean
        else:
            return mean + torch.exp(0.5 * self.posterior_log_variance[t]) * torch.randn_like(x, device=self.model.device)

    @torch.no_grad()
    def sample(self, cond, num_steps=None, debug=False, use_ddim=True):
        sample_list = []
        shape = cond.size()
        if num_steps is None:
            num_steps = self.timesteps
        x = torch.randn(shape, device=self.model.device)
        y = torch.zeros_like(x, device=self.model.device)
        for t in reversed(range(num_steps)):
            t = torch.tensor(t, device=self.model.device)
            t_prev = t - 1
            if use_ddim:
                x = self.p_sample_ddim(x, t, t_prev, cond)
            else:
                x = self.p_sample_ddpm(x, t, cond)
            if debug and t % 100 == 0: # Visualize the backward denoising process
                sample_list.append(x)
        return x if not debug else sample_list
    
    def compute_loss(self, data, debug=False):
        cond, x_start = data
        cond = cond.to(self.model.device)
        x_start = x_start.to(self.model.device)
        t = torch.randint(0, self.timesteps, (x_start.size(0),), device=self.model.device)
        noise, x_noisy = self.q_sample(x_start, t)
        noise_pred, x_noisy_pred = self.q_sample(x_noisy, t, cond)

        loss_diffusion = F.mse_loss(noise_pred, noise)

        loss = loss_diffusion

        if debug: # Visualize the forward process
            return loss, x_noisy, x_noisy_pred, t
        else:
            return loss
    
    def train(self, data):
        self.model.train()

        self.optimizer.zero_grad()

        loss = self.compute_loss(data)

        loss.backward()
        self.optimizer.step()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        return loss

    def eval(self, data, debug=False):
        self.model.eval()
        with torch.no_grad():
            if debug:
                loss, noisy_x, noisy_x_pred, sampled_t = self.compute_loss(data, debug)
                return loss, noisy_x[0], noisy_x_pred[0], sampled_t[0]
            else:
                loss = self.compute_loss(data)
                return loss
    
    def visualize(self, data, debug=True):
        cond, x_start = data
        cond = cond.to(self.model.device)
        x_start = x_start.to(self.model.device)
        x_recon = self.sample(cond, None, debug)
        return cond, x_start, x_recon
       
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

def build_model(device, use_cond, timesteps):
    cond_dim = 3 if use_cond else 0
    denoisingModel = UNet(3, 3, device=device, cond_dim=cond_dim)
    model = DiffusionModel(denoisingModel, timesteps)
    return model
        
if __name__ == '__main__':
    print('Testing diffusion model')
    model = UNet(3, 3, 'cuda')
    diffusion = DiffusionModel(model)
    diffusion.load('saved_models/test_best.pt')
    diffusion.model.eval()

    from PIL import Image
    import torchvision.transforms as transforms
    import torchvision

    testHE = Image.open('data/BCI_dataset/HE/train/00006_train_2+.png')
    testIHC = Image.open('data/BCI_dataset/IHC/train/00006_train_2+.png')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testHE = transform(testHE).unsqueeze(0).to('cuda')
    testIHC = transform(testIHC).unsqueeze(0).to('cuda')
    data = (testHE, testIHC)
    x_recons = diffusion.sample(testHE, 1000, False, use_ddim=True)

    testHE = testHE * 0.5 + 0.5
    testIHC = testIHC * 0.5 + 0.5
    x_recons = [x * 0.5 + 0.5 for x in x_recons]
    x_recons = [torch.clamp(x, 0, 1) for x in x_recons]
    x_recons = torch.cat(x_recons)
    torchvision.utils.save_image(torch.cat((testHE, testIHC), dim=0), 'data.png', nrow=2)
    torchvision.utils.save_image(x_recons, 'sample.png', nrow=5)