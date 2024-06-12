import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from diffusion.model import UNet

class DiffusionModel:
    def __init__(self, model, timesteps=200):
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4)
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.1, patience=10)
        self.scaler = GradScaler()

        self.model = model
        self.timesteps = timesteps
        self.betas = self._linear_beta_schedule(timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim = 0).to(device=self.model.device)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bars).to(device=self.model.device)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bars).to(device=self.model.device)

    def parameters(self):
        return self.model.parameters()
    
    def _linear_beta_schedule(self, timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps, device=self.model.device)
    
    def _cosine_beta_schedule(self, timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        betas = beta_start + 0.5 * (beta_end - beta_start) * (1 + torch.cos(torch.linspace(0, 1, timesteps) * 3.14159))
        return betas.to(self.model.device)
    
    def q_sample(self, x, t, cond = None):
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].unsqueeze(1).unsqueeze(2).unsqueeze(3)

        if cond is not None: # Predict noise form model
            noise = self.model(x, t, cond)
        else:
            noise = torch.randn_like(x, device=self.model.device)

        x_noisy = sqrt_alpha_bar_t * x + sqrt_one_minus_alpha_bar_t * noise

        return noise, x_noisy
    
    @torch.no_grad()
    def p_sample(self, x, t, cond):
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        self.model.eval()
        noise_pred = self.model(x, t, cond)
        return (
            1 / torch.sqrt(self.alphas[t]) * (x - self.betas[t] / sqrt_one_minus_alpha_bar_t * noise_pred)
        )

    @torch.no_grad()
    def sample(self, cond, num_steps=None, debug=False):
        sample_list = []
        shape = cond.size()
        if num_steps is None:
            num_steps = self.timesteps
        x = torch.randn(shape, device=self.model.device)
        for t in reversed(range(num_steps)):
            t = torch.tensor(t, device=self.model.device)
            x = self.p_sample(x, t, cond)
            if debug and t % 20 == 0: # Visualize the backward denoising process
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

        # self.scaler.scale(loss).backward()
        # self.scaler.step(self.optimizer)
        # self.scaler.update()

        # self.scheduler.step(loss)

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

def build_model(name, device):
    if name == 'diffusion':
        denoisingModel = UNet(3, 3, device=device)
        model = DiffusionModel(denoisingModel)
    return model
        
if __name__ == '__main__':
    model = UNet(3, 3, 'cuda')
    diffusion = DiffusionModel(model)
    diffusion.load('saved_models/test_950.pt')
    diffusion.model.eval()

    from PIL import Image
    import torchvision.transforms as transforms
    import torchvision

    testHE = Image.open('data/BCI_dataset/HE/test/00002_test_2+.png').convert('RGB')
    testIHC = Image.open('data/BCI_dataset/IHC/test/00002_test_2+.png').convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testHE = transform(testHE).unsqueeze(0).to('cuda')
    testIHC = transform(testIHC).unsqueeze(0).to('cuda')
    data = (testHE, testIHC)
    x_recons = diffusion.sample(testHE, None, True)

    testHE = testHE * 0.5 + 0.5
    testIHC = testIHC * 0.5 + 0.5
    x_recons = [x * 0.5 + 0.5 for x in x_recons]
    x_recons = [torch.clamp(x, 0, 1) for x in x_recons]
    x_recons = torch.cat(x_recons)
    torchvision.utils.save_image(torch.cat((testHE, testIHC), dim=0), 'data.png', nrow=2)
    torchvision.utils.save_image(x_recons, 'sample.png', nrow=5)