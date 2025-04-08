import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
# Anomaly Detection in Medical Imaging Using Diffusion Models

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = nn.Conv2d(1, 64, 3, padding=1)
        self.down2 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.up1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.out = nn.Conv2d(64, 1, 3, padding=1)
    
    def forward(self, x, t):
        x1 = torch.relu(self.down1(x))
        x2 = torch.relu(self.down2(x1))
        x3 = torch.relu(self.up1(x2))
        return self.out(x3)

class DDPM:
    def __init__(self, steps=50):  # 50 steps to save time
        self.steps = steps
        self.betas = torch.linspace(0.0001, 0.02, steps).cuda()
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, 0)

    def add_noise(self, x, t):
        noise = torch.randn_like(x)
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1)
        return torch.sqrt(alpha_bar) * x + torch.sqrt(1 - alpha_bar) * noise, noise

# model = UNet().cuda()
# ddpm = DDPM()

# Anomaly Detection
def detect_anomalies(model, image, process, threshold=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        t = torch.randint(0, process.num_steps, (1,)).to(device)
        noisy_image, _ = process.add_noise(image.to(device), t)
        denoised_image = model(noisy_image, t)
        error = nn.MSELoss()(denoised_image, image)
        return "Anomalous" if error > threshold else "Normal"

# Example Usage (assuming dataset is prepared)
# train_diffusion_model(model, dataloader)
# result = detect_anomalies(model, test_image, process)
