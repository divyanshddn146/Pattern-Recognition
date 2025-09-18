# vae_visualization_save_debug.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import matplotlib
matplotlib.use('Agg')  # Ensures compatibility without display
import matplotlib.pyplot as plt
import umap
import numpy as np

# ======================
# Config
# ======================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 128
latent_dim = 256
image_size = 64
model_path = 'conv_vae_oasis.pth'  # Path to your trained VAE
data_path = '/home/groups/comp3710/OASIS/keras_png_slices_train/'  # Training images
save_dir = './vae_outputs'  # Directory to save images
os.makedirs(save_dir, exist_ok=True)

print(f"Files will be saved in: {os.path.abspath(save_dir)}")

# ======================
# Dataset Class
# ======================
class MRIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('L')
        if self.transform:
            img = self.transform(img)
        return img

# ======================
# Dataset and Dataloader
# ======================
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])
dataset = MRIDataset(data_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

# ======================
# VAE Model
# ======================
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(ConvVAE, self).__init__()
        self.enc_conv1 = nn.Conv2d(1, 32, 4, 2, 1)
        self.enc_conv2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.enc_conv3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.enc_conv4 = nn.Conv2d(128, 256, 4, 2, 1)
        self.fc_mu = nn.Linear(256*4*4, latent_dim)
        self.fc_logvar = nn.Linear(256*4*4, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 256*4*4)
        self.dec_conv1 = nn.ConvTranspose2d(256,128,4,2,1)
        self.dec_conv2 = nn.ConvTranspose2d(128,64,4,2,1)
        self.dec_conv3 = nn.ConvTranspose2d(64,32,4,2,1)
        self.dec_conv4 = nn.ConvTranspose2d(32,1,4,2,1)

    def encode(self, x):
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = F.relu(self.enc_conv3(h))
        h = F.relu(self.enc_conv4(h))
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = self.fc_dec(z).view(-1, 256, 4, 4)
        h = F.relu(self.dec_conv1(h))
        h = F.relu(self.dec_conv2(h))
        h = F.relu(self.dec_conv3(h))
        return torch.sigmoid(self.dec_conv4(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# ======================
# Load Model
# ======================
model = ConvVAE(latent_dim=latent_dim).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("Model loaded successfully.")

# ======================
# Encode Dataset
# ======================
latent_vectors = []
with torch.no_grad():
    for batch in dataloader:
        batch = batch.to(device)
        mu, logvar = model.encode(batch)
        z = model.reparameterize(mu, logvar)
        latent_vectors.append(z.cpu())
latent_vectors = torch.cat(latent_vectors, dim=0).numpy()
print(f"Encoded {latent_vectors.shape[0]} images into latent vectors of size {latent_vectors.shape[1]}.")

# each MRI slice is now represented as a 256-dim vector in latent space.

# ======================
# UMAP 2D Reduction
# ======================
reducer = umap.UMAP(n_components=2, random_state=42)
latent_2d = reducer.fit_transform(latent_vectors)
print("UMAP reduction completed.")
#UMAP takes your 256-D latent vectors → reduces them into 2D coordinates.

# ======================
# Save UMAP Scatter Plot
# ======================
umap_path = os.path.join(save_dir, "latent_umap.png")
plt.figure(figsize=(8,6))
plt.scatter(latent_2d[:,0], latent_2d[:,1], s=5, alpha=0.7)
plt.title("VAE Latent Space Manifold (UMAP)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.savefig(umap_path)
plt.close()
print(f"Saved UMAP plot to {umap_path}")
assert os.path.exists(umap_path), "UMAP image not saved!"

# ======================
# Sample from latent space to generate images
# ======================
n_samples = 16
z_samples = torch.randn(n_samples, latent_dim).to(device)
with torch.no_grad():
    generated = model.decode(z_samples).cpu()

gen_path = os.path.join(save_dir, "generated_samples.png")
fig, axes = plt.subplots(4, 4, figsize=(8,8))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(generated[i,0], cmap='gray')
    ax.axis('off')
plt.suptitle("Generated Samples from Latent Space")
plt.savefig(gen_path)
plt.close()
print(f"Saved generated samples to {gen_path}")
assert os.path.exists(gen_path), "Generated samples image not saved!"
