import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch import optim
from PIL import Image
import os
from tqdm import tqdm

# ======================
# Config
# ======================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 128        # Large batch size for A100
epochs = 50
latent_dim = 256       # Increased latent dimension for MR images
learning_rate = 1e-3
image_size = 64         # Resize images to 64x64

# ======================
# Custom Dataset
# ======================
class MRIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Collect all PNG images from directory
        self.images = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('L')  # Convert to grayscale
        if self.transform:
            img = self.transform(img)
        return img  # No labels, just images

# ======================
# Dataset and DataLoader
# ======================
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),           # Scales pixels to [0,1]
])

train_dataset = MRIDataset('/home/groups/comp3710/OASIS/keras_png_slices_train/', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)

val_dataset = MRIDataset('/home/groups/comp3710/OASIS/keras_png_slices_validate/', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

# ======================
# VAE Model
# ======================
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(ConvVAE, self).__init__()
        # Encoder
        #Encoder: stack of Conv layers → shrink 64×64 → 4×4 feature map.
        self.enc_conv1 = nn.Conv2d(1, 32, 4, 2, 1)
        self.enc_conv2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.enc_conv3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.enc_conv4 = nn.Conv2d(128, 256, 4, 2, 1)
        self.fc_mu = nn.Linear(256*4*4, latent_dim)
        self.fc_logvar = nn.Linear(256*4*4, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 256*4*4)
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.dec_conv4 = nn.ConvTranspose2d(32, 1, 4, 2, 1)

    def encode(self, x):
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = F.relu(self.enc_conv3(h))
        h = F.relu(self.enc_conv4(h))
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
        # You don’t want to store the full image (too big).

        # Instead, you want a compact description:

        # mu (mean): the “average face” representation of a person.

        # variance (from logvar): how much that face can vary (smile, lighting, glasses, etc.).

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        # If you only store mu, every time you reconstruct you’d get the exact same average face → blurry results.

        # So instead:

        # You sample a little bit of random noise eps (like adding wiggle room).

        # Scale that noise by the standard deviation std = exp(0.5*logvar).

        # Add it to mu.

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
# Loss Function
# ======================
def vae_loss(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='sum')  # Better for MR images
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

# ======================
# Initialize Model
# ======================
model = ConvVAE(latent_dim=latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler()  # Mixed Precision

# ======================
# Training Loop
# ======================
for epoch in range(1, epochs+1):
    # --- Training ---
    model.train()
    train_loss = 0
    loop = tqdm(train_loader, desc=f'Epoch [{epoch}/{epochs}] Training')
    for batch in loop:
        batch = batch.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    avg_train_loss = train_loss / len(train_dataset)
    
    # --- Validation ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            recon, mu, logvar = model(batch)
            val_loss += vae_loss(recon, batch, mu, logvar).item()
    avg_val_loss = val_loss / len(val_dataset)
    
    print(f'Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')

# ======================
# Save Model
# ======================
torch.save(model.state_dict(), "conv_vae_oasis.pth")
print("Model saved successfully.")
