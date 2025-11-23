# train_ssrw_v2.py
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import lpips
from PIL import Image
import glob
import kornia.losses as KL
from tqdm import tqdm

from ssrw_v2 import CrossAttentionEncoder, EnhancedUNetDecoder, Discriminator, NoiseLayer

# -------------------------------------------------------
# Dataset
# -------------------------------------------------------
class SimpleImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.paths = glob.glob(os.path.join(root,"*.jpg")) + \
                     glob.glob(os.path.join(root,"*.png"))
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
        except:
            return torch.zeros(3,128,128)
        if self.transform:
            img = self.transform(img)
        return img

# -------------------------------------------------------
# SSIM Loss
# -------------------------------------------------------
def ssim_loss(a,b):
    return 1 - KL.ssim(a,b,window_size=11,reduction='mean')

# -------------------------------------------------------
# Train
# -------------------------------------------------------
def train(config_path):
    with open(config_path,'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["device"])

    encoder = CrossAttentionEncoder(
        img_size=cfg['data']['image_size'],
        patch_size=cfg['model']['patch_size'],
        watermark_len=cfg['data']['watermark_len'],
        embed_dim=cfg['model']['embed_dim'],
        num_heads=cfg['model']['num_heads'],
        hidden_dim=cfg['model']['hidden_dim']
    ).to(device)

    decoder = EnhancedUNetDecoder(cfg['data']['watermark_len']).to(device)
    disc = Discriminator().to(device)
    noise = NoiseLayer(cfg).to(device)

    lpips_fn = lpips.LPIPS(net="alex").to(device)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    opt_ED = optim.Adam(list(encoder.parameters())+list(decoder.parameters()),
                        lr=cfg["training"]["learning_rate"])
    opt_D = optim.Adam(disc.parameters(), lr=cfg["training"]["learning_rate"])

    transform = transforms.Compose([
        transforms.Resize((cfg['data']['image_size'], cfg['data']['image_size'])),
        transforms.ToTensor()
    ])
    dataset = SimpleImageDataset(cfg['data']['train_dir'], transform)
    loader = DataLoader(dataset, batch_size=cfg['data']['batch_size'], shuffle=True)

    w = cfg['loss_weights']

    for epoch in range(cfg['training']['epochs']):
        encoder.train(); decoder.train(); disc.train()

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for images in pbar:
            images = images.to(device)
            B = images.size(0)
            wm = torch.randint(0,2,(B,cfg['data']['watermark_len'])).float().to(device)

            # --- Train Discriminator ---
            opt_D.zero_grad()
            with torch.no_grad():
                fake = encoder(images, wm)
            d_real = disc(images)
            d_fake = disc(fake)
            loss_D = 0.5*(bce(d_real, torch.ones_like(d_real)) + 
                          bce(d_fake, torch.zeros_like(d_fake)))
            loss_D.backward()
            opt_D.step()

            # --- Train Encoder + Decoder ---
            opt_ED.zero_grad()

            enc = encoder(images, wm)
            enc_attacked = noise(enc)
            decoded = decoder(enc_attacked)
            d_fake2 = disc(enc)

            l_l2 = mse(enc, images)
            l_lpips = lpips_fn(enc, images).mean()
            l_ssim = ssim_loss(enc, images)
            l_dec = bce(decoded, wm)
            l_adv = bce(d_fake2, torch.ones_like(d_fake2))

            total = (w['lambda_l2'] * l_l2 +
                     w['lambda_lpips'] * l_lpips +
                     w['lambda_ssim'] * l_ssim +
                     w['lambda_dec'] * l_dec +
                     w['lambda_disc'] * l_adv)

            total.backward()
            opt_ED.step()

            acc = ((torch.sigmoid(decoded)>0.5)==wm).float().mean().item()
            pbar.set_postfix({"loss":f"{total.item():.3f}","acc":f"{acc:.3f}"})

        # SAVE CHECKPOINT
        if (epoch+1) % cfg["training"]["save_interval"] == 0:
            save_dir = cfg['training'].get('output_dir', '.')
            os.makedirs(save_dir, exist_ok=True)
            torch.save(encoder.state_dict(), f"{save_dir}/encoder_v2_epoch{epoch+1}.pt")
            torch.save(decoder.state_dict(), f"{save_dir}/decoder_v2_epoch{epoch+1}.pt")

    # Final save
    out = cfg['training'].get('output_dir','.')
    torch.save(encoder.state_dict(), f"{out}/encoder_v2_final.pt")
    torch.save(decoder.state_dict(), f"{out}/decoder_v2_final.pt")
    print("Training finished.")
    
if __name__ == "__main__":
    train("ssrw_v2.yaml")
