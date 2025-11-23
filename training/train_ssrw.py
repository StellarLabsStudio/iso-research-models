import argparse
from json import encoder
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import lpips
import glob
from PIL import Image
from tqdm import tqdm

# Import Architecture from Local Models Directory
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
from models.ssrw import CrossAttentionEncoder, EnhancedUNetDecoder, Discriminator, NoiseLayer

class SimpleImageDataset(Dataset):
    
    """
    Custom dataset to load images from a flat directory (no subfolders).
    """
    
    def __init__ (self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Find all jpg/png images in the directory
        self.image_paths = glob.glob(os.path.join(root_dir, '*.jpg')) + glob.glob(os.path.join(root_dir, '*.png'))
    
    def __len__ (self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Warning: Could not load image {img_path} - {e}")
            # Return a black image of correct size if load fails to prevent crash
            return torch.zeros(3, 128, 128)
    
def train (config_path):
    # Load Configuration
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device(cfg['device'])
    
    # --- Initialize Models ---
    encoder = CrossAttentionEncoder(
        img_size=cfg['data']['image_size'],
        patch_size=cfg['model']['patch_size'],
        watermark_len=cfg['data']['watermark_len'],
    ).to(device)
    
    decoder = EnhancedUNetDecoder(watermark_len=cfg['data']['watermark_len']).to(device)
    discriminator = Discriminator().to(device)
    noise_layer = NoiseLayer(cfg).to(device)
    
    # [cite-start] -- Loss Functions [cite: 188-192]
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss() # For watermark and discriminator
    
    # --- Optimizers ---
    opt_enc_dec = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=cfg['training']['learning_rate'])
    opt_disc = optim.Adam(discriminator.parameters(), lr=cfg['training']['learning_rate'])
    
    # Weights
    w = cfg["loss_weights"]
    
    # --- Data Loading ---
    transform = transforms.Compose([
        transforms.Resize((cfg['data']['image_size'], cfg['data']['image_size'])),
        transforms.ToTensor()
    ])
    
    print(f"Loading training data from: {cfg['data']['train_dir']}")
    train_dataset = SimpleImageDataset(cfg['data']['train_dir'], transform=transform)
    
    if len(train_dataset) == 0:
        raise ValueError(f"No images found in {cfg['data']['train_dir']}. Did you run setup_datasets.py?")

    dataloader = DataLoader(
        train_dataset, 
        batch_size=cfg['data']['batch_size'], 
        shuffle=True,
        num_workers=cfg['data']['num_workers']
    )
    
    print(f"Starting training loop with {len(train_dataset)} images...")
    
    # --- Training Loop ---
    for epoch in range(cfg['training']['epochs']):
        encoder.train()
        decoder.train()
        discriminator.train()
        
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg['training']['epochs']}")
        
        for images in progress_bar:
            images = images.to(device)
            current_batch_size = images.size(0)
            
            # Generate random watermarks on the fly for each batch
            # [cite_start]Paper: "random watermark sequence... 64 bits" [cite: 173]
            watermarks = torch.randint(0, 2, (current_batch_size, cfg['data']['watermark_len'])).float().to(device)
            
            # =================================
            # 1. Train Discriminator
            # =================================
            opt_disc.zero_grad()
            
            # Generate fake images (no grad for encoder here)
            with torch.no_grad():
                encoded_images_detached = encoder(images, watermarks)
            
            d_real_preds = discriminator(images)
            d_fake_preds = discriminator(encoded_images_detached)
            
            # [cite_start]Discriminator Loss [cite: 271-273]
            # Real images should be 1, Fake images should be 0
            loss_d_real = bce_loss(d_real_preds, torch.ones_like(d_real_preds))
            loss_d_fake = bce_loss(d_fake_preds, torch.zeros_like(d_fake_preds))
            loss_discriminator = (loss_d_real + loss_d_fake) * 0.5
            
            loss_discriminator.backward()
            opt_disc.step()
            
            # =================================
            # 2. Train Encoder & Decoder
            # =================================
            opt_enc_dec.zero_grad()
            
            # Encode
            encoded_images = encoder(images, watermarks)
            
            # [cite_start]Attack (Noise Layer) [cite: 195-198]
            # Simulate screen shooting distortions
            attacked_images = noise_layer(encoded_images)
            
            # Decode
            decoded_watermarks = decoder(attacked_images)
            
            # Discriminator Feedback (Adversarial Loss)
            d_fake_preds_for_enc = discriminator(encoded_images)
            
            # [cite_start]Calculate All Losses [cite: 277]
            l_l2 = mse_loss(encoded_images, images)
            l_lpips = lpips_fn(encoded_images, images).mean()
            l_ssim = 0 # (Optional: Add pytorch_msssim if strictly following paper)
            
            l_decoder = bce_loss(decoded_watermarks, watermarks) # Watermark recovery loss
            l_adv = bce_loss(d_fake_preds_for_enc, torch.ones_like(d_fake_preds_for_enc)) # Fool discriminator
            
            total_loss = (w['lambda_l2'] * l_l2 + 
                          w['lambda_lpips'] * l_lpips + 
                          w['lambda_ssim'] * l_ssim + 
                          w['lambda_dec'] * l_decoder + 
                          w['lambda_disc'] * l_adv)
                          
            total_loss.backward()
            opt_enc_dec.step()
            
            epoch_loss += total_loss.item()
            progress_bar.set_postfix({"Loss": f"{total_loss.item():.4f}", "DecAcc": f"{((torch.sigmoid(decoded_watermarks)>0.5)==watermarks).float().mean():.2f}"})

        # Save Checkpoint
        if (epoch + 1) % cfg['training']['save_interval'] == 0:
            save_path = f"ssrw_epoch_{epoch+1}.pt"
            print(f"Saving checkpoint to {save_path}")
            torch.save({
                'epoch': epoch,
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer': opt_enc_dec.state_dict(),
            }, save_path)

    # Save Final Model
    torch.save(encoder.state_dict(), "ssrw_encoder_final.pt")
    torch.save(decoder.state_dict(), "ssrw_decoder_final.pt")
    print("Training complete. Final models saved.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/Users/devmody/Documents/StellarLabs/iso-research-models/configs/ssrw_v1.yaml')
    args = parser.parse_args()
    train(args.config)
