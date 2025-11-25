# training/specguard_v1/train_specguard_v1.py

import os
import sys
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import lpips

# Import Model
from models.specguard_v1.specguard_v1 import SpecGuardEncoder, SpecGuardDecoder, Discriminator
# Reuse dataset from existing code or define simple one
from training.specguard_v1.train_specguard_v1 import SimpleImageDataset

def train(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device(cfg['device'])
    # --- Initialize ---
    encoder = SpecGuardEncoder().to(device)
    decoder = SpecGuardDecoder().to(device)
    # Discriminator is optional in the SpecGuard paper but implied by "GAN-based" comparisons
    # discriminator = Discriminator().to(device)
    # We will include it for perceptual quality if needed, but paper focusses on MSE/LPIPS
    
    # Losses
    mse_loss = nn.MSELoss()
    
    # Optimizers
    # "decoder learning rate set to 1e-3... encoder learnng rate set to 1e-2"
    opt_enc = optim.Adam(encoder.parameters(), lr=cfg['training']['learning_rate_enc'])
    opt_dec = optim.Adam(decoder.parameters(), lr=cfg['training']['learning_rate_dec'])    
    
    # Scheduler
    # 'decoder ... reduced by half every 100 steps'
    # Assuming 'steps' here means epochs given the total is 300
    sched_dec = optim.lr_scheduler.StepLR(opt_dec, step_size=100, gamma=0.5)
    
    # Data
    transform = transforms.Compose([
        transforms.Resize((cfg['data']['image_size'], cfg['data']['image_szie'])),
        transforms.ToTensor()
    ])
    dataset = SimpleImageDataset(cfg['data']['train_dir'], transform=transform)
    loader = DataLoader(dataset, batch_size=cfg['data']['batch_size'], shuffle=True)
    
    # Weights
    lambda_enc = cfg['loss_weights']['lambda_enc']
    lambda_dec = cfg['loss_weights']['lambda_dec']
    
    print("Starting SpecGuard v1 Training...")
    
    for epoch in range(cfg['training']['epochs']):
        encoder.train()
        decoder.train()
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1})")
        for images in pbar:
            images = images.to(device)
            B = images.shape[0]
            
            # Generate Random Binary Message
            watermarks = torch.randint(0, 2, (B, cfg['data']['watermark_len'])).float().to(device)
            
            # --- Forward Pass ---
            # 1. Embed
            watermarked_images = encoder(images, watermarks)
            
            # 2. Decode (No Attack Layer in Base Training, add NoiseLayer if needed for robustness)
            # The paper mentions "Robustness against various transformations"
            # This implies a noise layer should be here like SSRW
            # For basic implementation, we train clean first
            
            decoded_digits = decoder(watermarked_images)
            
            # --- Loss Calculation ---
            # Eq(18): L_enc = || I_embedded - I||^2
            l_enc = mse_loss(watermarked_images, images)
            
            # Eq(19): L_dec = || D(I_embedded) - M ||^2
            # Note: Using logits with MSE is possible if M is 0/1
            # or use BCEWithLogitsLoss for better stability. Paper says MSE
            # We pass probabilities through Sigmoid for MSE to make sense vs 0/1 targets
            decoded_probs = torch.sigmoid(decoded_digits)
            l_dec = mse_loss(decoded_probs, watermarks)
            
            # Eq(20): Total Loss
            total_loss = lambda_enc * l_enc + lambda_dec * l_dec
            
            # --- Optimization ---
            opt_enc.zero_grad()
            opt_dec.zero_grad()
            total_loss.backward()
            opt_enc.step()
            opt_dec.step()
            
            # Calculate Accuracy
            preds = (decoded_probs > 0.5).float()
            acc = (preds == watermarks).float().mean().item()
            
            pbar.set_postfix({"Loss" : f"{total_loss.item():.4f}", "Acc" : f"{acc:.2f}"})
            
        # Scheduler Step
        sched_dec.step()
        
        # Save
        if (epoch + 1) % cfg['training']['save_interval'] == 0:
            save_dir = f"training/specguard_v1/weights/"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(encoder.state_dict(), os.path.join(save_dir, f"encoder_epoch{epoch+1}.pth"))
            torch.save(decoder.state_dict(), os.path.join(save_dir, f"decoder_epoch{epoch+1}.pth"))
            print(f"Saved models at epoch {epoch+1}")
    
    print("Training Complete.")
    
if __name__ == "__main__":
    train("configs/specguard_v1.yaml")
            