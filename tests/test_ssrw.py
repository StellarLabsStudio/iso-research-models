import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import glob
from PIL import Image
from tqdm import tqdm
import kornia.metrics as KM

# Import Architecture
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.ssrw import CrossAttentionEncoder, EnhancedUNetDecoder, NoiseLayer

class SimpleImageDataset(Dataset):
    """Simple dataset loader for flat directories"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(root_dir, '*.jpg')) + glob.glob(os.path.join(root_dir, '*.png'))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image
        except Exception:
            return torch.zeros(3, 128, 128)

def compute_accuracy(preds, target):
    """Compute Bit Accuracy (1 - Bit Error Rate)"""
    preds_binary = (torch.sigmoid(preds) > 0.5).float()
    correct = (preds_binary == target).sum().item()
    total = target.numel()
    return correct / total

def test(config_path, weights_dir, test_data_path):
    # Load Config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device(cfg['device'])
    print(f"Running evaluation on: {device}")

    # --- Initialize Models ---
    encoder = CrossAttentionEncoder(
        img_size=cfg['data']['image_size'],
        patch_size=cfg['model']['patch_size'],
        watermark_len=cfg['data']['watermark_len']
    ).to(device)
    
    decoder = EnhancedUNetDecoder(watermark_len=cfg['data']['watermark_len']).to(device)
    
    # Load Weights
    enc_path = os.path.join(weights_dir, "ssrw_encoder_final.pt")
    dec_path = os.path.join(weights_dir, "ssrw_decoder_final.pt")
    
    # Handle case where final weights don't exist yet (use latest epoch)
    if not os.path.exists(enc_path):
        checkpoints = glob.glob(os.path.join(weights_dir, "ssrw_epoch_*.pt"))
        if checkpoints:
            latest_ckpt = max(checkpoints, key=os.path.getctime)
            print(f"Final weights not found. Loading latest checkpoint: {latest_ckpt}")
            checkpoint = torch.load(latest_ckpt, map_location=device)
            encoder.load_state_dict(checkpoint['encoder'])
            decoder.load_state_dict(checkpoint['decoder'])
        else:
            print("Error: No weights found. Run training first!")
            return
    else:
        print(f"Loading final weights from {weights_dir}")
        encoder.load_state_dict(torch.load(enc_path, map_location=device))
        decoder.load_state_dict(torch.load(dec_path, map_location=device))

    encoder.eval()
    decoder.eval()
    
    # Initialize Noise Layer for robustness testing
    # Note: You can disable specific attacks in config to test "clean" robustness
    noise_layer = NoiseLayer(cfg).to(device)

    # --- Data Loading ---
    transform = transforms.Compose([
        transforms.Resize((cfg['data']['image_size'], cfg['data']['image_size'])),
        transforms.ToTensor()
    ])
    
    if test_data_path is None:
        # Default to the COCO test set created by setup script
        test_data_path = "/Users/devmody/Documents/StellarLabs/iso-research-models/datasets/SSRW/split/coco/test" # Adjust if needed
        
    print(f"Loading test data from: {test_data_path}")
    test_dataset = SimpleImageDataset(test_data_path, transform=transform)
    dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # --- Metrics ---
    total_psnr = 0.0
    total_ssim = 0.0
    total_acc_clean = 0.0
    total_acc_attacked = 0.0
    batches = 0

    print("Starting evaluation loop...")
    with torch.no_grad():
        for images in tqdm(dataloader):
            images = images.to(device)
            B = images.size(0)
            
            # Generate random watermarks
            watermarks = torch.randint(0, 2, (B, cfg['data']['watermark_len'])).float().to(device)
            
            # 1. Encode
            encoded_images = encoder(images, watermarks)
            
            # 2. Metrics: Visual Quality (Invisibility)
            # Clamp to 0-1 for valid metric calculation
            encoded_clamped = torch.clamp(encoded_images, 0, 1)
            
            # PSNR
            psnr = KM.psnr(encoded_clamped, images, max_val=1.0)
            total_psnr += psnr.mean().item()
            
            # SSIM
            ssim = KM.ssim(encoded_clamped, images, window_size=11)
            total_ssim += ssim.mean().item()
            
            # 3. Metrics: Robustness (Decoder Accuracy)
            
            # Case A: Clean (No Attack)
            decoded_clean = decoder(encoded_images)
            total_acc_clean += compute_accuracy(decoded_clean, watermarks)
            
            # Case B: Attacked (Screen-Shooting Simulation)
            attacked_images = noise_layer(encoded_images)
            decoded_attacked = decoder(attacked_images)
            total_acc_attacked += compute_accuracy(decoded_attacked, watermarks)
            
            batches += 1

    # --- Results ---
    avg_psnr = total_psnr / batches
    avg_ssim = total_ssim / batches
    avg_acc_clean = total_acc_clean / batches
    avg_acc_attacked = total_acc_attacked / batches

    print("\n" + "="*30)
    print("       EVALUATION REPORT       ")
    print("="*30)
    print(f"Visual Quality (Invisibility):")
    print(f"  PSNR: {avg_psnr:.2f} dB  (Target: >40 dB)")
    print(f"  SSIM: {avg_ssim:.4f}     (Target: >0.98)")
    print("-" * 30)
    print(f"Robustness (Bit Accuracy):")
    print(f"  Clean Accuracy:    {avg_acc_clean*100:.2f}%")
    print(f"  Attacked Accuracy: {avg_acc_attacked*100:.2f}%")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/Users/devmody/Documents/StellarLabs/iso-research-models/configs/ssrw_v1.yaml')
    parser.add_argument('--weights_dir', type=str, default='/Users/devmody/Documents/StellarLabs/iso-research-models/training') # Current dir by default
    parser.add_argument('--data_dir', type=str, default=None)   # Overwrite test data path
    args = parser.parse_args()
    
    test(args.config, args.weights_dir, args.data_dir)