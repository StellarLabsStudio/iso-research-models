import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import lpips

# Import Architecture from Local Models Directory
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.ssrw import CrossAttentionEncoder, EnhancedUNetDecoder, Discriminator, NoiseLayer

def train (config_path):
    # Load Configuration
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device(cfg['device'])
    
    # Initialize Models
    encoder = CrossAttentionEncoder(
        img_size=cfg['data']['image_size']
        patch_size=cfg['model']['patch_size'],
        watermark_len=cfg['data']['watermark_len'],
    ).to(device)
    
    decoder = EnhancedUNetDecoder(watermark_len=cfg['data']['watermark_len']).to(device)
    discriminator = Discriminator().to(device)
    noise_layer = NoiseLayer(cfg).to(device)
    
    # Loss Functions [cite: 188-192]
    lpips_fn = lpips.LPIPS (net='alex').to(device)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss() # For watermark and discriminator
    
    # Optimizers
    opt_enc_dec = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=cfg['training']['learning_rate'])
    opt_disc = optim.Adam(discriminator.parameters(), lr=cfg['training']['learning_rate'])
    
    # Weights
    w = cfg['loss_weights']
    
    # Dummy Data Loader (Replace with real dataset)
    transform = transforms.Compose([
        transforms.Resize((cfg['data']['image_size'], cfg['data']['image_size'])),
        transforms.ToTensor()
    ])
    # Assuming standard Folder dataset structure
    # dataset = datasets.ImageFolder(cfg['data']['train_dir'], transform=transform)
    # dataloader = DataLoader(dataset, batch_size=cfg['data']['batch_size'], shuffle=True)
    
    print("Starting training loop...")
