# models/specguard/specguard.py

from sympy import zeta
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --------------------------------------------------------
# 1. Wavelet Projection (Haar DWT)
# Ref: Eq (4-6) - Decomposes image into sub-bands
# --------------------------------------------------------

class DWT (nn.Module):
    
    def __init__ (self):
        super().__init__()
        self.requires_grad_ = False # DWT is a fixed transform
        
    def forward (self, x):
        # x: (B, C, H, W)
        
        x01 = x[:, :, ::2, :] / 2 # even rows
        x02 = x[:, :, 1::2, :] / 2 # odd rows
        
        x1 = x01[:, :, :, ::2] # even rows, even cols
        x2 = x02[:, :, :, ::2] # odd rows, even cols
        x3 = x01[:, :, :, 1::2] # even rows, odd
        x4 = x02[:, :, :, 1::2] # odd rows, odd
        
        x_LL = x1 + x2 + x3 + x4
        x_LH = -x1 - x3 + x2 + x4
        x_HL = -x1 + x3 - x2 + x4
        x_HH = x1 - x3 - x2 + x4
        
        return x_LL, x_LH, x_HL, x_HH
    
class IWT (nn.Module):
    
    def __init__ (self):
        super().__init__()
        self.requires_grad_ = False # IWT is a fixed transform
        
    def forward(self, x_LL, x_LH, x_HL, x_HH):
        x_1 = (x_LL - x_LH - x_HL + x_HH) / 2
        x_2 = (x_LL + x_LH - x_HL - x_HH) / 2
        x_3 = (x_LL - x_LH + x_HL - x_HH) / 2
        x_4 = (x_LL + x_LH + x_HL + x_HH) / 2
        
        B, C, H, W = x_LL.shape
        y = torch.zeros((B, C, H * 2, W * 2), device=x_LL.device)
        
        y[:, :, 0::2, 0::2] = x_1
        y[:, :, 1::2, 0::2] = x_2
        y[:, :, 0::2, 1::2] = x_3
        y[:, :, 1::2, 1::2] = x_4
        
        return y

# --------------------------------------------------------
# 2. Spectral Projection (FFT Approximation)
# Ref: Eq (11-13) - Symmetrically extended FFT
# --------------------------------------------------------
class SpectralProjection(nn.Module):
    
    def forward (self, x):
        # x: (B, C, H, W) - usually the HH band
        
        #1. Symmetric Extension (Mirroring) to 2N x 2N
        #Ref: "Creating a symmetrically extended version... doubling size"
        x_flip_h = torch.flip(x, [3])
        x_flip_v = torch.flip(x, [2])
        x_flip_hv = torch.flip(x, [2, 3])
        
        top = torch.cat([x, x_flip_h], dim=3)
        bottom = torch.cat([x_flip_v, x_flip_hv], dim=3)
        x_extended = torch.cat([top, bottom], dim=2) # (B, C, 2H, 2W)
        
        # 2. FFT
        fft_coeffs = torch.fft.fft2(x_extended)
        
        # 3. Approximation (Real part of top-left quadrant)
        # Ref: Eq(13)
        
        H, W = x.shape[2], x.shape[3]
        zeta = fft_coeffs[:, :, :H, :W]
        
        # Return both for reconstruction validity, though embedding only uses zeta
        return zeta, fft_coeffs.imag
    
class InverseSpectralProjection(nn.Module):
    
    def forward(self, zeta, imag_part_orig=None):
        # Approximation of inverse: We assume embedding modification is symmetric
        # In strict theory, we would modify the full FFT coeff, but paper implies
        # operating on the projection zeta.
        
        # Simple reconstruction: Reverse the Real part extraction logic
        # Note: The paper is slightly vague on the exact inverse of the approximation 
        # so we use standard IFFT on the modified real component.
        
        # Re-mirror the modified zeta
        z_flip_h = torch.flip(zeta, [3])
        z_flip_v = torch.flip(zeta, [2])
        z_flip_hv = torch.flip(zeta, [2, 3])
        
        top = torch.cat([zeta, z_flip_h], dim=3)
        bottom = torch.cat([z_flip_v, z_flip_hv], dim=3)
        real_extended = torch.cat([top, bottom], dim=2) # (B, C, 2H, 2W)
        
        complex_extended = None
        # Use original imaginary part if available (for better fidelity), else 0
        if imag_part_orig is not None:
            # We must mirror the imaginary part too to maintain symmetry properties
            # or just use the stored full FFT if we had it. 
            # For simplicity/robustness as per "Approximation", we treat it as real signal.
            complex_extended = torch.complex(real_extended, torch.zeros_like(real_extended))
        else:
            complex_extended = torch.complex(real_extended, torch.zeros_like(real_extended))
            
        # Inverse FFT
        out_extended = torch.fft.ifft2(complex_extended).real
        
        # Crop back to N x N
        H, W = zeta.shape[2], zeta.shape[3]
        return out_extended[:, :, :H, :W]
    
# --------------------------------------------------------
# 3. SpecGuard Encoder
# --------------------------------------------------------

class SpecGuardEncoder (nn.Module):
    
    def __init__ (self, config):
        super().__init__()
        self.dwt = DWT()
        self.iwt = IWT()
        self.sp = SpectralProjection()
        self.isp = InverseSpectralProjection()
        
        # Convolutional Layers for Processing Spectral Domain
        # Ref: "variable k of convolutional layers"
        k = config['model']['conv_layers']
        kernel_size = config['model']['kernel_size'] # e.g. 32
        self.conv_stack = nn.Sequential()
        
        # Input to Conv is (B, C, H, W) - Processing Spectral Coeffs as Image
        for i in range (k):
            self.conv_stack.add_module (f"conv_{i}", nn.Conv2d(3, 3, kernel_size, padding=kernel_size // 2))
            self.conv_stack.add_module(f"act_{i}", nn.LeakyReLU(0.2))
        
        self.strength = config['model']['strength_factor']
        self.radius = config['model']['radius']
        
        # Final Smoothening Layer after Embedding
        self.final_conv = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(0.2)
        )
        
    def forward (self, x, watermark):
        
        # x: Cover Image (B, 3, 128, 128)
        # watermark: (B, L) binary message
        
        # 1. Wavelet Projection
        LL, LH, HL, HH = self.dwt(x)
        
        # 2. Spectral Projection on HH band
        zeta, _ = self.sp(HH)
        
        # 3. Process Spectral Features
        zeta_feat = self.conv_stack(zeta)
        
        # 4. Create Radial Mask
        # Ref: "radial mask... if distance <= r, mask=1"
        B, C, H, W = zeta.shape
        cx, cy = W // 2, H // 2
        
        y_grid, x_grid = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        dist = torch.sqrt((x_grid - cx)**2 + (y_grid - cy)**2).to(x.device)
        mask = (dist <= self.radius).float().unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
        
        # 5. Prepare Watermark
        # Expand Watermark to match Image Dimensions (B, C, H, W)
        # Paper says "reshaped and expanded across channels"
        # Simple Tiling Strategy
        wm_expanded = watermark.unsqueeze(-1).unsqueeze(-1).expand(B, C, H, W)
        # Ideally, L should match patch count, but here we broadcast for robustness
        
        # 6. Embed Watermark
        # Ref: Eq (15) zeta_new = zeta_feat + M * s
        zeta_embedded = zeta_feat + (wm_expanded * mask * self.strength)
        
        # 7. Final Smoothing
        zeta_refined = self.final_conv(zeta_embedded)
        
        # 8. Reconstruct
        # Eq (16) ISP
        HH_embedded = self.isp(zeta_refined)
        
        # Eq (17) IWP
        return self.iwt(LL, LH, HL, HH_embedded)
    
# --------------------------------------------------------
# 4. SpecGuard Decoder
# --------------------------------------------------------
class SpecGuardDecoder (nn.Module):
    
    def __init__ (self, config):
        super().__init__()
        self.dwt = DWT()
        self.sp = SpectralProjection()
        
        # Learnable Threshold Theta
        # Ref: "Learning threshold theta to decode each bit"
        # Ref: "theta <- theta - eta * grad"
        
        self.theta = nn.Parameter(torch.tensor(config['model'['initial_theta']]))
        
        # Extraction Network
        k = config['model']['conv_layers']
        kernel_size = config['model']['kernel_size']
        self.conv_stack = nn.Sequential()
        for i in range(k):
            self.conv_stack.add_module(f"conv_{i}", nn.Conv2d(3, 3, kernel_size, padding=kernel_size//2))
            self.conv_stack.add_module(f"act_{i}", nn.LeakyReLU(0.2))
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) # Global Pooling to get bits
        self.fc = nn.Linear(3, config['data']['watermark_len'])
        
    def forward (self, x):
        
        # 1. Wavelet Projection
        _, _, _, HH = self.dwt(x)
        
        # 2. Spectral Projection
        zeta, _ = self.sp(HH)
        
        # 3. Feature Extraction
        zeta_feat = self.conv_stack(zeta)
        
        # 4. Extraction Logic
        # "masked values compared against learnable threshold"
        # We simplify the spatial mask extraction to a global pooling for the dense layer
        # The paper describes extracting PER COORDINATE but for a vector message M,
        # We usually map features -> M
        
        # Global Average Pooling of Spectral Features
        feat_vector = self.pool(zeta_feat).flatten(1) # (B, 3)
        
        # Map to Message Size
        logits = self.fc(feat_vector) # (B, L)
        
        # Apply Learnable Threshold Logic for Loss Calculation
        # We return (logits - theta) so that
        # if logits > theta -> output > 0 -> Sigmoid > 0.5 -> Bit 1
        # if logits < theta -> output < 0 -> Sigmoid < 0.5 -> Bit 0
        
        return logits - self.theta        
        
# --------------------------------------------------------
# 5. Discriminator (Standard PatchGAN or Simple)
# --------------------------------------------------------
class Discriminator (nn.Module):
    
    def __init__ (self, config):
        super().__init__()
        self.net = nn.Sequential(
           nn.Conv2d(3, 64, 3, 2, 1),
           nn.LeakyReLU(0.2),
           nn.Conv2d(64, 128, 3, 2, 1),
           nn.BatchNorm2d(128),
           nn.LeakyReLU(0.2),
           nn.Conv2d(128, 256, 3, 2, 1),
           nn.BatchNorm2d(256),
           nn.LeakyReLU(0.2), 
           nn.AdaptiveAvgPool2d((1, 1)),
           nn.Flatten(),
           nn.Linear(256, 1),
        )
    
    def forward (self, x):
        return self.net(x)