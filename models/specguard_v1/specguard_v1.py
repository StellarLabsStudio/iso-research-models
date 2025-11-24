# models/specguard/specguard.py

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

