import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.augmentation as K
import kornia.filters as KF
import math

class CrossAttentionEncoder (nn.Module):
    
    """
    Encoder using Multi-Head Cross-Attention.
    Ref: Section 'Encoder' of the SSRW paper. [cite: 173]
    """
    
    def __init__ (self, img_size=128, patch_size=8, watermark_len=64, embed_dim=64, num_heads=16):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        #Watermark embedding
        self.watermark_fc = nn.Linear(watermark_len, self.num_patches)
        self.watermark_proj = nn.Linear(1, embed_dim)
        
        #Image patch embedding (3 channels * 8 * 8 = 192)
        patch_input_dim = 3 * patch_size * patch_size
        self.image_patch_embed = nn.Linear(patch_input_dim, embed_dim)
        
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
        #Reconstruction
        self.final_fc = nn.Linear(2 * embed_dim, patch_input_dim)
        self.pixel_shuffle = nn.PixelShuffle(patch_size)
        
    def forward (self, x, watermark):
        B = x.shape[0]
        #1. Process Watermark [cite: 180]
        w = self.watermark_fc(watermark).unsqueeze(-1)  # (B, 256, 1)
        w_emb = self.watermark_proj(w) + self.pos_embed  
        
        #2. Process Image Patches
        x_patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x_patches = x_patches.permute(0,2,3,1,4,5).contiguous().view(B, self.num_patches, -1)
        x_emb = self.image_patch_embed(x_patches) + self.pos_embed
        
        #3. Cross-Attention (Query=Image, Key/Value=Watermark) [cite:182]
        attn_out, _ = self.mha(query=x_emb, key=w_emb, value=w_emb)
        
        #4. Concatenate and Reconstruct Patches [cite:184]
        combined = torch.cat([x_emb + attn_out, w_emb], dim=-1)
        out_patches = self.final_fc(combined)
        
        #Reshape back to Image
        out_patches = out_patches.view(B, self.img_size // self.patch_size, self.img_size // self.patch_size, 3, self.patch_size, self.patch_size)
        out_img = out_patches.permute(0,3,1,4,2,5).contiguous().view(B, 3, self.img_size, self.img_size)
        
        # Paper implies blending loss constrains this to look like the cover.
        # We can add a residual connection to the original image for stability if needed, 
        # but the paper describes direct generation.
        return out_img
    
class NoiseLayer (nn.Module):
    
    """
    Simulates screen-shooting distortions.
    Ref: Section 'Noise Layer' [cite: 194]
    """
    
    def __init__ (self, config):
        super().__init__()
        self.probs = config['noise_layer']
        
        # Perspective Distortion [cite: 201]
        self.perspective = K.RandomPerspective(distortion_scale=0.2, p=self.probs['prob_perspective'])
        
        # Blur (Gaussian + Motion Approximation) [cite: 234]
        self.blur = K.RandomGaussianBlur((3,3), (0.1, 2.0), p=self.probs['prob_blur'])
        
        # Lighting (approximated with ColorJitter for brightness/contrast) [cite: 202]
        self.lighting = K.ColorJitter(brightness=0.2, contrast=0.2, p=self.probs['prob_light'])
        
        
    def forward (self, x):
        # Apply distortions sequentially
        out = self.perspective(x)
        out = self.blur(out)
        out = self.lighting(out)
        
        # Note: True Moiré simulation [cite: 237] requires complex frequency interference 
        # which is often custom-coded. For 'iso-research-models', this is a functional baseline.
        return out
    
class EnhancedUNetDecoder (nn.Module):
    
    """
    Decoder using Enhanced U-Net.
    Ref: Section 'Decoder' of the paper [cite: 243]
    """
    
    def __init__ (self, watermark_len=64):
        super().__init__()
        
        #Encoder Path (Strided Convs) [cite:245]
        self.enc1 = self._block(3, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.bottleneck = self._block(256, 512, stride=1)
        
        #Decoder Path (Upsample + Concat)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dec3 = self._block(512 + 256, 256, stride=1)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dec2 = self._block(256 + 128, 128, stride=1)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dec1 = self._block(128 + 64, 64, stride=1)
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, watermark_len)
        )
        
    def _block (self, in_c, out_c, stride=2):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    
    def forward (self, x):
        #Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
                
        #Decoder    
        d3 = self.dec3(torch.cat([b, e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.head(d1) # Logits

class Discriminator (nn.Module):
    
    """
    Discriminator for Adversarial Training.
    Ref: Section 'Discriminator' of the paper [cite: 269]
    """
    
    def __init__ (self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 1)
        )
        
    def forward (self, x):
        return self.net(x)