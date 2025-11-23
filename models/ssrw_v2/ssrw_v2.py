import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.filters as KF
import kornia.geometry.transform as GT
import math
import random

# --------------------------------------------------------
# Basic Conv Block
# --------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel, 1, padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): 
        return self.net(x)
    

# --------------------------------------------------------
# Cross-Attention Encoder (V2)
# --------------------------------------------------------

class CrossAttentionEncoder (nn.Module):
    
    def __init__ (self, img_size=128, patch_size=8, watermark_len=64, embed_dim=64, num_heads=8, hidden_dim=256):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.patch_input_dim = 3 * patch_size * patch_size
        
        # Watermark Tokens
        self.watermark_fc = nn.Linear(watermark_len, self.num_patches)
        self.watermark_proj = nn.Linear(1, embed_dim)
        
        # Image patch → embedding
        self.image_patch_embed = nn.Linear(self.patch_input_dim, embed_dim)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        # Cross-attention
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim)
        )

        # Patch reconstruction
        self.to_patch_features = nn.Linear(embed_dim, self.patch_input_dim)

        # Refinement CNN
        self.refine_head = nn.Sequential(
            ConvBlock(3, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(128, 64),
            nn.Conv2d(64, 3, kernel_size=1)
        )

        # Residual blend
        self.alpha = nn.Parameter(torch.tensor(0.7))
    
    def forward (self, x, watermark):
        B = x.size(0)
        device = x.device

        # Watermark → patch tokens
        w = self.watermark_fc(watermark).unsqueeze(-1)  # B,P,1
        w_emb = self.watermark_proj(w) + self.pos_embed.to(device)

        # Patchify image
        ps = self.patch_size
        x_p = x.unfold(2, ps, ps).unfold(3, ps, ps)  # B,C,nh,nw,ps,ps
        n_h, n_w = x_p.size(2), x_p.size(3)

        x_p = x_p.permute(0,2,3,1,4,5).contiguous().view(B, self.num_patches, -1)
        x_emb = self.image_patch_embed(x_p) + self.pos_embed.to(device)

        # Cross attention
        attn_out, _ = self.mha(x_emb, w_emb, w_emb)
        x_after = x_emb + attn_out
        x_after = x_after + self.mlp(x_after)

        # Reconstruct patches → image
        patches = self.to_patch_features(x_after)
        patches = patches.view(B, n_h, n_w, 3, ps, ps)
        out_img = patches.permute(0,3,1,4,2,5).contiguous().view(B,3,self.img_size,self.img_size)

        # Refinement
        refined = self.refine_head(out_img)

        # Residual output
        return torch.clamp(self.alpha * refined + (1 - self.alpha) * x, 0, 1)
    

# --------------------------------------------------------
# Noise Layer (V2)
# --------------------------------------------------------

class NoiseLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n = cfg.get('noise_layer', {})
        self.p_pers = n.get('prob_perspective', 0.9)
        self.p_light = n.get('prob_light', 0.6)
        self.p_blur = n.get('prob_blur', 0.5)
        self.p_moire = n.get('prob_moire', 0.8)

    def forward(self, img):
        B,C,H,W = img.size()
        out = img

        # Perspective
        if random.random() < self.p_pers:
            warped = []
            for i in range(B):
                src = torch.tensor([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], device=img.device).float()
                jitter = max(2, int(0.02 * max(H,W)))
                dst = src + (torch.rand_like(src)-0.5)*2*jitter
                H_mat = GT.find_homography_dlt(src.unsqueeze(0), dst.unsqueeze(0))[0]
                warped.append(GT.warp_perspective(img[i:i+1], H_mat.unsqueeze(0), (H,W)))
            out = torch.cat(warped, dim=0)

        # Lighting gradient + point
        if random.random() < self.p_light:
            masks = []
            for i in range(B):
                xs = torch.linspace(0,1,W, device=img.device)[None,None,:].expand(1,1,W)
                ys = torch.linspace(0,1,H, device=img.device)[None,:,None].expand(1,H,1)
                a = random.uniform(0.05,0.2)
                b = random.uniform(0,0.5)
                grad = a + b*(xs+ys)

                x0 = random.uniform(0,W)
                y0 = random.uniform(0,H)
                xv = torch.arange(W, device=img.device)[None,None,:].expand(1,H,W)
                yv = torch.arange(H, device=img.device)[None,:,None].expand(1,H,W)
                d2 = (xv-x0)**2 + (yv-y0)**2
                point = random.uniform(0.2,1.0)/(d2+1e-6)

                masks.append((grad + 0.01*point).clamp(0.6,1.4))
            masks = torch.stack(masks,0)
            out = out*masks

        # Blur
        if random.random() < self.p_blur:
            sigma = random.uniform(0.8,2.5)
            out = KF.gaussian_blur2d(out, (3,3), (sigma,sigma))
            try:
                k = random.choice([3,5,7])
                ang = random.uniform(0,360)
                out = KF.motion_blur(out,k,ang)
            except:
                pass

        # Moire
        if random.random() < self.p_moire:
            moires=[]
            for i in range(B):
                freq = random.uniform(0.5,4.0)
                amp = random.uniform(0.003,0.03)
                xs = torch.linspace(0,2*math.pi*freq,W,device=img.device)[None,None,:].expand(1,H,W)
                ys = torch.linspace(0,2*math.pi*freq,H,device=img.device)[None,:,None].expand(1,H,W)
                pattern = torch.sin(xs+ys).unsqueeze(0)
                moires.append(1 + amp*pattern)
            moires = torch.cat(moires,0)
            out = out * moires.clamp(0.9,1.1)

        return out
    
# --------------------------------------------------------
# Decoder (U-Net Lite)
# --------------------------------------------------------

class EnhancedUNetDecoder(nn.Module):
    def __init__(self, watermark_len=64):
        super().__init__()
        self.down1 = ConvBlock(3,64)
        self.pool = nn.MaxPool2d(2)
        self.down2 = ConvBlock(64,128)
        self.mid = ConvBlock(128,128)
        self.up = nn.ConvTranspose2d(128,64,2,2)
        self.up_conv = ConvBlock(128,64)
        self.out = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, watermark_len)
        )

    def forward(self,x):
        d1=self.down1(x)
        p=self.pool(d1)
        d2=self.down2(p)
        m=self.mid(d2)
        u=self.up(m)
        cat=torch.cat([u,d1],1)
        return self.out(self.up_conv(cat))
    
# --------------------------------------------------------
# Discriminator
# --------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,32,4,2,1), nn.LeakyReLU(0.2),
            nn.Conv2d(32,64,4,2,1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64,128,4,2,1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128,1)
        )
    def forward(self,x):
        return self.net(x).squeeze(-1)