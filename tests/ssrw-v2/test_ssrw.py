# test_ssrw_v2.py
import os, yaml, glob, torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import kornia.metrics as KM
from tqdm import tqdm

from ssrw_v2 import CrossAttentionEncoder, EnhancedUNetDecoder, NoiseLayer

class SimpleImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.paths = glob.glob(os.path.join(root,"*.jpg")) + \
                     glob.glob(os.path.join(root,"*.png"))
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self,i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

def compute_acc(pred, target):
    return ((torch.sigmoid(pred)>0.5)==target).float().mean().item()

def test(config_path, weights_dir, data_dir):
    with open(config_path,'r') as f:
        cfg=yaml.safe_load(f)

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
    noise = NoiseLayer(cfg).to(device)

    enc_w = os.path.join(weights_dir,"encoder_v2_final.pt")
    dec_w = os.path.join(weights_dir,"decoder_v2_final.pt")
    encoder.load_state_dict(torch.load(enc_w, map_location=device))
    decoder.load_state_dict(torch.load(dec_w, map_location=device))
    encoder.eval(); decoder.eval()

    trans = transforms.Compose([
        transforms.Resize((cfg['data']['image_size'], cfg['data']['image_size'])),
        transforms.ToTensor()
    ])
    dataset = SimpleImageDataset(data_dir,transform=trans)
    loader = DataLoader(dataset,batch_size=32)

    psnr_total=0; ssim_total=0
    acc_clean=0; acc_attacked=0
    batches=0

    for images in tqdm(loader):
        images=images.to(device)
        B=images.size(0)
        wm=torch.randint(0,2,(B,cfg['data']['watermark_len'])).float().to(device)

        enc = encoder(images,wm)
        psnr_total += KM.psnr(enc, images, max_val=1.0).mean().item()
        ssim_total += KM.ssim(enc, images, window_size=11).mean().item()

        dec_clean = decoder(enc)
        acc_clean += compute_acc(dec_clean,wm)

        attacked = noise(enc)
        dec_attacked = decoder(attacked)
        acc_attacked += compute_acc(dec_attacked,wm)

        batches+=1

    print("\n========== EVAL REPORT ==========")
    print(f"PSNR: {psnr_total/batches:.2f} dB")
    print(f"SSIM: {ssim_total/batches:.4f}")
    print(f"Clean Accuracy: {100*(acc_clean/batches):.2f}%")
    print(f"Attacked Accuracy: {100*(acc_attacked/batches):.2f}%")
    print("=================================")

if __name__=="__main__":
    test("ssrw_v2.yaml","./","./test_images")
