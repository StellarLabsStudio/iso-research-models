import os
import shutil
import random
from pathlib import Path
import glob
from tqdm import tqdm

def setup_mirflickr (source_dir, dest_root, train_count=9000, val_count=1000):
    
    """
    Splits MirFlickr25k into train and val sets.
    Paper: "9000" images as training sets and 1000 images as validation sets"
    """
    
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"Error: MirFlickr25k source directory not found at {source_dir}")
        return
    
    #Create Directories
    train_dir = Path(dest_root) / 'mirflickr' / 'train'
    val_dir = Path(dest_root) / 'mirflickr' / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    #Get all images (MirFlickr25k has .jpg images)
    images = list(source_path.glob('*.jpg'))
    if len(images) < (train_count + val_count):
        print(f"Error: Not enough images in source directory. Found {len(images)}, required {train_count + val_count}.")
        return
    
    # Shuffle and split
    print(f"Found {len(images)} images in MirFlickr25k. Shuffling...")
    random.seed(42) # Fixed seed for reproducibility
    random.shuffle(images)
    
    train_imgs = images[:train_count]
    val_imgs = images[train_count:train_count+val_count]
    
    print(f"Copying {len(train_imgs)} images to {train_dir}...")
    for img in tqdm(train_imgs):
        shutil.copy2(img, train_dir / img.name)

    print(f"Copying {len(val_imgs)} images to {val_dir}...")
    for img in tqdm(val_imgs):
        shutil.copy2(img, val_dir / img.name)
        
    print("MirFlickr25k dataset setup complete.")

def setup_coco (source_dir, dest_root, test_count=200):
    """
    Selects random images from COCO for the test set.
    Paper: "200 images were randomly chosen from the COCO dataset"
    """
    
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"Error: COCO source directory not found at {source_dir}")
        return

    test_dir = Path(dest_root) / "coco" / "test"
    os.makedirs(test_dir, exist_ok=True)

    images = list(source_path.glob("*.jpg"))
    if len(images) < test_count:
        print(f"Error: Not enough images in {source_dir}. Found {len(images)}")
        return

    print(f"Found {len(images)} images in COCO. Selecting {test_count} for testing...")
    random.seed(42)
    test_imgs = random.sample(images, test_count)

    print(f"Copying {len(test_imgs)} images to {test_dir}...")
    for img in tqdm(test_imgs):
        shutil.copy2(img, test_dir / img.name)

    print("COCO dataset setup complete.")
    
if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Update these paths to where you extracted the downloaded zips
    RAW_MIRFLICKR_PATH = "/Users/devmody/Documents/StellarLabs/iso-research-models/datasets/SSRW/raw/mirflickr" 
    RAW_COCO_PATH = "/Users/devmody/Documents/StellarLabs/iso-research-models/datasets/SSRW/raw/coco" 
    
    # Destination is your current repo's datasets folder
    DESTINATION_ROOT = "/Users/devmody/Documents/StellarLabs/iso-research-models/datasets/SSRW/split" 
    
    # --- EXECUTION ---
    print("--- Setting up MirFlickr (Train/Val) ---")
    setup_mirflickr(RAW_MIRFLICKR_PATH, DESTINATION_ROOT)
    
    print("\n--- Setting up COCO (Test) ---")
    setup_coco(RAW_COCO_PATH, DESTINATION_ROOT)
    
    print("\nDone! Dataset structure created.")


