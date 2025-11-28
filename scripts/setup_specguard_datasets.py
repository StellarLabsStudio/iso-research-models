import os
import shutil
import random
from pathlib import Path
import glob
from tqdm import tqdm

def setup_coco_train_val(source_dir, dest_root, train_count=10000, val_count=1000):
    """
    Splits COCO images into train and val sets for SpecGuard training.
    """

    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"Error: COCO source directory not found at {source_dir}")
        return

    # Create Directories
    train_dir = Path(dest_root) / 'coco' / 'train'
    val_dir = Path(dest_root) / 'coco' / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Get all images (COCO has .jpg images)
    images = list(source_path.glob('*.jpg'))
    if len(images) < (train_count + val_count):
        print(f"Error: Not enough images in source directory. Found {len(images)}, required {train_count + val_count}.")
        return

    # Shuffle and split
    print(f"Found {len(images)} images in COCO. Shuffling...")
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

    print("COCO train/val dataset setup complete for SpecGuard.")