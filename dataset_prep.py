"""
Face Mask Detection - Dataset Preparation Guide & Helper
==========================================================
Recommended dataset: Kaggle "Face Mask Detection" by Prajna Bhandary
  → https://www.kaggle.com/datasets/omkargurav/face-mask-dataset
  → ~7553 images: 3725 with_mask, 3828 without_mask

Steps:
  1. Download and extract dataset into dataset/
  2. Run: python dataset_prep.py --verify
  3. Run: python dataset_prep.py --augment   (optional, if class imbalance)
"""

import os
import argparse
import shutil
import random
import cv2
import numpy as np
from pathlib import Path

DATASET_DIR = Path("dataset")
CLASSES     = ["with_mask", "without_mask"]
IMG_SIZE    = (224, 224)


# ─── Verify dataset structure ─────────────────────────────────────────────────
def verify_dataset():
    print("\n📁 Dataset Verification")
    print("=" * 40)
    all_ok = True
    for cls in CLASSES:
        cls_path = DATASET_DIR / cls
        if not cls_path.exists():
            print(f"  ❌  Missing folder: {cls_path}")
            all_ok = False
            continue
        images = list(cls_path.glob("*.jpg")) + \
                 list(cls_path.glob("*.jpeg")) + \
                 list(cls_path.glob("*.png"))
        print(f"  ✅  {cls:20s} → {len(images):>5} images")

    if all_ok:
        print("\n✅  Dataset structure looks good!")
        print("\n💡 Recommended minimum: 1000 images per class")
        print("   Ideal split: 80% train / 20% test (handled automatically in train_model.py)")
    else:
        print("\n❌  Please fix the missing folders and re-run.")
        _print_setup_instructions()


def _print_setup_instructions():
    print("""
─── Dataset Setup Instructions ──────────────────────────────────────
Option A: Kaggle Dataset (Recommended)
  1. Visit: https://www.kaggle.com/datasets/omkargurav/face-mask-dataset
  2. Download and unzip
  3. Place images into:
       dataset/with_mask/     ← images of faces WITH mask
       dataset/without_mask/  ← images of faces WITHOUT mask

Option B: Custom Dataset
  • Collect ~500-1000 images per class
  • Ensure faces are clearly visible and varied
  • Recommended sources: internet scraping, phone camera, open datasets

Option C: Auto-download via Kaggle API
  pip install kaggle
  kaggle datasets download -d omkargurav/face-mask-dataset
  unzip face-mask-dataset.zip -d dataset_raw/
  # Then sort images into dataset/with_mask/ and dataset/without_mask/
─────────────────────────────────────────────────────────────────────
""")


# ─── Data augmentation for class balancing ────────────────────────────────────
def augment_class(cls_name, target_count=3000):
    cls_path = DATASET_DIR / cls_name
    images   = list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.jpeg")) + \
                list(cls_path.glob("*.png"))
    current  = len(images)
    if current >= target_count:
        print(f"  {cls_name}: {current} images already ≥ target {target_count}, skipping.")
        return

    needed = target_count - current
    print(f"  {cls_name}: {current} images → generating {needed} augmented images...")

    aug_dir = cls_path / "augmented"
    aug_dir.mkdir(exist_ok=True)

    for i in range(needed):
        src = random.choice(images)
        img = cv2.imread(str(src))
        if img is None:
            continue

        # Random augmentations
        if random.random() > 0.5:
            img = cv2.flip(img, 1)                          # horizontal flip
        angle = random.uniform(-20, 20)                     # rotation
        M     = cv2.getRotationMatrix2D(
                    (img.shape[1]//2, img.shape[0]//2), angle, 1.0)
        img   = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        if random.random() > 0.5:
            factor = random.uniform(0.7, 1.3)               # brightness
            img    = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        out_path = aug_dir / f"aug_{i:05d}.jpg"
        cv2.imwrite(str(out_path), img)

    print(f"  ✅  Done. Augmented images saved to {aug_dir}")


def augment_dataset(target=3000):
    print("\n🔄 Augmenting dataset for class balance...")
    for cls in CLASSES:
        augment_class(cls, target)


# ─── Check image integrity ────────────────────────────────────────────────────
def check_integrity():
    print("\n🔍 Checking image integrity...")
    corrupt = []
    for cls in CLASSES:
        cls_path = DATASET_DIR / cls
        for img_path in cls_path.rglob("*.*"):
            img = cv2.imread(str(img_path))
            if img is None:
                corrupt.append(str(img_path))

    if corrupt:
        print(f"\n⚠️  Found {len(corrupt)} corrupt/unreadable images:")
        for p in corrupt[:10]:
            print(f"    {p}")
        if len(corrupt) > 10:
            print(f"    ... and {len(corrupt)-10} more")
        resp = input("\nDelete corrupt images? [y/N] ")
        if resp.lower() == "y":
            for p in corrupt:
                os.remove(p)
            print(f"  Deleted {len(corrupt)} files.")
    else:
        print("  ✅  All images readable!")


# ─── Print dataset summary ────────────────────────────────────────────────────
def summary():
    print("\n📊 Dataset Summary")
    print("=" * 40)
    total = 0
    for cls in CLASSES:
        cls_path = DATASET_DIR / cls
        imgs     = list(cls_path.rglob("*.jpg")) + \
                   list(cls_path.rglob("*.jpeg")) + \
                   list(cls_path.rglob("*.png"))
        total   += len(imgs)
        print(f"  {cls:20s} → {len(imgs):>5} images")
    print(f"  {'TOTAL':20s} → {total:>5} images")
    print(f"\n  Train (80%): ~{int(total*0.8)}")
    print(f"  Test  (20%): ~{int(total*0.2)}")


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Dataset Preparation Helper")
    ap.add_argument("--verify",    action="store_true", help="Verify dataset structure")
    ap.add_argument("--augment",   action="store_true", help="Augment to balance classes")
    ap.add_argument("--integrity", action="store_true", help="Check for corrupt images")
    ap.add_argument("--summary",   action="store_true", help="Print dataset summary")
    ap.add_argument("--target",    type=int, default=3000,
                    help="Target images per class for augmentation (default: 3000)")
    ap.add_argument("--all",       action="store_true", help="Run all checks")
    args = ap.parse_args()

    if args.verify or args.all:
        verify_dataset()
    if args.integrity or args.all:
        check_integrity()
    if args.augment or args.all:
        augment_dataset(args.target)
    if args.summary or args.all:
        summary()
    if not any(vars(args).values()):
        ap.print_help()
        print()
        _print_setup_instructions()
