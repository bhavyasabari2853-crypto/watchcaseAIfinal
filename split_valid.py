import os
import shutil
from collections import defaultdict

RAW_DIR = "dataset/raw"
TRAIN_IMG = "dataset/images/train"
TRAIN_LBL = "dataset/labels/train"
VAL_IMG = "dataset/images/valid"
VAL_LBL = "dataset/labels/valid"

os.makedirs(VAL_IMG, exist_ok=True)
os.makedirs(VAL_LBL, exist_ok=True)

# Map filename -> case folder
raw_map = {}
for case_dir in sorted(os.listdir(RAW_DIR)):
    case_path = os.path.join(RAW_DIR, case_dir)
    if not os.path.isdir(case_path):
        continue
    for f in os.listdir(case_path):
        raw_map[f.lower()] = case_dir

# Map train images to their case
case_images = defaultdict(list)
for f in os.listdir(TRAIN_IMG):
    fp = os.path.join(TRAIN_IMG, f)
    if not os.path.isfile(fp):
        continue
    case = raw_map.get(f.lower())
    if case:
        case_images[case].append(f)

print("Images per case in train:")
for c in sorted(case_images):
    print(f"  {c}: {len(case_images[c])} images")

# Pick 1 image per case, move to valid
moved = []
for case in sorted(case_images):
    imgs = case_images[case]
    # Pick first one
    pick = imgs[0]
    stem = os.path.splitext(pick)[0]
    lbl = stem + ".txt"

    src_img = os.path.join(TRAIN_IMG, pick)
    dst_img = os.path.join(VAL_IMG, pick)
    shutil.move(src_img, dst_img)

    src_lbl = os.path.join(TRAIN_LBL, lbl)
    dst_lbl = os.path.join(VAL_LBL, lbl)
    if os.path.exists(src_lbl):
        shutil.move(src_lbl, dst_lbl)
        lbl_ok = "yes"
    else:
        lbl_ok = "MISSING"
    moved.append((case, pick, lbl_ok))

print("\nMoved 1 per case:")
for c, img, lbl_ok in moved:
    print(f"  {c}: {img}  (label: {lbl_ok})")

# Update data.yaml to point val to images/valid
yaml_path = "dataset/data.yaml"
with open(yaml_path, 'r') as f:
    content = f.read()
content = content.replace("val: images/train", "val: images/valid")
with open(yaml_path, 'w') as f:
    f.write(content)

print("\ndata.yaml updated: val -> images/valid")

# Final counts
train_imgs = len([f for f in os.listdir(TRAIN_IMG) if os.path.isfile(os.path.join(TRAIN_IMG, f))])
val_imgs = len([f for f in os.listdir(VAL_IMG) if os.path.isfile(os.path.join(VAL_IMG, f))])
train_lbls = len([f for f in os.listdir(TRAIN_LBL) if os.path.isfile(os.path.join(TRAIN_LBL, f)) and f != 'train.cache'])
val_lbls = len([f for f in os.listdir(VAL_LBL) if os.path.isfile(os.path.join(VAL_LBL, f))])

print(f"\nFinal:")
print(f"  Train images: {train_imgs}")
print(f"  Train labels: {train_lbls}")
print(f"  Valid images: {val_imgs}")
print(f"  Valid labels: {val_lbls}")
