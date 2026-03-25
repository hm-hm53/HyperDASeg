import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

COLOR_MAP = OrderedDict(
    BgClutter=[255, 0, 0],
    imp_surf=[255, 255, 255],
    building=[0, 0, 255],
    low_vege=[0, 255, 255],
    tree=[0, 255, 0],
    car=[255, 255, 0],
)

def decode_segmap(mask, color_map):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    id_to_color = {i: color for i, color in enumerate(color_map.values())}
    for class_id, color in id_to_color.items():
        color_mask[mask == class_id] = color
    return color_mask

def overlay_mask(image, color_mask, alpha=0.5):
    return ((1 - alpha) * image + alpha * color_mask).astype(np.uint8)

def save_rgb(path, image_rgb):
    cv2.imwrite(path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

# input
image_dir = "data/IsprsDA/Vaihingen/img_dir/test"

# mask
mask_dir = "/log/hyperdaseg/SegFormer_MiT-B2/vaihingen/src_warmup_s/ids-Vaihingen_stu_best.pth-test"

# output
overlay_dir = "log/hyperdaseg/SegFormer_MiT-B2/vaihingen/src_warmup_s/overlay_ids-Vaihingen_stu_best.pth-test"
os.makedirs(overlay_dir, exist_ok=True)

valid_ext = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

image_names = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(valid_ext)])

for name in image_names:
    image_path = os.path.join(image_dir, name)
    mask_path = os.path.join(mask_dir, os.path.splitext(name)[0] + ".png")

    if not os.path.exists(mask_path):
        print(f"Skip, mask not found: {mask_path}")
        continue

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Skip, cannot read image: {image_path}")
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        print(f"Skip, cannot read mask: {mask_path}")
        continue
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    color_mask = decode_segmap(mask, COLOR_MAP)
    overlay = overlay_mask(image, color_mask, alpha=0.5)

    save_path = os.path.join(overlay_dir, os.path.splitext(name)[0] + ".png")
    save_rgb(save_path, overlay)
    # print(f"Saved: {save_path}")

print("Done.")

