import json
import numpy as np
import matplotlib.pyplot as plt

scene = "dataset/level_07_three_to_ten_objects/scene_00001"   # change to your scene folder

img  = np.load(f"{scene}/image.npy")     # (H, W, 3)  linear HDR radiance
gbuf = np.load(f"{scene}/gbuffer.npy")   # (H, W, 18) G-buffer
with open(f"{scene}/scene.json") as f:
    info = json.load(f)

print("image  ", img.shape, img.dtype, "min %.3f  max %.3f  mean %.3f" % (img.min(), img.max(), img.mean()))
print("gbuffer", gbuf.shape, gbuf.dtype)
print("level", info["level"], info["level_name"], "| split", info["split"], "|", len(info["objects"]), "objects")

# --- The rendered image ---------------------------------------------------------
# Values above 1 (the light, highlights) are real; for display, apply gamma 2 and
# clip to 0..1, the same as the PPM preview.
def to_display(x):
    return np.clip(np.sqrt(np.maximum(x, 0)), 0, 1)

plt.figure(figsize=(5, 5))
plt.imshow(to_display(img))
plt.title("image.npy (gamma 2, clipped for display)")
plt.axis("off")
plt.show()

# --- The G-buffer -------------------------------------------------------------------
depth     = gbuf[..., 0]
position  = gbuf[..., 1:4]
normal    = gbuf[..., 4:7]
albedo    = gbuf[..., 7:10]
view_dir  = gbuf[..., 10:13]
material  = gbuf[..., 13:16]    # one-hot: diffuse, metal, glass
roughness = gbuf[..., 16]
ior       = gbuf[..., 17]
hit       = depth > 0

def normalize01(x):             # stretch any range to 0..1 for display
    lo, hi = np.nanmin(x), np.nanmax(x)
    return (x - lo) / (hi - lo + 1e-8)

panels = [
    ("depth",                np.where(hit, depth, np.nan),        "viridis"),
    ("position",             normalize01(position) * hit[..., None], None),
    ("normal",               (normal * 0.5 + 0.5) * hit[..., None],  None),
    ("albedo",               to_display(albedo),                  None),
    ("view direction",       view_dir * 0.5 + 0.5,                None),
    ("material (R diffuse, G metal, B glass)", material,          None),
    ("roughness (metal)",    roughness,                           "magma"),
    ("IOR (glass)",          ior,                                 "magma"),
]
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, (title, data, cmap) in zip(axes.ravel(), panels):
    im = ax.imshow(data, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    if cmap is not None:
        fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.show()

# --- Looking up one pixel --------------------------------------------------------------
y, x = 128, 128
names = info["files"]["gbuffer_channels"]
print(f"pixel ({x}, {y}): radiance", img[y, x])
for name, value in zip(names, gbuf[y, x]):
    print(f"  {name:10s} {value: .4f}")
