import numpy as np
import matplotlib.pyplot as plt


cm = np.load("//home/yons/下载/代码/hyperdaseg/confusion matrix/urban.npy")

classes = [
    "Backgd",
    "Building",
    "Road",
    "Water",
    "Barren",
    "Forest",
    "Agricult"
]

# classes = [
#     "BgClutter",
#     "Imp_surf",
#     "Building",
#     "Low_vege",
#     "Tree",
#     "Car",
# ]



cm = cm[1:, 1:]
classes = classes[1:]
cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
cm_norm = np.nan_to_num(cm_norm)



fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm_norm, cmap="Oranges", vmin=0, vmax=1)

ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(classes, rotation=45, ha="right")
ax.set_yticklabels(classes)


ax.set_xlabel("Predicted label", fontsize=14)
ax.set_ylabel("True label", fontsize=14)

cbar = plt.colorbar(im, ax=ax)


threshold = cm_norm.max() / 2.0
for i in range(cm_norm.shape[0]):
    for j in range(cm_norm.shape[1]):
        ax.text(
            j, i, f"{cm_norm[i, j]:.2%}",
            ha="center", va="center",
            color="white" if cm_norm[i, j] > threshold else "black",
            fontsize=14
        )

plt.tight_layout()
plt.savefig("urban.png", dpi=600, bbox_inches="tight", pad_inches=0)
plt.show()

