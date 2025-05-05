import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# parameters - need to set both num_images and rows, cols to match accordingly
num_images = 500
rows, cols = 25, 20
img_size = (224, 224)

# config
base_dir = os.path.expanduser("~/Documents/OEIS_Sequence_Repository")
csv_path = os.path.join(base_dir, "sequence_cluster_scores.csv")
linear_dir = os.path.join(base_dir, "Linear_Scatterplots")
log_dir = os.path.join(base_dir, "Logarithmic_Scatterplots")
output_image = os.path.join(base_dir, f"top_{num_images}_outliers_aligned.png")
filter_noise_only = True  # Set to True to restrict to cluster_label == -1

# load and filter scores to top num_images
df = pd.read_csv(csv_path)
if filter_noise_only:
    df = df[df["cluster_label"] == -1]
top_sequences = df.nsmallest(num_images, "isolation_score")

# Load and store image data
images, labels = [], []
for seq_id in top_sequences["sequence_id"]:
    for subdir in [linear_dir, log_dir]:
        file_path = os.path.join(subdir, f"{seq_id}.png")
        if os.path.exists(file_path):
            img = Image.open(file_path).convert("RGB").resize(img_size)
            images.append(img)
            labels.append(seq_id)
            break

# Create and save aligned plot
fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
for ax, img, label in zip(axes.flatten(), images, labels):
    ax.imshow(img)
    ax.set_title(label, fontsize=10, pad=2)
    ax.axis("off")

# Hide empty cells if fewer than 100 images
for i in range(len(images), rows * cols):
    axes.flatten()[i].axis("off")

plt.suptitle(f"Top {num_images} Most Interesting Cases (Original Images)", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.savefig(output_image, dpi=300)
plt.close()

print(f"Saved image grid to: {output_image}")
