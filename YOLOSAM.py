# 🟢 Block 1: Install dependencies
!pip install ultralytics
!pip install git+https://github.com/facebookresearch/segment-anything.git
!pip install opencv-python matplotlib

# Download SAM model weights (ViT-B by default, you can switch to H or L)
!wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O sam_vit_b.pth

# 🟢 Block 2: Import libraries
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np

from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# 🟢 Block 3: Load YOLO model
yolo_model = YOLO("best.pt")  # your trained crater model

# 🟢 Block 4: Show raw input image
img_path = "test.jpg"
image = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.title("Raw Input Image")
plt.axis("off")
plt.show()

# 🟢 Block 5: Run YOLO inference
results = yolo_model.predict(img_path, conf=0.25)  # adjust conf threshold

# Show YOLO detections
results[0].show()   # opens in new window (Colab may show inline with cv2_imshow)

# 🟢 Block 6: Extract YOLO bounding boxes
# Format: [x1, y1, x2, y2]
boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

print("Detected boxes:", boxes)

# 🟢 Block 7: Load SAM model
sam_checkpoint = "sam_vit_b.pth"
model_type = "vit_b"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)
predictor.set_image(image_rgb)

# Block 8: Run SAM segmentation with pastel transparency
segmented_masks = []
overlay = image_rgb.copy()

alpha = 0.4
pastel_colors = [
    np.array([173, 216, 230]),  # Light blue
    np.array([255, 182, 193]),  # Light pink
    np.array([152, 251, 152]),  # Light green
    np.array([221, 160, 221]),  # Plum
    np.array([255, 228, 181])   # Moccasin
]

for i, box in enumerate(boxes):
    input_box = np.array(box)
    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False
    )
    mask = masks[0]
    segmented_masks.append(mask)

    color = pastel_colors[i % len(pastel_colors)]
    overlay[mask] = (overlay[mask] * (1 - alpha) + color * alpha).astype(np.uint8)

plt.figure(figsize=(10,10))
plt.imshow(overlay)
plt.title("YOLO + SAM Pastel Segmentation")
plt.axis("off")
plt.show()

# Block 9: Extract crater metrics
import pandas as pd
from skimage import measure

crater_data = []

for i, mask in enumerate(segmented_masks):
    # Pixel area (count of True pixels)
    area_px = np.sum(mask)

    # Extract contours to estimate diameter
    contours = measure.find_contours(mask.astype(np.uint8), 0.5)
    if len(contours) > 0:
        contour = max(contours, key=len)
        y, x = contour[:, 0], contour[:, 1]
        min_x, max_x, min_y, max_y = np.min(x), np.max(x), np.min(y), np.max(y)
        diameter_px = np.mean([max_x - min_x, max_y - min_y])
    else:
        # fallback: equivalent circle diameter
        diameter_px = np.sqrt(area_px/np.pi) * 2

    crater_data.append({
        "Crater_ID": i+1,
        "Area_px": int(area_px),
        "Diameter_px": float(diameter_px),
        "Box": boxes[i].tolist()
    })

df = pd.DataFrame(crater_data)

# ✅ Show results directly in Colab
print("Crater metrics extracted:")
print(df)

# Or use a nice table
from IPython.display import display
display(df)

# Block 10: Complex crater analysis
import matplotlib.pyplot as plt

# Histogram of crater diameters
plt.figure(figsize=(8,6))
plt.hist(df["Diameter_px"], bins=10, color="skyblue", edgecolor="black")
plt.title("Crater Diameter Distribution")
plt.xlabel("Diameter (pixels)")
plt.ylabel("Count")
plt.show()

# Crater area vs diameter scatter
plt.figure(figsize=(8,6))
plt.scatter(df["Diameter_px"], df["Area_px"], c="purple", alpha=0.6)
plt.title("Crater Area vs Diameter")
plt.xlabel("Diameter (px)")
plt.ylabel("Area (px^2)")
plt.show()

# Spatial distribution (centroids on map)
plt.figure(figsize=(10,10))
plt.imshow(image_rgb)
for i, mask in enumerate(segmented_masks):
    coords = np.column_stack(np.where(mask))
    y_mean, x_mean = coords.mean(axis=0)
    plt.scatter(x_mean, y_mean, marker="o", c="red", s=50)
plt.title("Spatial Distribution of Craters")
plt.axis("off")
plt.show()

# Density plot (simple heatmap)
density_map = np.zeros(mask.shape)
for mask in segmented_masks:
    density_map += mask.astype(int)

plt.figure(figsize=(10,8))
plt.imshow(density_map, cmap="hot", alpha=0.6)
plt.colorbar(label="Crater Density")
plt.title("Crater Density Heatmap")
plt.axis("off")
plt.show()

# 🔧 Config: Define image scale
# Example: if 1 pixel = 5 meters, set this to 5
meters_per_pixel = 5.0  

# Automatically add real-world columns to the dataframe
df["Diameter_m"] = df["Diameter_px"] * meters_per_pixel
df["Area_m2"] = df["Area_px"] * (meters_per_pixel**2)

print("Scale applied: 1 pixel =", meters_per_pixel, "meters")
display(df.head())

import matplotlib.pyplot as plt

# ----- 1. Histogram of crater diameters -----
plt.figure(figsize=(8,6))
plt.hist(df["Diameter_m"], bins=10, color="skyblue", edgecolor="black")
plt.title("Crater Diameter Distribution")
plt.xlabel("Diameter (meters)")
plt.ylabel("Count")
plt.show()

# ----- 2. Scatter plot: Area vs Diameter -----
plt.figure(figsize=(8,6))
plt.scatter(df["Diameter_m"], df["Area_m2"], c="purple", alpha=0.6)
plt.title("Crater Area vs Diameter")
plt.xlabel("Diameter (m)")
plt.ylabel("Area (m²)")
plt.show()

# ----- 3. Spatial distribution (centroids on original image) -----
plt.figure(figsize=(10,10))
plt.imshow(image_rgb)
for i, mask in enumerate(segmented_masks):
    coords = np.column_stack(np.where(mask))
    y_mean, x_mean = coords.mean(axis=0)
    plt.scatter(x_mean, y_mean, marker="o", c="red", s=50, label=f"Crater {i+1}" if i==0 else "")
plt.title("Spatial Distribution of Craters")
plt.axis("off")
plt.legend()
plt.show()

# ----- 4. Density heatmap -----
density_map = np.zeros(mask.shape)
for mask in segmented_masks:
    density_map += mask.astype(int)

plt.figure(figsize=(10,8))
plt.imshow(image_rgb, alpha=0.5)
plt.imshow(density_map, cmap="hot", alpha=0.6)
plt.colorbar(label="Crater Density")
plt.title("Crater Density Heatmap")
plt.axis("off")
plt.show()

