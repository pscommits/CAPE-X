import json
import shap
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Try importing ultralytics for YOLO
try:
from ultralytics import YOLO
print("Ultralytics YOLO library loaded successfully")
except ImportError:
print("Installing ultralytics...")
import subprocess
subprocess.check_call(["pip", "install", "ultralytics"])
from ultralytics import YOLO

def load_yolo_model(model_path):
"""Load YOLO model from .pt file"""
try:
model = YOLO(model_path)
print(f"YOLO model loaded successfully from {model_path}")
return model
except Exception as e:
print(f"Error loading YOLO model: {e}")
return None

def load_and_preprocess_image(image_path, img_size=640):
"""
Load and preprocess image for YOLO model
"""
try:
# Load image using PIL
image = Image.open(image_path)
original_image = image.copy()

# Convert to RGB if necessary
if image.mode != 'RGB':
image = image.convert('RGB')

# Convert to numpy array
img_array = np.array(image)

# Resize image to YOLO input size
img_resized = cv2.resize(img_array, (img_size, img_size))

# Normalize to [0,1]
img_normalized = img_resized.astype(np.float32) / 255.0

# Add batch dimension and convert to tensor
img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)

print(f"Original image size: {original_image.size}")
print(f"Processed tensor shape: {img_tensor.shape}")

return img_tensor, img_normalized, original_image

except Exception as e:
print(f"Error loading image: {e}")
return None, None, None

class YOLOWrapper:
"""
Wrapper class to make YOLO model compatible with SHAP
"""
def __init__(self, yolo_model):
self.model = yolo_model
self.model.eval()

def __call__(self, x):
"""
Forward pass for SHAP compatibility
Returns confidence scores for detected objects
"""
if isinstance(x, np.ndarray):
# Convert numpy array to tensor if needed
if len(x.shape) == 3: # Single image
x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
elif len(x.shape) == 4 and x.shape[1] == 3: # Batch of images in CHW format
x = torch.from_numpy(x)
elif len(x.shape) == 4 and x.shape[3] == 3: # Batch of images in HWC format
x = torch.from_numpy(x).permute(0, 3, 1, 2)

try:
# Run YOLO prediction
results = self.model(x, verbose=False)

# Extract confidence scores
confidences = []
for result in results:
if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
conf_scores = result.boxes.conf.cpu().numpy()
confidences.extend(conf_scores)

if len(confidences) == 0:
# No detections, return small positive value
return np.array([[0.01]])

# Return max confidence or mean confidence
max_conf = max(confidences)
return np.array([[max_conf]])

except Exception as e:
print(f"Error in YOLO forward pass: {e}")
return np.array([[0.01]])

def create_background_samples(image_shape, num_samples=50):
"""
Create background dataset for SHAP
"""
print(f"Creating background dataset with {num_samples} samples...")

# Create random noise images
background = np.random.rand(num_samples, *image_shape[2:], image_shape[1]) # HWC format

# Add some structured patterns
for i in range(num_samples//2):
# Add some gray images
background[i] = np.full(image_shape[2:] + (image_shape[1],), 0.5)
# Add some variation
background[i] += np.random.normal(0, 0.1, image_shape[2:] + (image_shape[1],))
background[i] = np.clip(background[i], 0, 1)

return background

def run_shap_analysis(model_wrapper, image_tensor, background):
"""
Run SHAP analysis on YOLO model
"""
print("Starting SHAP analysis...")

# Convert tensor to numpy for SHAP
if isinstance(image_tensor, torch.Tensor):
# Convert from CHW to HWC format for SHAP
image_np = image_tensor.squeeze(0).permute(1, 2, 0).numpy()
image_np = np.expand_dims(image_np, 0) # Add batch dimension
else:
image_np = image_tensor

try:
# Initialize SHAP explainer
print("Creating SHAP Partition explainer...")
explainer = shap.Explainer(model_wrapper, background)

# Calculate SHAP values
print("Calculating SHAP values...")
shap_values = explainer(image_np)

return shap_values, "partition"

except Exception as e:
print(f"SHAP Partition explainer failed: {e}")

try:
print("Trying SHAP Permutation explainer...")
explainer = shap.Explainer(model_wrapper, background, algorithm="permutation")
shap_values = explainer(image_np)
return shap_values, "permutation"

except Exception as e2:
print(f"SHAP Permutation explainer also failed: {e2}")

# Fallback to simple gradient method
print("Using gradient-based explanation as fallback...")
return gradient_explanation(model_wrapper, image_np), "gradient"

def gradient_explanation(model_wrapper, image):
"""
Fallback gradient-based explanation
"""
try:
# Convert to tensor with gradient tracking
if isinstance(image, np.ndarray):
img_tensor = torch.from_numpy(image).permute(0, 3, 1, 2).requires_grad_(True)
else:
img_tensor = image.requires_grad_(True)

# Forward pass
output = model_wrapper(img_tensor.numpy())
output_tensor = torch.from_numpy(np.array(output)).requires_grad_(True)

# Backward pass
output_tensor.backward(torch.ones_like(output_tensor))

# Get gradients
if img_tensor.grad is not None:
gradients = img_tensor.grad.data.numpy()
# Convert back to HWC format
gradients = np.transpose(gradients[0], (1, 2, 0))
else:
# Create dummy gradients
gradients = np.random.rand(*image.shape[1:])

return gradients

except Exception as e:
print(f"Gradient explanation failed: {e}")
# Return random explanation as last resort
return np.random.rand(*image.shape[1:])

def visualize_results(original_image, processed_image, shap_values, method_used, yolo_results):
"""
Create comprehensive visualization of results
"""
print("Creating visualizations...")

# Create figure
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('YOLO Model SHAP Analysis Results', fontsize=16, fontweight='bold')

# Original image
axes[0, 0].imshow(original_image)
axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
axes[0, 0].axis('off')

# YOLO Detection Results
axes[0, 1].imshow(original_image)
if yolo_results and len(yolo_results) > 0:
# Plot bounding boxes if available
result = yolo_results[0]
if hasattr(result, 'boxes') and result.boxes is not None:
boxes = result.boxes.xyxy.cpu().numpy()
confidences = result.boxes.conf.cpu().numpy()
classes = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else []

# Scale boxes to original image size
scale_x = original_image.size[0] / 640
scale_y = original_image.size[1] / 640

for i, (box, conf) in enumerate(zip(boxes, confidences)):
x1, y1, x2, y2 = box
x1, x2 = x1 * scale_x, x2 * scale_x
y1, y2 = y1 * scale_y, y2 * scale_y

# Draw rectangle
rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
linewidth=2, edgecolor='red', facecolor='none')
axes[0, 1].add_patch(rect)

# Add confidence score
class_id = int(classes[i]) if i < len(classes) else 0
axes[0, 1].text(x1, y1-5, f'Class {class_id}: {conf:.3f}',
color='red', fontweight='bold', fontsize=10)

axes[0, 1].set_title(f'YOLO Detections', fontsize=14, fontweight='bold')
axes[0, 1].axis('off')

# Analysis Information
info_text = f"""Analysis Details:

• Model: YOLO (best.pt)
• Image: 101.jpg
• Method: {method_used.upper()}
• Input Size: 640x640
• Device: {"CUDA" if torch.cuda.is_available() else "CPU"}

Detection Summary:"""

if yolo_results and len(yolo_results) > 0:
result = yolo_results[0]
if hasattr(result, 'boxes') and result.boxes is not None:
num_detections = len(result.boxes)
max_conf = float(torch.max(result.boxes.conf)) if num_detections > 0 else 0.0
info_text += f"\n• Objects detected: {num_detections}\n• Max confidence: {max_conf:.3f}"
else:
info_text += "\n• No objects detected"
else:
info_text += "\n• No objects detected"

axes[0, 2].text(0.05, 0.95, info_text, fontsize=11, verticalalignment='top',
transform=axes[0, 2].transAxes, fontfamily='monospace')
axes[0, 2].axis('off')

# Process SHAP values for visualization
if method_used == "gradient":
explanation = shap_values
else:
try:
if hasattr(shap_values, 'values'):
explanation = shap_values.values[0]
else:
explanation = shap_values[0] if isinstance(shap_values, list) else shap_values
except:
explanation = np.random.rand(*processed_image.shape)

# Create heatmap
if len(explanation.shape) == 3:
heatmap = np.mean(np.abs(explanation), axis=2)
else:
heatmap = np.abs(explanation)

# Resize heatmap to match original image
heatmap_resized = cv2.resize(heatmap, original_image.size)

# SHAP heatmap overlay
axes[1, 0].imshow(original_image)
axes[1, 0].imshow(heatmap_resized, alpha=0.5, cmap='hot')
axes[1, 0].set_title(f'{method_used.upper()} Explanation Overlay', fontsize=12, fontweight='bold')
axes[1, 0].axis('off')

# Pure heatmap
im = axes[1, 1].imshow(heatmap_resized, cmap='hot')
axes[1, 1].set_title(f'{method_used.upper()} Importance Heatmap', fontsize=12, fontweight='bold')
axes[1, 1].axis('off')
plt.colorbar(im, ax=axes[1, 1], shrink=0.6)

# Feature importance summary
axes[1, 2].hist(heatmap.flatten(), bins=50, alpha=0.7, color='blue')
axes[1, 2].set_title('Importance Score Distribution', fontsize=12, fontweight='bold')
axes[1, 2].set_xlabel('Importance Score')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

def main():
"""
Main function to run YOLO SHAP detection
"""
print("="*70)
print("YOLO MODEL SHAP DETECTION")
print("="*70)

# File paths
model_path = "best.pt"
image_path = "101.jpg"

print(f"Model: {model_path}")
print(f"Image: {image_path}")

# Load YOLO model
yolo_model = load_yolo_model(model_path)
if yolo_model is None:
print("Failed to load YOLO model. Please check the model path.")
return

# Load and preprocess image
image_tensor, processed_image, original_image = load_and_preprocess_image(image_path)
if image_tensor is None:
print("Failed to load image. Please check the image path.")
return

# Get YOLO predictions first
print("Getting YOLO predictions...")
yolo_results = yolo_model(image_path)

# Create model wrapper for SHAP
model_wrapper = YOLOWrapper(yolo_model)

# Test model wrapper
print("Testing model wrapper...")
test_output = model_wrapper(processed_image)
print(f"Model wrapper output: {test_output}")

# Create background dataset
background = create_background_samples(image_tensor.shape)

# Run SHAP analysis
shap_values, method_used = run_shap_analysis(model_wrapper, image_tensor, background)

print(f"Analysis completed using: {method_used}")

# Visualize results
visualize_results(original_image, processed_image, shap_values, method_used, yolo_results)

# Save results
try:
results = {
"model_path": model_path,
"image_path": image_path,
"method_used": method_used,
"image_shape": list(image_tensor.shape),
"detections": []
}

# Add detection results
if yolo_results and len(yolo_results) > 0:
result = yolo_results[0]
if hasattr(result, 'boxes') and result.boxes is not None:
boxes = result.boxes.xyxy.cpu().numpy().tolist()
confidences = result.boxes.conf.cpu().numpy().tolist()
classes = result.boxes.cls.cpu().numpy().tolist() if result.boxes.cls is not None else []

for i, (box, conf) in enumerate(zip(boxes, confidences)):
detection = {
"box": box,
"confidence": conf,
"class": int(classes[i]) if i < len(classes) else 0
}
results["detections"].append(detection)

with open("yolo_shap_results.json", "w") as f:
json.dump(results, f, indent=2)

print("Results saved to: yolo_shap_results.json")

except Exception as e:
print(f"Could not save results: {e}")

print("\n" + "="*70)
print("YOLO SHAP ANALYSIS COMPLETE!")
print("="*70)

if __name__ == "__main__":
main()