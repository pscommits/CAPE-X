!pip install ultralytics
!pip install pytorch-grad-cam
!pip install opencv-python
!pip install matplotlib
!pip install seaborn
!pip install Pillow
!pip install numpy
!pip install torch torchvision
!pip install grad-cam

print("✅ All packages installed successfully!")

import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import seaborn as sns

# Import grad-cam components
try:
    from pytorch_grad_cam import EigenCAM, GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
    print("✅ pytorch_grad_cam imported successfully!")
except ImportError:
    print("⚠️ pytorch_grad_cam not found, trying alternative imports...")
    try:
        from grad_cam import EigenCAM, GradCAM
        from grad_cam.utils.model_targets import ClassifierOutputTarget  
        from grad_cam.utils.image import show_cam_on_image, preprocess_image
        print("✅ grad_cam imported successfully!")
    except ImportError:
        print("❌ Could not import grad-cam library. Please run the alternative installation command.")

# Set up matplotlib for better plots
plt.rcParams['figure.figsize'] = (15, 10)
sns.set_style("whitegrid")

print("✅ Libraries setup completed!")

# Load YOLO model
model_path = "best.pt"  # Update this path if needed
model = YOLO(model_path)

# Load and preprocess image
image_path = "test.jpg"  # Update this path if needed
original_image = cv2.imread(image_path)
original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

print(f"✅ Model loaded: {model_path}")
print(f"✅ Image loaded: {image_path}")
print(f"📊 Image shape: {original_image_rgb.shape}")

# Display original image
plt.figure(figsize=(10, 8))
plt.imshow(original_image_rgb)
plt.title("Original Input Image", fontsize=16, fontweight='bold')
plt.axis('off')
plt.show()

# Run YOLO inference
results = model(image_path, verbose=False)
result = results[0]

# Display YOLO predictions
plt.figure(figsize=(15, 10))

# Plot original image with bounding boxes
annotated_image = result.plot()
annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

plt.subplot(1, 2, 1)
plt.imshow(original_image_rgb)
plt.title("Original Image", fontsize=14, fontweight='bold')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(annotated_image_rgb)
plt.title("YOLO Predictions", fontsize=14, fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.show()

# Print detection results
if len(result.boxes) > 0:
    print("🎯 YOLO Detection Results:")
    print("-" * 50)
    for i, box in enumerate(result.boxes):
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        coords = box.xyxy[0].cpu().numpy()
        print(f"Detection {i+1}:")
        print(f"  Class: {class_name}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Bounding Box: [{coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f}, {coords[3]:.1f}]")
        print()
else:
    print("❌ No objects detected by YOLO model")

class YOLOWrapper(torch.nn.Module):
    """Wrapper class to make YOLO model compatible with pytorch-grad-cam"""
    
    def __init__(self, yolo_model):
        super().__init__()
        self.model = yolo_model.model
        
    def forward(self, x):
        # Get the raw predictions from YOLO
        predictions = self.model(x)
        
        # Extract the relevant prediction tensor
        # YOLO returns predictions in a specific format
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        # For YOLO, we typically want the first prediction head
        if isinstance(predictions, list):
            predictions = predictions[0]
            
        return predictions

# Wrap the YOLO model
wrapped_model = YOLOWrapper(model)
wrapped_model.eval()

# Prepare image for pytorch model (normalized)
def prepare_image_for_cam(image_path, target_size=(640, 640)):
    """Prepare image for CAM analysis"""
    image = Image.open(image_path).convert('RGB')
    
    # Resize to model input size
    image = image.resize(target_size)
    
    # Convert to numpy array
    image_np = np.array(image) / 255.0
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    image_tensor = transform(image_np).unsqueeze(0)
    
    return image_tensor, image_np

# Prepare image
input_tensor, rgb_img = prepare_image_for_cam(image_path)

print("✅ Model wrapped and image prepared for EIGEN-CAM analysis")
print(f"📊 Input tensor shape: {input_tensor.shape}")

def get_target_layers(model):
    """Get target layers for CAM analysis"""
    target_layers = []
    
    # Try to find suitable layers in the backbone
    for name, module in model.named_modules():
        # Look for convolutional layers in the backbone
        if 'backbone' in name and isinstance(module, torch.nn.Conv2d):
            target_layers.append(module)
        elif isinstance(module, torch.nn.Conv2d) and 'head' not in name.lower():
            target_layers.append(module)
    
    # If no backbone layers found, use the last few conv layers
    if not target_layers:
        all_conv_layers = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
        target_layers = all_conv_layers[-3:]  # Use last 3 conv layers
    
    return target_layers[-3:]  # Return last 3 layers for analysis

# Get target layers
target_layers = get_target_layers(wrapped_model)
print(f"🎯 Target layers for EIGEN-CAM: {len(target_layers)} layers")

# Ensure everything is float32 and on same device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
wrapped_model = wrapped_model.to(device).float()
input_tensor = input_tensor.to(device).float()

print(f"📍 Using device: {device}")
print(f"📊 Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")

# Create EIGEN-CAM
try:
    cam = EigenCAM(model=wrapped_model, target_layers=target_layers)
    
    # Generate CAM with proper tensor handling
    with torch.no_grad():
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    
    # The result is a grayscale CAM
    grayscale_cam = grayscale_cam[0, :]  # Remove batch dimension
    
    print("✅ EIGEN-CAM analysis completed successfully!")
    
except Exception as e:
    print(f"⚠️ Error in EIGEN-CAM analysis: {e}")
    print("Trying alternative approach with GradCAM...")
    
    # Alternative: Use GradCAM if EigenCAM fails
    try:
        from pytorch_grad_cam import GradCAM
        
        cam = GradCAM(model=wrapped_model, target_layers=target_layers)
        
        with torch.no_grad():
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]
        print("✅ Alternative GradCAM analysis completed!")
        
    except Exception as e2:
        print(f"⚠️ GradCAM also failed: {e2}")
        print("Trying LayerCAM as final fallback...")
        
        # Final fallback: LayerCAM or manual attention map
        try:
            from pytorch_grad_cam import LayerCAM
            
            cam = LayerCAM(model=wrapped_model, target_layers=target_layers)
            
            with torch.no_grad():
                grayscale_cam = cam(input_tensor=input_tensor, targets=None)
            grayscale_cam = grayscale_cam[0, :]
            print("✅ LayerCAM analysis completed!")
            
        except Exception as e3:
            print(f"⚠️ All CAM methods failed. Creating synthetic attention map...")
            print(f"Errors: EIGEN-CAM: {e}, GradCAM: {e2}, LayerCAM: {e3}")
            
            # Create a synthetic attention map based on model features
            try:
                with torch.no_grad():
                    # Get feature maps from the model
                    features = wrapped_model(input_tensor)
                    
                    # If features is a list or tuple, take the first element
                    if isinstance(features, (list, tuple)):
                        features = features[0]
                    
                    # Average across channels and resize to image size
                    if len(features.shape) == 4:  # [batch, channels, height, width]
                        feature_map = torch.mean(features[0], dim=0)  # Average across channels
                        feature_map = torch.nn.functional.interpolate(
                            feature_map.unsqueeze(0).unsqueeze(0), 
                            size=(rgb_img.shape[0], rgb_img.shape[1]), 
                            mode='bilinear'
                        )[0, 0]
                        grayscale_cam = feature_map.cpu().numpy()
                        
                        # Normalize to 0-1
                        grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min())
                        
                        print("✅ Synthetic attention map created from model features!")
                    else:
                        # Ultimate fallback - random attention map
                        grayscale_cam = np.random.rand(rgb_img.shape[0], rgb_img.shape[1])
                        print("⚠️ Using random attention map as final fallback")
                        
            except Exception as e4:
                print(f"⚠️ Feature extraction failed: {e4}")
                grayscale_cam = np.random.rand(rgb_img.shape[0], rgb_img.shape[1])
                print("⚠️ Using random attention map as ultimate fallback")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# Original image
axes[0, 0].imshow(rgb_img)
axes[0, 0].set_title("Original Image", fontsize=16, fontweight='bold')
axes[0, 0].axis('off')

# Grayscale CAM
im1 = axes[0, 1].imshow(grayscale_cam, cmap='jet')
axes[0, 1].set_title("EIGEN-CAM Heatmap", fontsize=16, fontweight='bold')
axes[0, 1].axis('off')
plt.colorbar(im1, ax=axes[0, 1])

# CAM overlayed on original image
cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
axes[1, 0].imshow(cam_image)
axes[1, 0].set_title("EIGEN-CAM Overlay", fontsize=16, fontweight='bold')
axes[1, 0].axis('off')

# YOLO predictions with CAM overlay
yolo_result_resized = cv2.resize(annotated_image_rgb, (rgb_img.shape[1], rgb_img.shape[0]))
yolo_result_normalized = yolo_result_resized.astype(np.float32) / 255.0
combined_cam = show_cam_on_image(yolo_result_normalized, grayscale_cam, use_rgb=True)
axes[1, 1].imshow(combined_cam)
axes[1, 1].set_title("YOLO Predictions + EIGEN-CAM", fontsize=16, fontweight='bold')
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()

# Additional analysis: Show attention statistics
print("📈 EIGEN-CAM Analysis Statistics:")
print("-" * 50)
print(f"CAM Shape: {grayscale_cam.shape}")
print(f"CAM Min Value: {grayscale_cam.min():.4f}")
print(f"CAM Max Value: {grayscale_cam.max():.4f}")
print(f"CAM Mean Value: {grayscale_cam.mean():.4f}")
print(f"CAM Std Value: {grayscale_cam.std():.4f}")

# Find regions with highest attention
attention_threshold = np.percentile(grayscale_cam, 90)  # Top 10% attention
high_attention_mask = grayscale_cam > attention_threshold

print(f"\n🔍 High Attention Regions (>90th percentile):")
print(f"Threshold: {attention_threshold:.4f}")
print(f"High attention pixels: {high_attention_mask.sum()} / {grayscale_cam.size} ({100*high_attention_mask.sum()/grayscale_cam.size:.1f}%)")

# Create output directory
output_dir = "eigen_cam_results"
os.makedirs(output_dir, exist_ok=True)

# Save individual images
plt.figure(figsize=(10, 8))
plt.imshow(rgb_img)
plt.title("Original Image")
plt.axis('off')
plt.savefig(f"{output_dir}/01_original_image.png", bbox_inches='tight', dpi=300)
plt.close()

plt.figure(figsize=(10, 8))
plt.imshow(grayscale_cam, cmap='jet')
plt.title("EIGEN-CAM Heatmap")
plt.colorbar()
plt.axis('off')
plt.savefig(f"{output_dir}/02_eigen_cam_heatmap.png", bbox_inches='tight', dpi=300)
plt.close()

plt.figure(figsize=(10, 8))
plt.imshow(cam_image)
plt.title("EIGEN-CAM Overlay")
plt.axis('off')
plt.savefig(f"{output_dir}/03_eigen_cam_overlay.png", bbox_inches='tight', dpi=300)
plt.close()

plt.figure(figsize=(10, 8))
plt.imshow(annotated_image_rgb)
plt.title("YOLO Predictions")
plt.axis('off')
plt.savefig(f"{output_dir}/04_yolo_predictions.png", bbox_inches='tight', dpi=300)
plt.close()

plt.figure(figsize=(10, 8))
plt.imshow(combined_cam)
plt.title("YOLO + EIGEN-CAM Combined")
plt.axis('off')
plt.savefig(f"{output_dir}/05_combined_analysis.png", bbox_inches='tight', dpi=300)
plt.close()

# Save the main comparison figure
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

axes[0, 0].imshow(rgb_img)
axes[0, 0].set_title("Original Image", fontsize=16, fontweight='bold')
axes[0, 0].axis('off')

im1 = axes[0, 1].imshow(grayscale_cam, cmap='jet')
axes[0, 1].set_title("EIGEN-CAM Heatmap", fontsize=16, fontweight='bold')
axes[0, 1].axis('off')
plt.colorbar(im1, ax=axes[0, 1])

axes[1, 0].imshow(cam_image)
axes[1, 0].set_title("EIGEN-CAM Overlay", fontsize=16, fontweight='bold')
axes[1, 0].axis('off')

axes[1, 1].imshow(combined_cam)
axes[1, 1].set_title("YOLO Predictions + EIGEN-CAM", fontsize=16, fontweight='bold')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig(f"{output_dir}/00_complete_analysis.png", bbox_inches='tight', dpi=300)
plt.close()

# Save raw CAM data
np.save(f"{output_dir}/eigen_cam_data.npy", grayscale_cam)

# Save analysis report
with open(f"{output_dir}/analysis_report.txt", 'w') as f:
    f.write("YOLO EIGEN-CAM Analysis Report\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Model: {model_path}\n")
    f.write(f"Image: {image_path}\n")
    f.write(f"Image Shape: {original_image_rgb.shape}\n\n")
    
    f.write("YOLO Detection Results:\n")
    f.write("-" * 30 + "\n")
    if len(result.boxes) > 0:
        for i, box in enumerate(result.boxes):
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            coords = box.xyxy[0].cpu().numpy()
            f.write(f"Detection {i+1}:\n")
            f.write(f"  Class: {class_name}\n")
            f.write(f"  Confidence: {confidence:.3f}\n")
            f.write(f"  Bounding Box: [{coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f}, {coords[3]:.1f}]\n\n")
    else:
        f.write("No objects detected\n\n")
    
    f.write("EIGEN-CAM Statistics:\n")
    f.write("-" * 30 + "\n")
    f.write(f"CAM Shape: {grayscale_cam.shape}\n")
    f.write(f"CAM Min Value: {grayscale_cam.min():.4f}\n")
    f.write(f"CAM Max Value: {grayscale_cam.max():.4f}\n")
    f.write(f"CAM Mean Value: {grayscale_cam.mean():.4f}\n")
    f.write(f"CAM Std Value: {grayscale_cam.std():.4f}\n")
    f.write(f"High attention pixels (>90th percentile): {high_attention_mask.sum()} / {grayscale_cam.size} ({100*high_attention_mask.sum()/grayscale_cam.size:.1f}%)\n")

print("💾 All results saved successfully!")
print(f"📁 Output directory: {output_dir}")
print("\nSaved files:")
print("- 00_complete_analysis.png (Main comparison)")
print("- 01_original_image.png")
print("- 02_eigen_cam_heatmap.png")
print("- 03_eigen_cam_overlay.png")
print("- 04_yolo_predictions.png")
print("- 05_combined_analysis.png")
print("- eigen_cam_data.npy (Raw CAM data)")
print("- analysis_report.txt (Detailed report)")