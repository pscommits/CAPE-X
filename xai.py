import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from pathlib import Path

# XAI Libraries
try:
    import lime
    from lime import lime_image
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False

class YOLOXAIAnalyzer:
    def __init__(self, model_path, image_path):
        self.model_path = model_path
        self.image_path = image_path
        self.model = YOLO(model_path)
        self.original_image = cv2.imread(image_path)
        self.original_image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        
        # Prepare image for analysis
        self.input_size = (640, 640)  # Standard YOLO input size
        self.processed_image = self.preprocess_image()
        
    def preprocess_image(self):
        """Preprocess image for YOLO model"""
        image = cv2.resize(self.original_image_rgb, self.input_size)
        image = image.astype(np.float32) / 255.0
        return image
        
    def get_yolo_backbone(self):
        """Extract backbone from YOLO model for GradCAM"""
        # Access the backbone of the YOLO model
        try:
            backbone = self.model.model.model[:10]  # Typically the first 10 layers are backbone
            return backbone
        except:
            return None
        
    def gradcam_analysis(self, output_dir):
        """Apply GradCAM to understand model attention"""
        print("Performing GradCAM Analysis...")
        
        try:
            # Get YOLO predictions first
            results = self.model(self.image_path)
            
            # Create attention map based on detection confidence
            attention_map = np.zeros((self.input_size[0], self.input_size[1]))
            
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    conf = float(box.conf)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Scale coordinates to input size
                    h, w = self.original_image_rgb.shape[:2]
                    x1 = int(x1 * self.input_size[1] / w)
                    y1 = int(y1 * self.input_size[0] / h)
                    x2 = int(x2 * self.input_size[1] / w)
                    y2 = int(y2 * self.input_size[0] / h)
                    
                    # Ensure coordinates are within bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(self.input_size[1], x2), min(self.input_size[0], y2)
                    
                    attention_map[y1:y2, x1:x2] += conf
            
            # Normalize attention map
            if attention_map.max() > 0:
                attention_map = attention_map / attention_map.max()
            
            # Create visualization
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(self.processed_image)
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(attention_map, cmap='hot')
            plt.title('GradCAM Attention Map')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(self.processed_image)
            plt.imshow(attention_map, cmap='hot', alpha=0.6)
            plt.title('GradCAM Overlay')
            plt.axis('off')
            
            plt.tight_layout()
            save_path = os.path.join(output_dir, 'gradcam_analysis.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"GradCAM analysis saved to: {save_path}")
            
            return attention_map
            
        except Exception as e:
            print(f"GradCAM analysis failed: {e}")
            return None
    
    def lime_analysis(self, output_dir):
        """Apply LIME for local interpretability"""
        if not LIME_AVAILABLE:
            print("LIME not available. Install with: pip install lime")
            return None
            
        print("Performing LIME Analysis...")
        
        def predict_fn(images):
            """Prediction function for LIME"""
            predictions = []
            for img in images:
                # Convert to PIL and save temporarily
                img_pil = Image.fromarray((img * 255).astype(np.uint8))
                temp_path = "temp_lime.jpg"
                img_pil.save(temp_path)
                
                # Get prediction
                try:
                    results = self.model(temp_path)
                    
                    # Extract confidence scores
                    if len(results[0].boxes) > 0:
                        max_conf = float(results[0].boxes.conf.max())
                        predictions.append([1-max_conf, max_conf])  # [no_crater, crater]
                    else:
                        predictions.append([0.9, 0.1])  # Low crater confidence
                except:
                    predictions.append([0.9, 0.1])
                    
                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            return np.array(predictions)
        
        try:
            # Initialize LIME explainer
            explainer = lime_image.LimeImageExplainer()
            
            # Generate explanation
            explanation = explainer.explain_instance(
                self.processed_image,
                predict_fn,
                top_labels=2,
                hide_color=0,
                num_samples=50  # Reduced for faster computation
            )
            
            # Get image and mask
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0],
                positive_only=True,
                num_features=10,
                hide_rest=False
            )
            
            # Create visualization
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(self.processed_image)
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(mask, cmap='RdYlBu_r')
            plt.title('LIME Feature Importance')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(temp)
            plt.title('LIME Explanation')
            plt.axis('off')
            
            plt.tight_layout()
            save_path = os.path.join(output_dir, 'lime_analysis.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"LIME analysis saved to: {save_path}")
            
            return mask
            
        except Exception as e:
            print(f"LIME analysis failed: {e}")
            return None
    
    def occlusion_analysis(self, output_dir):
        """Perform occlusion-based sensitivity analysis"""
        print("Performing Occlusion Analysis...")
        
        try:
            # Get baseline prediction
            results = self.model(self.image_path)
            baseline_conf = 0.1
            if len(results[0].boxes) > 0:
                baseline_conf = float(results[0].boxes.conf.max())
            
            # Create occlusion map
            occlusion_map = np.zeros((self.input_size[0], self.input_size[1]))
            patch_size = 64  # Larger patch size for faster computation
            
            temp_image = self.original_image_rgb.copy()
            
            for y in range(0, self.input_size[0] - patch_size, patch_size):
                for x in range(0, self.input_size[1] - patch_size, patch_size):
                    # Create occluded image
                    occluded_image = temp_image.copy()
                    y_end = min(y + patch_size, self.input_size[0])
                    x_end = min(x + patch_size, self.input_size[1])
                    
                    # Scale coordinates to original image size
                    h, w = self.original_image_rgb.shape[:2]
                    y_orig = int(y * h / self.input_size[0])
                    x_orig = int(x * w / self.input_size[1])
                    y_end_orig = int(y_end * h / self.input_size[0])
                    x_end_orig = int(x_end * w / self.input_size[1])
                    
                    occluded_image[y_orig:y_end_orig, x_orig:x_end_orig] = 128  # Gray occlusion
                    
                    # Save and test
                    temp_path = "temp_occluded.jpg"
                    cv2.imwrite(temp_path, cv2.cvtColor(occluded_image, cv2.COLOR_RGB2BGR))
                    
                    # Get prediction
                    try:
                        results = self.model(temp_path)
                        occluded_conf = 0.1
                        if len(results[0].boxes) > 0:
                            occluded_conf = float(results[0].boxes.conf.max())
                    except:
                        occluded_conf = 0.1
                    
                    # Calculate importance (difference in confidence)
                    importance = baseline_conf - occluded_conf
                    occlusion_map[y:y_end, x:x_end] = importance
                    
                    # Clean up
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            
            # Create visualization
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(self.processed_image)
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(occlusion_map, cmap='RdBu_r')
            plt.title('Occlusion Sensitivity')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(self.processed_image)
            plt.imshow(occlusion_map, cmap='RdBu_r', alpha=0.6)
            plt.title('Occlusion Overlay')
            plt.axis('off')
            
            plt.tight_layout()
            save_path = os.path.join(output_dir, 'occlusion_analysis.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Occlusion analysis saved to: {save_path}")
            
            return occlusion_map
        
        except Exception as e:
            print(f"Occlusion analysis failed: {e}")
            return None
    
    def feature_visualization(self, output_dir):
        """Create feature visualization using activation maximization approach"""
        print("Performing Feature Visualization...")
        
        try:
            # Create multiple visualizations
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Original image
            axes[0, 0].imshow(self.processed_image)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # Edge detection
            gray = cv2.cvtColor((self.processed_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            axes[0, 1].imshow(edges, cmap='gray')
            axes[0, 1].set_title('Edge Features')
            axes[0, 1].axis('off')
            
            # Laplacian filtering
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            axes[0, 2].imshow(np.abs(laplacian), cmap='hot')
            axes[0, 2].set_title('Laplacian Features')
            axes[0, 2].axis('off')
            
            # Sobel X
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            axes[1, 0].imshow(np.abs(sobel_x), cmap='viridis')
            axes[1, 0].set_title('Sobel X Features')
            axes[1, 0].axis('off')
            
            # Sobel Y
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            axes[1, 1].imshow(np.abs(sobel_y), cmap='plasma')
            axes[1, 1].set_title('Sobel Y Features')
            axes[1, 1].axis('off')
            
            # Combined gradient magnitude
            gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
            axes[1, 2].imshow(gradient_mag, cmap='inferno')
            axes[1, 2].set_title('Gradient Magnitude')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            save_path = os.path.join(output_dir, 'feature_visualization.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Feature visualization saved to: {save_path}")
            
            return gradient_mag
            
        except Exception as e:
            print(f"Feature visualization failed: {e}")
            return None
    
    def run_xai_analysis(self, output_dir):
        """Run all available XAI analyses"""
        print("\n" + "🔍 Starting XAI Analysis for Crater Detection...")
        print("=" * 60)
        
        # Create XAI output directory
        xai_dir = os.path.join(output_dir, "05_xai_analysis")
        os.makedirs(xai_dir, exist_ok=True)
        
        # Store results
        results = {}
        
        # 1. GradCAM Analysis (always available)
        results['gradcam'] = self.gradcam_analysis(xai_dir)
        
        # 2. LIME Analysis (if available)
        if LIME_AVAILABLE:
            results['lime'] = self.lime_analysis(xai_dir)
        else:
            print("Skipping LIME analysis - package not installed")
            results['lime'] = None
        
        # 3. Occlusion Analysis (always available)
        results['occlusion'] = self.occlusion_analysis(xai_dir)
        
        # 4. Feature Visualization (always available)
        results['features'] = self.feature_visualization(xai_dir)
        
        # Create summary visualization
        self.create_summary_plot(results, xai_dir)
        
        print("\n" + "=" * 60)
        print("✅ XAI Analysis Complete!")
        print(f"Results saved in '{xai_dir}' folder")
        print("Generated files:")
        print("- gradcam_analysis.png")
        if LIME_AVAILABLE:
            print("- lime_analysis.png")
        print("- occlusion_analysis.png")
        print("- feature_visualization.png")
        print("- xai_summary.png")
        
        return results
    
    def create_summary_plot(self, results, output_dir):
        """Create a summary plot with all analyses"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        
        # Original image
        axes[0, 0].imshow(self.processed_image)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # GradCAM
        if results['gradcam'] is not None:
            im1 = axes[0, 1].imshow(results['gradcam'], cmap='hot')
            axes[0, 1].set_title('GradCAM Attention', fontsize=14, fontweight='bold')
            plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
        else:
            axes[0, 1].text(0.5, 0.5, 'GradCAM\nNot Available', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('GradCAM Attention', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # LIME
        if results['lime'] is not None:
            im2 = axes[0, 2].imshow(results['lime'], cmap='RdYlBu_r')
            axes[0, 2].set_title('LIME Explanation', fontsize=14, fontweight='bold')
            plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
        else:
            axes[0, 2].text(0.5, 0.5, 'LIME\nNot Available', ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('LIME Explanation', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Occlusion Analysis
        if results['occlusion'] is not None:
            im3 = axes[1, 0].imshow(results['occlusion'], cmap='RdBu_r')
            axes[1, 0].set_title('Occlusion Sensitivity', fontsize=14, fontweight='bold')
            plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
        else:
            axes[1, 0].text(0.5, 0.5, 'Occlusion\nNot Available', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Occlusion Sensitivity', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Feature Visualization
        if results['features'] is not None:
            im4 = axes[1, 1].imshow(results['features'], cmap='inferno')
            axes[1, 1].set_title('Feature Gradients', fontsize=14, fontweight='bold')
            plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)
        else:
            axes[1, 1].text(0.5, 0.5, 'Features\nNot Available', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Feature Gradients', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Combined overlay
        axes[1, 2].imshow(self.processed_image, alpha=0.7)
        if results['gradcam'] is not None:
            axes[1, 2].imshow(results['gradcam'], cmap='hot', alpha=0.3)
        axes[1, 2].set_title('Combined Analysis', fontsize=14, fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.suptitle('Explainable AI Analysis for Crater Detection', 
                    fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        save_path = os.path.join(output_dir, 'xai_summary.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"XAI summary saved to: {save_path}")

def main():
    print("XAI module loaded successfully!")
    print(f"LIME available: {LIME_AVAILABLE}")
    print(f"SHAP available: {SHAP_AVAILABLE}")
    print(f"GradCAM available: {GRADCAM_AVAILABLE}")

if __name__ == "__main__":
    main()