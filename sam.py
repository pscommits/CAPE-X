import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from skimage import measure

class SAMSegmenter:
    def __init__(self, sam_checkpoint, model_type="vit_b"):
        """Initialize SAM segmenter with checkpoint path"""
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.predictor = SamPredictor(self.sam)
        
    def segment_craters(self, yolo_output, save_path=None):
        """
        Segment craters using SAM based on YOLO detections
        
        Args:
            yolo_output: Dictionary from YOLO containing image and boxes
            save_path: Path to save segmented image
            
        Returns:
            dict: Contains masks and crater data for analysis
        """
        image_rgb = yolo_output['image_rgb']
        boxes = yolo_output['boxes']
        
        if len(boxes) == 0:
            print("No craters detected by YOLO, skipping segmentation")
            return {'masks': [], 'crater_data': [], 'image_rgb': image_rgb}
        
        # Set image for SAM predictor
        self.predictor.set_image(image_rgb)
        
        segmented_masks = []
        overlay = image_rgb.copy()
        
        # Pastel colors for visualization
        alpha = 0.7
        pastel_colors = [
            np.array([173, 216, 230]),  # Light blue
            np.array([255, 182, 193]),  # Light pink
            np.array([152, 251, 152]),  # Light green
            np.array([221, 160, 221]),  # Plum
            np.array([255, 228, 181])   # Moccasin
        ]
        
        print(f"Segmenting {len(boxes)} detected craters...")
        
        for i, box in enumerate(boxes):
            input_box = np.array(box)
            masks, scores, logits = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False
            )
            mask = masks[0]
            segmented_masks.append(mask)
            
            # Apply pastel overlay
            color = pastel_colors[i % len(pastel_colors)]
            overlay[mask] = (overlay[mask] * (1 - alpha) + color * alpha).astype(np.uint8)
            
            # Add black border around each segmented mask
            contours = measure.find_contours(mask.astype(np.uint8), 0.5)
            for contour in contours:
                contour = np.round(contour).astype(int)
                for c in contour:
                    rr, cc = c
                    if 0 <= rr < overlay.shape[0] and 0 <= cc < overlay.shape[1]:
                        overlay[rr, cc] = [0, 0, 0]  # Black pixel border
        
        # Save segmented image if path provided
        if save_path:
            plt.figure(figsize=(12, 8))
            plt.imshow(overlay)
            plt.title("YOLO + SAM Crater Segmentation")
            plt.axis("off")
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
            print(f"Segmented image saved to: {save_path}")
        
        crater_data = self._extract_crater_metrics(segmented_masks, boxes)
        
        analysis_input = {
            'masks': segmented_masks,
            'crater_data': crater_data,
            'image_rgb': image_rgb,
            'overlay': overlay
        }
        
        return analysis_input
    
    def _extract_crater_metrics(self, masks, boxes):
        """Extract basic metrics from segmented masks"""
        crater_data = []
        
        for i, mask in enumerate(masks):
            area_px = np.sum(mask)
            contours = measure.find_contours(mask.astype(np.uint8), 0.5)
            if len(contours) > 0:
                contour = max(contours, key=len)
                y, x = contour[:, 0], contour[:, 1]
                min_x, max_x, min_y, max_y = np.min(x), np.max(x), np.min(y), np.max(y)
                diameter_px = np.mean([max_x - min_x, max_y - min_y])
            else:
                diameter_px = np.sqrt(area_px/np.pi) * 2
            
            crater_data.append({
                "Crater_ID": i+1,
                "Area_px": int(area_px),
                "Diameter_px": float(diameter_px),
                "Box": boxes[i].tolist()
            })
        
        return crater_data

def main():
    print("SAM segmenter (pastel colors + black borders) loaded successfully!")

if __name__ == "__main__":
    main()
