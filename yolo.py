import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os


class YOLODetector:
    def __init__(self, model_path):
        """Initialize YOLO detector with model path"""
        self.model = YOLO(model_path)

    def detect_craters(self, image_path, conf_threshold=0.25, save_path=None):
        """
        Detect craters in image using YOLO
        
        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold for detection
            save_path: Path to save bounded box image
            
        Returns:
            dict: Contains image data, boxes, and detection results
        """
        # Load and process image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run YOLO inference
        results = self.model.predict(image_path, conf=conf_threshold)
        
        # Extract bounding boxes
        boxes = (
            results[0].boxes.xyxy.cpu().numpy().astype(int)
            if len(results[0].boxes) > 0 else np.array([])
        )
        
        print(f"Detected {len(boxes)} craters")
        
        # Create visualization with bounding boxes
        vis_image = self._draw_bounding_boxes(image_rgb.copy(), boxes)
        
        # Save bounded box image if path provided
        if save_path:
            plt.figure(figsize=(12, 8))
            plt.imshow(vis_image)
            plt.title("YOLO Crater Detection")
            plt.axis("off")
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
            print(f"Bounded box image saved to: {save_path}")
        
        # Prepare data for SAM
        sam_input = {
            'image_rgb': image_rgb,
            'boxes': boxes,
            'image_path': image_path
        }
        
        return sam_input

    def _draw_bounding_boxes(self, image, boxes):
        """Draw bounding boxes on image"""
        vis_image = image.copy()
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            # Draw rectangle
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (254, 255, 0), 2)
            # Add label
            cv2.putText(
                vis_image, f'Crater {i+1}', (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
            )
        
        return vis_image


def main():
    # Test the YOLO detector independently
    detector = YOLODetector("best.pt")
    result = detector.detect_craters("test.png", save_path="yolo_output.png")
    print("YOLO detection completed!")


if __name__ == "__main__":
    main()
