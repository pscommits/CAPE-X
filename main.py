import os
import argparse
import shutil
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Import our custom modules
from yolo import YOLODetector
from sam import SAMSegmenter
from analysis import CraterAnalyzer

class CraterDetectionPipeline:
    def __init__(self, yolo_model_path, sam_model_path, meters_per_pixel=1.0):
        """
        Initialize the complete crater detection pipeline
        
        Args:
            yolo_model_path: Path to YOLO model (.pt file)
            sam_model_path: Path to SAM model (.pth file)
            meters_per_pixel: Scale factor for real-world measurements
        """
        print("Initializing Crater Detection Pipeline...")
        
        self.yolo_detector = YOLODetector(yolo_model_path)
        self.sam_segmenter = SAMSegmenter(sam_model_path)
        self.analyzer = CraterAnalyzer(meters_per_pixel)
        
        print("Pipeline initialized successfully!")
    
    def process_image(self, image_path, output_dir="output", conf_threshold=0.25):
        """
        Process a single image through the complete pipeline
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save all outputs
            conf_threshold: Confidence threshold for YOLO detection
            
        Returns:
            pandas.DataFrame: Final crater analysis data
        """
        # Create output directory structure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(output_dir, f"crater_analysis_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"CRATER DETECTION PIPELINE - {timestamp}")
        print(f"{'='*60}")
        print(f"Input image: {image_path}")
        print(f"Output directory: {run_dir}")
        
        # Save raw input image
        raw_image = cv2.imread(image_path)
        raw_image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(raw_image_rgb)
        plt.title("Raw Input Image")
        plt.axis("off")
        plt.savefig(os.path.join(run_dir, "01_raw_input.png"), 
                   bbox_inches='tight', dpi=150)
        plt.close()
        
        # Step 1: YOLO Detection
        print(f"\n🚀 Step 1: Running YOLO crater detection...")
        yolo_output_path = os.path.join(run_dir, "02_yolo_detections.png")
        yolo_output = self.yolo_detector.detect_craters(
            image_path, 
            conf_threshold=conf_threshold,
            save_path=yolo_output_path
        )
        
        # Step 2: SAM Segmentation
        print(f"\n🎯 Step 2: Running SAM segmentation...")
        sam_output_path = os.path.join(run_dir, "03_sam_segmentation.png")
        sam_output = self.sam_segmenter.segment_craters(
            yolo_output,
            save_path=sam_output_path
        )
        
        # Step 3: Scientific Analysis
        print(f"\n📊 Step 3: Performing scientific analysis...")
        analysis_dir = os.path.join(run_dir, "04_analysis")
        crater_df = self.analyzer.analyze_craters(
            sam_output,
            save_dir=analysis_dir
        )
        
        # Save crater data as CSV
        if not crater_df.empty:
            csv_path = os.path.join(run_dir, "crater_data.csv")
            crater_df.to_csv(csv_path, index=False)
            print(f"Crater data saved to: {csv_path}")
            
            # Display summary
            print(f"\n📋 ANALYSIS SUMMARY")
            print(f"{'='*30}")
            print(f"Total craters detected: {len(crater_df)}")
            print(f"Average diameter: {crater_df['Diameter_m'].mean():.2f} meters")
            print(f"Total crater area: {crater_df['Area_m2'].sum():.2f} m²")
            print(f"Largest crater: {crater_df['Diameter_m'].max():.2f} m")
            print(f"Smallest crater: {crater_df['Diameter_m'].min():.2f} m")
        
        # Create pipeline summary
        self._create_pipeline_summary(run_dir, image_path, crater_df)
        
        print(f"\n✅ Pipeline completed successfully!")
        print(f"All outputs saved to: {run_dir}")
        
        return crater_df
    
    def _create_pipeline_summary(self, run_dir, image_path, crater_df):
        """Create a summary report of the pipeline execution"""
        
        summary_path = os.path.join(run_dir, "pipeline_summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write("CRATER DETECTION PIPELINE SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input Image: {os.path.basename(image_path)}\n")
            f.write(f"Scale Factor: {self.analyzer.meters_per_pixel} meters/pixel\n\n")
            
            if not crater_df.empty:
                f.write("DETECTION RESULTS:\n")
                f.write(f"Total Craters: {len(crater_df)}\n")
                f.write(f"Average Diameter: {crater_df['Diameter_m'].mean():.2f} m\n")
                f.write(f"Standard Deviation: {crater_df['Diameter_m'].std():.2f} m\n")
                f.write(f"Total Area: {crater_df['Area_m2'].sum():.2f} m²\n")
                f.write(f"Largest Crater: {crater_df['Diameter_m'].max():.2f} m\n")
                f.write(f"Smallest Crater: {crater_df['Diameter_m'].min():.2f} m\n\n")
                
                f.write("OUTPUT FILES:\n")
                f.write("- 01_raw_input.png: Original input image\n")
                f.write("- 02_yolo_detections.png: YOLO bounding box detections\n")
                f.write("- 03_sam_segmentation.png: SAM segmentation overlay\n")
                f.write("- 04_analysis/: Scientific analysis plots\n")
                f.write("- crater_data.csv: Detailed crater measurements\n")
                f.write("- pipeline_summary.txt: This summary file\n")
            else:
                f.write("No craters detected in the input image.\n")

def main():
    parser = argparse.ArgumentParser(description='Crater Detection Pipeline')
    parser.add_argument('--yolo_model', default='best.pt', 
                       help='Path to YOLO model file')
    parser.add_argument('--sam_model', default='sam_vit_b.pth', 
                       help='Path to SAM model file')
    parser.add_argument('--image', default='test.png', 
                       help='Path to input image')
    parser.add_argument('--output_dir', default='output', 
                       help='Output directory for results')
    parser.add_argument('--scale', type=float, default=1.0, 
                       help='Meters per pixel scale factor')
    parser.add_argument('--conf', type=float, default=0.25, 
                       help='Confidence threshold for YOLO detection')
    
    args = parser.parse_args()
    
    # Auto-detect model files if they exist in models folder
    def find_model_file(model_name):
        possible_paths = [
            model_name,
            f"models/{model_name}",
            f"models/sam.pth" if "sam" in model_name else None
        ]
        for path in possible_paths:
            if path and os.path.exists(path):
                return path
        return model_name  # Return original if not found
    
    yolo_model_path = find_model_file(args.yolo_model)
    sam_model_path = find_model_file(args.sam_model)
    
    # Validate input files
    if not os.path.exists(yolo_model_path):
        print(f"Error: YOLO model file '{yolo_model_path}' not found!")
        print("Checked locations:")
        print(f"  - {args.yolo_model}")
        print(f"  - models/{args.yolo_model}")
        return
    
    if not os.path.exists(sam_model_path):
        print(f"Error: SAM model file '{sam_model_path}' not found!")
        print("Checked locations:")
        print(f"  - {args.sam_model}")
        print(f"  - models/{args.sam_model}")
        print(f"  - models/sam.pth")
        print("\nTip: Run 'python setup.py' to download SAM models")
        return
    
    if not os.path.exists(args.image):
        print(f"Error: Input image '{args.image}' not found!")
        return
    
    try:
        # Initialize and run pipeline
        pipeline = CraterDetectionPipeline(
            yolo_model_path=yolo_model_path,
            sam_model_path=sam_model_path,
            meters_per_pixel=args.scale
        )
        
        # Process the image
        results = pipeline.process_image(
            image_path=args.image,
            output_dir=args.output_dir,
            conf_threshold=args.conf
        )
        
    except Exception as e:
        print(f"Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()