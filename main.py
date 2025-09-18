import os
import argparse
import shutil
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import glob

# Import our custom modules
from yolo import YOLODetector
from sam import SAMSegmenter
from analysis import CraterAnalyzer
from xai import YOLOXAIAnalyzer

class CraterDetectionPipeline:
    def __init__(self, yolo_model_path, sam_model_path, meters_per_pixel=1.0, enable_xai=True):
        """
        Initialize the complete crater detection pipeline
        
        Args:
            yolo_model_path: Path to YOLO model (.pt file)
            sam_model_path: Path to SAM model (.pth file)
            meters_per_pixel: Scale factor for real-world measurements
            enable_xai: Whether to enable XAI analysis
        """
        print("Initializing Crater Detection Pipeline...")
        
        self.yolo_detector = YOLODetector(yolo_model_path)
        self.sam_segmenter = SAMSegmenter(sam_model_path)
        self.analyzer = CraterAnalyzer(meters_per_pixel)
        
        # XAI components
        self.enable_xai = enable_xai
        self.xai_analyzer = None
        
        print(f"Pipeline initialized successfully with model: {os.path.basename(yolo_model_path)}")
        if enable_xai:
            print("XAI analysis enabled")
    
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
        print(f"YOLO Model: {os.path.basename(self.yolo_detector.model_path)}")
        print(f"Scale: {self.analyzer.meters_per_pixel} meters/pixel")
        print(f"XAI Analysis: {'Enabled' if self.enable_xai else 'Disabled'}")
        
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
        
        # Step 2: XAI Analysis (if enabled and craters were detected)
        if self.enable_xai and len(yolo_output['boxes']) > 0:
            print(f"\n🔍 Step 2: Running XAI analysis...")
            try:
                self.xai_analyzer = YOLOXAIAnalyzer(
                    self.yolo_detector.model_path, 
                    image_path
                )
                xai_results = self.xai_analyzer.run_xai_analysis(run_dir)
                print("XAI analysis completed successfully!")
            except Exception as e:
                print(f"XAI analysis failed: {e}")
                print("Continuing with normal pipeline...")
        elif self.enable_xai and len(yolo_output['boxes']) == 0:
            print(f"\n⚠️  Skipping XAI analysis - no craters detected")
        
        # Step 3: SAM Segmentation
        step_num = 3 if self.enable_xai else 2
        print(f"\n🎯 Step {step_num}: Running SAM segmentation...")
        sam_output_path = os.path.join(run_dir, f"0{step_num}_sam_segmentation.png")
        sam_output = self.sam_segmenter.segment_craters(
            yolo_output,
            save_path=sam_output_path
        )
        
        # Step 4: Scientific Analysis
        step_num += 1
        print(f"\n📊 Step {step_num}: Performing scientific analysis...")
        analysis_dir = os.path.join(run_dir, f"0{step_num}_analysis")
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
            f.write(f"YOLO Model: {os.path.basename(self.yolo_detector.model_path)}\n")
            f.write(f"Scale Factor: {self.analyzer.meters_per_pixel} meters/pixel\n")
            f.write(f"XAI Analysis: {'Enabled' if self.enable_xai else 'Disabled'}\n\n")
            
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
                if self.enable_xai:
                    f.write("- 05_xai_analysis/: XAI explainability analysis\n")
                    f.write("  - gradcam_analysis.png: GradCAM attention maps\n")
                    f.write("  - lime_analysis.png: LIME interpretability (if available)\n")
                    f.write("  - occlusion_analysis.png: Occlusion sensitivity maps\n")
                    f.write("  - feature_visualization.png: Feature analysis\n")
                    f.write("  - xai_summary.png: Combined XAI visualization\n")
                    f.write("- 03_sam_segmentation.png: SAM segmentation overlay\n")
                    f.write("- 04_analysis/: Scientific analysis plots\n")
                else:
                    f.write("- 03_sam_segmentation.png: SAM segmentation overlay\n")
                    f.write("- 04_analysis/: Scientific analysis plots\n")
                f.write("- crater_data.csv: Detailed crater measurements\n")
                f.write("- pipeline_summary.txt: This summary file\n")
            else:
                f.write("No craters detected in the input image.\n")

def get_available_models(models_dir="Models"):
    """Get list of available YOLO models"""
    if not os.path.exists(models_dir):
        return []
    
    model_files = glob.glob(os.path.join(models_dir, "*.pt"))
    return [os.path.basename(f) for f in model_files]

def find_sam_model(models_dir="Models"):
    """Find SAM model in Models directory"""
    possible_sam_names = [
        "sam_vit_b.pth", "sam_vit_l.pth", "sam_vit_h.pth", 
        "sam.pth", "sam_vit_b_01ec64.pth"
    ]
    
    for sam_name in possible_sam_names:
        sam_path = os.path.join(models_dir, sam_name)
        if os.path.exists(sam_path):
            return sam_path
    
    # Check in current directory as fallback
    for sam_name in possible_sam_names:
        if os.path.exists(sam_name):
            return sam_name
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Crater Detection Pipeline with XAI')
    parser.add_argument('--model', 
                       help='YOLO model name (e.g., moon.pt, mars.pt, mercury.pt)')
    parser.add_argument('--image', default='test.png', 
                       help='Path to input image')
    parser.add_argument('--output_dir', default='output', 
                       help='Output directory for results')
    parser.add_argument('--scale', type=float, default=1.0, 
                       help='Meters per pixel scale factor')
    parser.add_argument('--conf', type=float, default=0.25, 
                       help='Confidence threshold for YOLO detection')
    parser.add_argument('--no_xai', action='store_true',
                       help='Disable XAI analysis (faster execution)')
    parser.add_argument('--list_models', action='store_true',
                       help='List available models and exit')
    
    args = parser.parse_args()
    
    # List available models if requested
    if args.list_models:
        available_models = get_available_models()
        print("Available YOLO models in Models/ directory:")
        if available_models:
            for i, model in enumerate(available_models, 1):
                print(f"  {i}. {model}")
        else:
            print("  No models found in Models/ directory")
        return
    
    # Get available models
    available_models = get_available_models()
    
    if not available_models:
        print("Error: No YOLO models found in Models/ directory!")
        print("Please place your .pt model files in the Models/ directory")
        print("Expected models: moon.pt, mars.pt, mercury.pt, best.pt")
        return
    
    # Select model
    if args.model:
        if args.model not in available_models:
            print(f"Error: Model '{args.model}' not found in Models/ directory!")
            print("Available models:")
            for model in available_models:
                print(f"  - {model}")
            return
        selected_model = args.model
    else:
        # Interactive model selection
        print("Available YOLO models:")
        for i, model in enumerate(available_models, 1):
            print(f"  {i}. {model}")
        
        while True:
            try:
                choice = input(f"\nSelect model (1-{len(available_models)}): ").strip()
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(available_models):
                    selected_model = available_models[choice_idx]
                    break
                else:
                    print("Invalid choice. Please try again.")
            except (ValueError, KeyboardInterrupt):
                print("\nExiting...")
                return
    
    # Construct model paths
    yolo_model_path = os.path.join("Models", selected_model)
    sam_model_path = find_sam_model()
    
    if not sam_model_path:
        print("Error: SAM model not found!")
        print("Please place a SAM model file in the Models/ directory")
        print("Expected names: sam_vit_b.pth, sam_vit_l.pth, sam_vit_h.pth, sam.pth")
        return
    
    # Validate input image
    if not os.path.exists(args.image):
        print(f"Error: Input image '{args.image}' not found!")
        return
    
    # XAI settings
    enable_xai = not args.no_xai
    
    try:
        print(f"\nSelected YOLO model: {selected_model}")
        print(f"SAM model: {os.path.basename(sam_model_path)}")
        print(f"Scale: {args.scale} meters/pixel")
        print(f"XAI Analysis: {'Enabled' if enable_xai else 'Disabled'}")
        
        if enable_xai:
            print("\nXAI Features:")
            print("- GradCAM attention mapping")
            print("- Occlusion sensitivity analysis")
            print("- Feature visualization")
            try:
                import lime
                print("- LIME local interpretability (available)")
            except ImportError:
                print("- LIME local interpretability (not available - install with: pip install lime)")
        
        # Initialize and run pipeline
        pipeline = CraterDetectionPipeline(
            yolo_model_path=yolo_model_path,
            sam_model_path=sam_model_path,
            meters_per_pixel=args.scale,
            enable_xai=enable_xai
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