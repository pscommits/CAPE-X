#!/usr/bin/env python3
"""
Simple setup script for Crater Detection Pipeline
Just checks for models and creates directories without downloading
"""

import os
import sys
import subprocess
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ["output", "models", "test_images"]
    
    print("Creating directories...")
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created/verified directory: {directory}/")

def check_models():
    """Check if required models exist"""
    print("\nChecking for model files...")
    
    # Check for YOLO model
    yolo_locations = ["best.pt", "models/best.pt"]
    yolo_found = []
    for loc in yolo_locations:
        if os.path.exists(loc):
            size_mb = os.path.getsize(loc) / (1024*1024)
            yolo_found.append(f"{loc} ({size_mb:.1f} MB)")
    
    if yolo_found:
        print("✅ YOLO model(s) found:")
        for model in yolo_found:
            print(f"   - {model}")
    else:
        print("❌ YOLO model not found!")
        print("   Expected locations:")
        for loc in yolo_locations:
            print(f"   - {loc}")
        print("   Please place your trained YOLO model in one of these locations.")
    
    # Check for SAM model
    sam_locations = [
        "sam_vit_b.pth", "sam_vit_l.pth", "sam_vit_h.pth",
        "models/sam_vit_b.pth", "models/sam_vit_l.pth", "models/sam_vit_h.pth",
        "sam.pth", "models/sam.pth"
    ]
    
    sam_found = []
    for loc in sam_locations:
        if os.path.exists(loc):
            size_mb = os.path.getsize(loc) / (1024*1024)
            sam_found.append(f"{loc} ({size_mb:.1f} MB)")
    
    if sam_found:
        print("✅ SAM model(s) found:")
        for model in sam_found:
            print(f"   - {model}")
    else:
        print("❌ SAM model not found!")
        print("   Expected locations:")
        for loc in sam_locations:
            print(f"   - {loc}")
        print("\n   To get SAM models:")
        print("   1. Download manually from: https://github.com/facebookresearch/segment-anything")
        print("   2. Use wget:")
        print("      wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O models/sam_vit_b.pth")
        print("   3. If you already have a SAM model, rename it to sam.pth and place in models/ folder")
    
    return len(yolo_found) > 0 and len(sam_found) > 0

def check_test_image():
    """Check for test image"""
    test_images = ["test.png", "test.jpg", "test_images/test.png"]
    
    found_images = [img for img in test_images if os.path.exists(img)]
    
    if found_images:
        print("✅ Test image(s) found:")
        for img in found_images:
            print(f"   - {img}")
    else:
        print("⚠️  No test image found!")
        print("   Expected locations:")
        for img in test_images:
            print(f"   - {img}")
        print("   Place a test image to run the pipeline.")

def test_imports():
    """Test if required packages can be imported"""
    print("\nTesting package imports...")
    
    required_packages = [
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("matplotlib.pyplot", "Matplotlib"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("ultralytics", "Ultralytics YOLO"),
    ]
    
    failed_imports = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name}")
            failed_imports.append(name)
    
    # Test SAM separately (might not be installed yet)
    try:
        import segment_anything
        print("✅ Segment Anything Model (SAM)")
    except ImportError:
        print("❌ Segment Anything Model (SAM)")
        failed_imports.append("SAM")
    
    if failed_imports:
        print(f"\n⚠️  Missing packages: {', '.join(failed_imports)}")
        print("   Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All required packages available!")
        return True

def main():
    """Main setup function"""
    print("🚀 Crater Detection Pipeline - Simple Setup")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    # Check if user wants to install requirements
    install_choice = input("\nInstall/update Python requirements? (y/n): ").lower().strip()
    if install_choice in ['y', 'yes']:
        install_requirements()
    
    # Test imports
    imports_ok = test_imports()
    
    # Check models
    models_ok = check_models()
    
    # Check test image
    check_test_image()
    
    print("\n" + "=" * 50)
    
    if imports_ok and models_ok:
        print("🎉 Setup completed successfully!")
        print("\nYou can now run the pipeline:")
        print("   python main.py --image test.png --scale 5.0")
    elif imports_ok:
        print("⚠️  Setup partially complete - missing models")
        print("Please add the required model files and run setup again.")
    else:
        print("❌ Setup incomplete - please install requirements first")
        print("   pip install -r requirements.txt")
    
    print("\nFor help:")
    print("   python main.py --help")

if __name__ == "__main__":
    main()