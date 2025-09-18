#!/usr/bin/env python3
"""
Simple setup script for Crater Detection Pipeline with XAI
Installs requirements and checks for required models
"""

import os
import sys
import subprocess

def install_requirements():
    """Install required packages"""
    print("🚀 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def install_xai_optional():
    """Install optional XAI packages"""
    print("\n🔬 Installing optional XAI packages...")
    xai_packages = [
        "lime==0.2.0.1",
        "shap>=0.41.0", 
        "grad-cam>=1.4.6"
    ]
    
    success = True
    for package in xai_packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Failed to install {package}: {e}")
            success = False
    
    return success

def check_models():
    """Check if required models exist in Models folder"""
    models_dir = "Models"
    
    print(f"\n🔍 Checking for models in {models_dir}/ directory...")
    
    if not os.path.exists(models_dir):
        print(f"❌ {models_dir}/ directory not found!")
        print(f"   Please create {models_dir}/ directory and add your model files")
        return False
    
    # Required YOLO models
    required_yolo_models = ["moon.pt", "mars.pt", "mercury.pt", "best.pt"]
    found_yolo_models = []
    
    print("\n📡 YOLO Models:")
    for model in required_yolo_models:
        model_path = os.path.join(models_dir, model)
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024*1024)
            found_yolo_models.append(model)
            print(f"   ✅ {model} ({size_mb:.1f} MB)")
        else:
            print(f"   ❌ {model} - Not found")
    
    # Check for any additional YOLO models
    all_pt_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
    additional_models = [f for f in all_pt_files if f not in required_yolo_models]
    
    if additional_models:
        print("\n📡 Additional YOLO Models found:")
        for model in additional_models:
            model_path = os.path.join(models_dir, model)
            size_mb = os.path.getsize(model_path) / (1024*1024)
            print(f"   ✅ {model} ({size_mb:.1f} MB)")
    
    # Check for SAM model
    sam_models = [
        "sam_vit_b.pth", "sam_vit_l.pth", "sam_vit_h.pth", 
        "sam.pth", "sam_vit_b_01ec64.pth"
    ]
    
    found_sam_models = []
    print("\n🎯 SAM Models:")
    
    for sam_model in sam_models:
        model_path = os.path.join(models_dir, sam_model)
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024*1024)
            found_sam_models.append(sam_model)
            print(f"   ✅ {sam_model} ({size_mb:.1f} MB)")
    
    if not found_sam_models:
        print("   ❌ No SAM models found")
        print("   Expected files in Models/ directory:")
        for model in sam_models:
            print(f"     - {model}")
        print("\n   To download SAM model:")
        print("   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O Models/sam_vit_b.pth")
    
    # Summary
    total_yolo_found = len(found_yolo_models) + len(additional_models)
    total_sam_found = len(found_sam_models)
    
    print(f"\n📊 Summary:")
    print(f"   YOLO models found: {total_yolo_found}")
    print(f"   SAM models found: {total_sam_found}")
    
    if total_yolo_found > 0 and total_sam_found > 0:
        print("   ✅ Minimum requirements met!")
        return True
    else:
        print("   ❌ Missing required models")
        if total_yolo_found == 0:
            print("   Please add at least one .pt YOLO model to Models/ directory")
        if total_sam_found == 0:
            print("   Please add a SAM model to Models/ directory")
        return False

def test_imports():
    """Test if required packages can be imported"""
    print("\n🧪 Testing package imports...")
    
    required_packages = [
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("matplotlib.pyplot", "Matplotlib"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("ultralytics", "Ultralytics YOLO"),
        ("flask", "Flask"),
        ("werkzeug", "Werkzeug"),
    ]
    
    failed_imports = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name}")
            failed_imports.append(name)
    
    # Test SAM separately
    try:
        import segment_anything
        print("   ✅ Segment Anything Model (SAM)")
    except ImportError:
        print("   ❌ Segment Anything Model (SAM)")
        failed_imports.append("SAM")
    
    # Test XAI packages (optional)
    print("\n🔬 Testing XAI packages (optional):")
    xai_packages = [
        ("lime", "LIME"),
        ("shap", "SHAP"),
        ("pytorch_grad_cam", "GradCAM")
    ]
    
    xai_available = []
    for package, name in xai_packages:
        try:
            __import__(package)
            print(f"   ✅ {name}")
            xai_available.append(name)
        except ImportError:
            print(f"   ⚠️  {name} (optional for XAI)")
    
    if failed_imports:
        print(f"\n⚠️  Missing core packages: {', '.join(failed_imports)}")
        print("   Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All required packages available!")
        if len(xai_available) < len(xai_packages):
            print(f"📊 XAI packages available: {len(xai_available)}/{len(xai_packages)}")
            print("   Some XAI features may be disabled")
        else:
            print("🎉 All XAI packages available!")
        return True

def main():
    """Main setup function"""
    print("🌙 Crater Detection Pipeline with XAI - Setup")
    print("=" * 60)
    
    # Install requirements
    print("\n1️⃣  Installing Python Requirements...")
    install_choice = input("Install/update Python requirements? (y/n): ").lower().strip()
    
    imports_ok = True
    if install_choice in ['y', 'yes']:
        imports_ok = install_requirements()
        
        # Ask about XAI packages
        xai_choice = input("Install optional XAI packages (LIME, SHAP, GradCAM)? (y/n): ").lower().strip()
        if xai_choice in ['y', 'yes']:
            install_xai_optional()
        
        print("\n" + "-" * 30)
    
    # Test imports
    print("2️⃣  Testing Package Imports...")
    if install_choice not in ['y', 'yes']:
        imports_ok = test_imports()
    else:
        imports_ok = test_imports() and imports_ok
    
    print("\n" + "-" * 30)
    
    # Check models
    print("3️⃣  Checking Model Files...")
    models_ok = check_models()
    
    # Final summary
    print("\n" + "=" * 60)
    
    if imports_ok and models_ok:
        print("🎉 Setup completed successfully!")
        print("\n🚀 You can now run:")
        print("   Command Line (with XAI): python main.py --model moon.pt --image test.png")
        print("   Command Line (no XAI):   python main.py --model moon.pt --image test.png --no_xai")
        print("   Web Interface:            python app.py")
        print("   List models:              python main.py --list_models")
        
        print("\n🔬 XAI Features Available:")
        print("   - GradCAM attention mapping (always available)")
        print("   - Occlusion sensitivity analysis (always available)")
        print("   - Feature visualization (always available)")
        
        try:
            import lime
            print("   - LIME interpretability ✅")
        except ImportError:
            print("   - LIME interpretability ❌ (install with: pip install lime)")
            
        try:
            import shap
            print("   - SHAP analysis ✅")
        except ImportError:
            print("   - SHAP analysis ❌ (install with: pip install shap)")
            
    elif imports_ok:
        print("⚠️  Setup partially complete - missing models")
        print("   Please add required model files to Models/ directory")
    else:
        print("❌ Setup incomplete")
        if not imports_ok:
            print("   Install requirements: pip install -r requirements.txt")
        if not models_ok:
            print("   Add model files to Models/ directory")
    
    print("\n📚 For help:")
    print("   python main.py --help")
    print("   Check README.md for detailed instructions")

if __name__ == "__main__":
    main()