#!/usr/bin/env python3
"""
Quick test to verify the setup is correct
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 60)
print("ML-HEA Screening - Setup Verification")
print("=" * 60)
print()

# Test 1: Check directory structure
print("[1] Checking directory structure...")
required_dirs = [
    "src", "data", "models", "notebooks", "configs", "models/checkpoints"
]
missing_dirs = []
for d in required_dirs:
    p = Path(__file__).parent / d
    if p.exists():
        print(f"  ✓ {d}/")
    else:
        print(f"  ✗ {d}/ NOT FOUND")
        missing_dirs.append(d)

if missing_dirs:
    print(f"\n✗ Missing directories: {', '.join(missing_dirs)}")
    sys.exit(1)
print()

# Test 2: Check data files
print("[2] Checking data files...")
required_files = [
    "data/dataset.csv",
    "data/dataset_for_hardness_regression_language_model.csv",
]
missing_files = []
for f in required_files:
    p = Path(__file__).parent / f
    if p.exists():
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  ✓ {f} ({size_mb:.1f} MB)")
    else:
        print(f"  ✗ {f} NOT FOUND")
        missing_files.append(f)

if missing_files:
    print(f"\n⚠ Missing data files: {', '.join(missing_files)}")
    print("  (Some files are optional for basic testing)")
print()

# Test 3: Check Python packages
print("[3] Checking Python packages...")
packages = {
    'torch': 'PyTorch',
    'transformers': 'Transformers',
    'pandas': 'Pandas',
    'numpy': 'NumPy',
    'sklearn': 'scikit-learn',
    'scipy': 'SciPy',
}

failed_packages = []
for pkg, name in packages.items():
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f"  ✓ {name} ({version})")
    except ImportError:
        print(f"  ✗ {name} NOT INSTALLED")
        failed_packages.append(name)

if failed_packages:
    print(f"\n✗ Missing packages: {', '.join(failed_packages)}")
    print("  Run: pip install -r requirements.txt")
    sys.exit(1)
print()

# Test 4: Check custom modules
print("[4] Checking custom modules...")
try:
    from composition import parse_formula
    print("  ✓ composition.py")
except ImportError as e:
    print(f"  ✗ composition.py - {e}")

try:
    from configs.config import PROJECT_ROOT
    print(f"  ✓ configs/config.py (Project root: {PROJECT_ROOT})")
except ImportError as e:
    print(f"  ✗ configs/config.py - {e}")

print()

# Test 5: Check GPU availability
print("[5] Checking GPU availability...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"    CUDA Version: {torch.version.cuda}")
        print(f"    Device Count: {torch.cuda.device_count()}")
    else:
        print("  ⚠ No GPU detected (will use CPU)")
except Exception as e:
    print(f"  ⚠ Could not check GPU: {e}")

print()
print("=" * 60)
print("✓ Setup verification complete!")
print("=" * 60)
print()
print("Next steps:")
print("  1. cd src")
print("  2. python train_for_hardness.py        (classical models)")
print("  3. python train_hardness_regressor_scheduler_skip.py  (best model)")
print("  4. python bo_language_model.py         (virtual screening)")
print()

