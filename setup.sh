#!/bin/bash

# ML-HEA Screening Setup Script
# Automates environment setup and dependency installation

set -e  # Exit on error

echo "=========================================="
echo "ML-HEA Screening - Setup Script"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${YELLOW}[1/5] Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $PYTHON_VERSION"

# Verify Python 3.9+
if ! python3 -c 'import sys; exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
    echo -e "${RED}Error: Python 3.9+ required (found $PYTHON_VERSION)${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python version OK${NC}"
echo ""

# Create virtual environment
echo -e "${YELLOW}[2/5] Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi
echo ""

# Activate virtual environment
echo -e "${YELLOW}[3/5] Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Install dependencies
echo -e "${YELLOW}[4/5] Installing dependencies...${NC}"
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dependencies installed successfully${NC}"
else
    echo -e "${RED}✗ Failed to install dependencies${NC}"
    exit 1
fi
echo ""

# Verify installation
echo -e "${YELLOW}[5/5] Verifying installation...${NC}"

# Check imports
python3 << PYEOF
import sys
failed = []

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except ImportError:
    failed.append("torch")

try:
    import transformers
    print(f"✓ Transformers {transformers.__version__}")
except ImportError:
    failed.append("transformers")

try:
    import pandas
    print(f"✓ Pandas {pandas.__version__}")
except ImportError:
    failed.append("pandas")

try:
    import sklearn
    print(f"✓ scikit-learn {sklearn.__version__}")
except ImportError:
    failed.append("scikit-learn")

try:
    import numpy
    print(f"✓ NumPy {numpy.__version__}")
except ImportError:
    failed.append("numpy")

if failed:
    print(f"\n✗ Missing packages: {', '.join(failed)}")
    sys.exit(1)
else:
    print("\n✓ All core packages installed")
PYEOF

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Verification failed${NC}"
    exit 1
fi
echo ""

# Check data files
echo -e "${YELLOW}Checking data files...${NC}"
if [ -f "data/dataset.csv" ]; then
    echo -e "${GREEN}✓ dataset.csv found${NC}"
else
    echo -e "${RED}✗ dataset.csv not found${NC}"
fi

if [ -f "data/dataset_for_hardness_regression_language_model.csv" ]; then
    echo -e "${GREEN}✓ dataset_for_hardness_regression_language_model.csv found${NC}"
else
    echo -e "${RED}✗ dataset_for_hardness_regression_language_model.csv not found${NC}"
fi

if [ -f "data/mlm_pretraining_dataset.csv" ]; then
    echo -e "${GREEN}✓ mlm_pretraining_dataset.csv found${NC}"
else
    echo -e "${YELLOW}⚠ mlm_pretraining_dataset.csv not found (optional for basic usage)${NC}"
fi
echo ""

# Test GPU availability
echo -e "${YELLOW}Checking GPU availability...${NC}"
python3 << PYEOF
import torch
if torch.cuda.is_available():
    print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  Device Count: {torch.cuda.device_count()}")
else:
    print("⚠ No GPU detected - will use CPU (training will be slower)")
PYEOF
echo ""

# Final message
echo "=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   ${YELLOW}source venv/bin/activate${NC}"
echo ""
echo "2. Try training a classical model:"
echo "   ${YELLOW}cd src && python train_for_hardness.py${NC}"
echo ""
echo "3. Train the best language model:"
echo "   ${YELLOW}python train_hardness_regressor_scheduler_skip.py${NC}"
echo ""
echo "4. Screen virtual candidates:"
echo "   ${YELLOW}python bo_language_model.py${NC}"
echo ""
echo "For more information, see README.md"
echo ""
