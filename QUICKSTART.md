# Quick Start Guide - ML-HEA Screening

Get up and running in 5 minutes!

## 1. Automated Setup (Recommended)

```bash
cd /Users/rahulbouri/Developer/ml-hea-screening
./setup.sh
```

This will:
- ✓ Create a Python virtual environment
- ✓ Install all dependencies
- ✓ Verify your installation
- ✓ Check GPU availability

**Time: ~2-5 minutes**

## 2. Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify setup
python3 test_setup.py
```

## 3. Run Your First Training

```bash
source venv/bin/activate
cd src

# Train a lightweight classical model (fastest)
python train_for_hardness.py
```

**Expected output:**
```
Train shape: (332, 12), Test shape: (83, 12)
Lasso - R²: 0.389, MAE: 130.9 HV, RMSE: 26956 HV
Gradient Boosting - R²: 0.748, MAE: 78.0 HV, RMSE: 11141 HV
LightGBM - R²: 0.773, MAE: 73.9 HV, RMSE: 10038 HV
```

## 4. Train the Best Model (Language Model)

```bash
# This trains the best-performing model
python train_hardness_regressor_scheduler_skip.py
```

**Expected performance:**
- R² = 0.762 (test set)
- MAE = 83.4 HV
- Training time: ~45 minutes (GPU) / ~2 hours (CPU)

## 5. Screen Virtual Candidates

```bash
# Find high-hardness alloy compositions
python bo_language_model.py
```

**Output:** Top predicted candidates for experimental validation

## Troubleshooting

### "module not found" error
```bash
pip install -r requirements.txt --upgrade
```

### GPU out of memory
Edit `configs/config.py` and reduce `BATCH_SIZE` or set `DEVICE = "cpu"`

### MatSci BERT download fails
The model will auto-download from Hugging Face Hub on first run. Ensure you have internet connection and ~5 GB free disk space.

## File Locations

- **Data**: `data/`
- **Models**: `models/saved_models/`
- **Notebooks**: `notebooks/`
- **Source code**: `src/`
- **Configuration**: `configs/config.py`

## Next Steps

1. Read `README.md` for full documentation
2. Explore `notebooks/` for detailed analysis
3. Modify `configs/config.py` for custom settings
4. Run `python test_setup.py` to verify dependencies

## Need Help?

- Check README.md section "Troubleshooting"
- Review notebooks for examples
- Open an issue on GitHub

---
Last updated: May 2025
