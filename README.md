# ML-Driven Prediction and Screening of High-Entropy Alloys for High Hardness

**From Data to Alloys: Predicting and Screening High-Entropy Alloys for High Hardness Using Machine Learning**

This repository implements a machine learning framework for predicting Vickers hardness (HV) of high-entropy alloys (HEAs) and screening virtual alloy candidates for high-hardness compositions.

## Overview

High-entropy alloys (HEAs) are promising materials for extreme environments, but their vast compositional design space makes experimental exploration challenging. This project accelerates HEA discovery using:

- **Classical ML models**: LightGBM, Gradient Boosting Regressor, and Lasso Regression
- **Transformer-based models**: Custom transformer encoder with attention mechanisms
- **Language models**: Fine-tuned MatSci BERT for predicting hardness from elemental descriptors
- **Bayesian Optimization**: Virtual screening of 9.12M candidate compositions to identify high-hardness alloys

### Key Results

- **Best Model**: Language model with skip connections (R² = 0.762, MAE = 83.4 HV)
- **Top Predicted Candidates**:
  - Al57.8Co13.2Fe14.0Cr15.0 (626.8 HV)
  - Al61.4Ni13.4Mn13.6Zn11.5 (570.8 HV)
  - Al61.4Ni12.8Mn14.4Zn11.5 (571.5 HV)

## Quick Start (5 minutes)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train a Classical Model (LightGBM)

```bash
cd src
python train_for_hardness.py
```

Expected output:
- Train/test metrics (R², MAE, RMSE)
- Model saved to `../models/checkpoints/`

### 3. Train Language Model with Fine-tuning

```bash
python train_hardness_regressor_scheduler_skip.py
```

This trains the best-performing model:
- Loads pre-trained MatSci BERT
- Fine-tunes on HEA-specific vocabulary (MLM)
- Adds skip connections for improved hardness prediction

### 4. Screen Virtual Candidates

```bash
python bo_language_model.py
```

This runs Bayesian Optimization to find promising alloy compositions from 9.12M candidates.

## Installation

### Requirements
- Python 3.9+
- GPU recommended (CUDA 11.0+) for faster training
- ~50 GB disk space for pre-trained models

### Step-by-Step Setup

```bash
# Clone repository
git clone <repository-url>
cd ml-hea-screening

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained MatSci BERT weights (auto-downloaded on first use)
# Or manually: models/matscibert_weights will be downloaded from Hugging Face Hub

# Verify installation
python -c "import torch; import transformers; print('Setup successful!')"
```

### Using the Setup Script

```bash
chmod +x setup.sh
./setup.sh
```

This will:
1. Create a Python virtual environment
2. Install all dependencies
3. Download pre-trained model weights
4. Verify the installation

## Project Structure

```
ml-hea-screening/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup.sh                           # Quick installation script
│
├── data/
│   ├── dataset.csv                    # Original HEA dataset (415 unique compositions)
│   ├── dataset_for_hardness_regression_language_model.csv  # Processed features + targets
│   └── mlm_pretraining_dataset.csv   # 150K samples for MLM pre-training
│
├── src/
│   ├── composition.py                 # Feature generation from alloy formulas
│   ├── convesrion_weightpct_atpct.py # Weight % to atomic % conversion
│   ├── fractional.py                 # Composition parsing utilities
│   ├── utils.py                      # Bootstrap & Expected Improvement utilities
│   │
│   ├── train_for_hardness.py         # Classical ML baseline training
│   ├── train_hardness_regressor_scheduler.py  # Transformer with frozen encoder
│   ├── train_hardness_regressor_scheduler_skip.py  # Transformer + skip connections (BEST)
│   ├── bo_language_model.py          # Bayesian Optimization for screening
│   ├── analysis_self_attention.py    # Self-attention visualization
│   └── model_shap_explainer.py       # SHAP feature importance analysis
│
├── models/
│   ├── saved_models/                 # Trained model checkpoints
│   ├── matscibert_weights/           # Pre-trained MatSci BERT (original)
│   ├── finetuned_matscibert_weights/ # MLM fine-tuned MatSci BERT
│   └── checkpoints/                  # Training checkpoints & scalers
│
├── notebooks/
│   ├── Custom Data MLM Encoder Pretraining.ipynb  # MLM fine-tuning workflow
│   ├── Z_my_exp.ipynb                # Classical models training & evaluation
│   └── Z_my_exp_lgbm.ipynb           # LightGBM hyperparameter search
│
└── configs/
    └── config.py                      # Configuration (paths, hyperparameters)
```

## Workflow

### Stage 1: Data Preparation

1. **Load Dataset**: `dataset.csv` contains 415 HEA compositions with Vickers hardness values
2. **Feature Engineering**: Generate 14 elemental descriptors (atomic radius, electronegativity, etc.)
3. **Feature Scaling**: StandardScaler normalization

```
Input: HEA formula (e.g., Al68Ni16Zn16)
         ↓
Output: 14 scalar features per composition
         + Language model text representation
```

### Stage 2: Model Training

#### Classical Models (Fast Baseline)
```bash
python src/train_for_hardness.py
```
- **Lasso**: R² = 0.389 (sparse, interpretable)
- **Gradient Boosting**: R² = 0.748
- **LightGBM**: R² = 0.773 (best classical model)

#### Transformer Models
```bash
python src/train_hardness_regressor_scheduler.py
```
- Three pooling strategies: Mean, Attention, CLS token
- **Best**: Mean pooling (R² = 0.742)

#### Language Models (BEST OVERALL)
```bash
python src/train_hardness_regressor_scheduler_skip.py
```

Three strategies in order:
1. **Frozen Encoder** (R² = 0.586): MatSci BERT frozen, only regression head trained
2. **MLM Fine-tuning** (R² = 0.601): Fine-tune on 150K HEA samples with 15% masking
3. **Skip Connections** (R² = 0.762): ✅ Add raw descriptors to regression head

### Stage 3: Virtual Screening

```bash
python src/bo_language_model.py
```

Process:
1. Generate 9.12M virtual quaternary alloy compositions
2. Use language model as surrogate function
3. Estimate uncertainty via Bayesian neural network
4. Run Bayesian Optimization (5 init points + 45 iterations)
5. Select candidates balancing exploitation (high hardness) and exploration (high uncertainty)

**Output**: Top 5-10 predicted high-hardness compositions with Expected Improvement scores

### Stage 4: Analysis & Visualization

```bash
python src/analysis_self_attention.py     # Visualize important features
python src/model_shap_explainer.py        # SHAP feature importance
```

## Jupyter Notebooks

This repository includes **3 fully-documented Jupyter notebooks** with **137 markdown cells** explaining every step.

### Overview

| Notebook | Purpose | Cells | Learning Level |
|----------|---------|-------|-----------------|
| **Z_my_exp.ipynb** | Classical ML models comparison | 161 (77 MD + 84 code) | Beginner-friendly |
| **Z_my_exp_lgbm.ipynb** | LightGBM optimization deep-dive | 80 (41 MD + 39 code) | Intermediate |
| **Custom Data MLM.ipynb** | MatSci BERT fine-tuning | 31 (19 MD + 12 code) | Advanced |

### 1. Z_my_exp.ipynb - Classical ML Models Training

**What You'll Learn:**
- Complete ML pipeline from data loading to evaluation
- How to train Lasso, Gradient Boosting, and LightGBM
- Feature engineering from chemical formulas
- Comparison of 3 classical models
- How to interpret model metrics (R², MAE, RMSE)

**Sections:**
1. **Data Loading** - Load 415 HEA compositions
2. **Feature Engineering** - Generate 14 elemental descriptors
3. **Train-Test Split** - Prepare data for modeling
4. **Lasso Regression** - Sparse linear model (R² = 0.389)
5. **Gradient Boosting** - Sequential ensembles (R² = 0.748)
6. **LightGBM** - Best classical model (R² = 0.773) ✓
7. **Feature Importance** - Which properties drive hardness?
8. **Model Evaluation** - Compare performance metrics

**Runtime:** ~5-10 minutes on CPU / ~2-5 minutes on GPU

**When to use:**
- Start here if new to the project
- Understand baseline ML models
- Learn feature engineering
- See model comparison

**markdown cells include:**
- Clear explanations of each algorithm
- Expected outputs with numeric ranges
- Parameter explanations
- Physical interpretation of features
- Results comparison table

### 2. Z_my_exp_lgbm.ipynb - LightGBM Hyperparameter Optimization

**What You'll Learn:**
- Systematic hyperparameter tuning with GridSearchCV
- Why each hyperparameter matters
- How to optimize machine learning models
- 5-fold cross-validation strategy
- Best configuration for your dataset

**Sections:**
1. **Data Preparation** - Same as Z_my_exp.ipynb
2. **GridSearchCV Setup** - Define hyperparameter grid
3. **Hyperparameter Search** - Test 243 combinations
4. **Best Configuration** - Results from optimization
5. **Model Evaluation** - Performance on test set

**Best Hyperparameters Found:**
```
n_estimators: 100      # Number of trees
max_depth: 3          # Tree complexity
num_leaves: 31        # Leaves per tree
learning_rate: 0.1    # Update step size
subsample: 0.8        # Samples per iteration
colsample_bytree: 0.8 # Features per iteration
```

**Results:**
- R² = 0.773
- MAE = 73.9 HV (average prediction error)
- RMSE = 10,038 HV

**Runtime:** ~30 minutes on CPU / ~10 minutes on GPU

**When to use:**
- Understand hyperparameter tuning
- Learn GridSearchCV methodology
- Optimize for your own dataset
- Deep dive into best classical model

**markdown cells include:**
- GridSearchCV explanation (why it matters)
- Hyperparameter effects on performance
- Cross-validation strategy
- Best configuration details
- Performance metrics interpretation

### 3. Custom Data MLM Encoder Pretraining.ipynb - Language Model Fine-tuning

**What You'll Learn:**
- Transfer learning with pre-trained models
- Masked Language Modeling (MLM) concept
- Fine-tuning BERT for domain-specific tasks
- Tokenization and text processing
- Advanced NLP techniques

**Key Concepts:**
- **MLM**: Mask 15% of tokens, predict from context
- **Transfer Learning**: Start with pre-trained MatSci BERT
- **Fine-tuning**: Adapt to HEA compositions
- **Unlabeled Data**: Uses 150K samples without hardness labels

**Sections:**
1. **Setup** - Install required packages
2. **Load Data** - Load 150K HEA composition samples
3. **Tokenization** - Convert text to numeric tokens
4. **Create MLM Dataset** - Prepare data collator
5. **MLM Training** - Fine-tune BERT (40 epochs)
6. **Save Weights** - Store fine-tuned model

**Expected Results:**
- MLM Loss: Decreases 0.5 → 0.3
- Training time: 2-3 hours on GPU
- Improves downstream R²: 0.601 (vs 0.586 frozen)

**Runtime:** ~2-3 hours on GPU / ~8-10 hours on CPU

**When to use:**
- Learn advanced transfer learning
- Understand language models for materials
- Fine-tune BERT for other datasets
- Connect to hardness prediction task

**markdown cells include:**
- MLM concept explanation (with examples)
- Why transfer learning matters
- Tokenization process step-by-step
- Training dynamics and convergence
- Connection to hardness prediction

### How to Run the Notebooks

**Option 1: Using Jupyter (Recommended for Learning)**
```bash
# Activate virtual environment
source venv/bin/activate

# Start Jupyter
jupyter notebook notebooks/

# Open in browser, click Z_my_exp.ipynb
# Read markdown → Run code → Review output
```

**Option 2: Using JupyterLab (More Modern Interface)**
```bash
pip install jupyterlab
jupyter lab notebooks/
```

**Option 3: Using VS Code**
```bash
# Install Jupyter extension in VS Code
# Open notebook files directly
# Integrated markdown + code execution
```

### Notebook Features

✅ **50% Markdown Coverage**
- Nearly 1 markdown cell for every code cell
- Explains the WHY, not just the WHAT

✅ **Clear Section Headers**
- Easy navigation with numbered sections
- Consistent formatting across notebooks

✅ **Expected Outputs**
- Metric ranges provided (R², MAE, RMSE)
- Performance benchmarks included
- Example outputs shown

✅ **Physical Interpretations**
- Why features matter in materials science
- Domain knowledge context
- Connection to alloy behavior

✅ **Parameter Explanations**
- What each hyperparameter does
- Why we chose certain values
- How to modify for your use case

✅ **Step-by-Step Algorithms**
- How Lasso works (sparse penalty)
- Gradient Boosting process (sequential trees)
- LightGBM advantages (histogram-based)
- MLM concept (masked token prediction)

### Relationship to Python Scripts

| Notebook | Equivalent Script | Difference |
|----------|-------------------|-----------|
| Z_my_exp.ipynb | `train_for_hardness.py` | Notebooks include visualization + explanation |
| Z_my_exp_lgbm.ipynb | `train_for_hardness.py` | Focused on LightGBM optimization only |
| Custom MLM.ipynb | N/A (preprocessing) | Required before `train_hardness_regressor_scheduler_skip.py` |

**Scripts** = Production-ready, scriptable code  
**Notebooks** = Educational, interactive exploration

### Tips for Using the Notebooks

1. **Read markdown BEFORE running code**
   - Markdown sets context and expectations
   - Code implements the ideas

2. **Run cells sequentially**
   - Each section depends on previous cells
   - Don't skip sections

3. **Examine outputs carefully**
   - Compare to expected values in markdown
   - Debug if results differ significantly

4. **Modify and experiment**
   - Change hyperparameters to see effects
   - Notebooks are for learning and exploration

5. **Save outputs**
   - Models, scalers, feature rankings
   - Reuse for downstream tasks

### Further Reading

For detailed explanations of all notebook sections, see:
- **NOTEBOOK_GUIDE.md** - Complete section-by-section walkthrough
- **MARKDOWN_ENHANCEMENT_SUMMARY.txt** - What was added and why

## Data Files Description

### dataset.csv
- **Source**: Borg et al. (2020) - MPEA dataset
- **Size**: 415 unique HEA compositions
- **Columns**:
  - `FORMULA`: Alloy composition (e.g., Al68Ni16Zn16)
  - `PROPERTY: HV`: Vickers hardness (target variable)
  - `PROPERTY: Test temperature (°C)`: Testing temperature

### dataset_for_hardness_regression_language_model.csv
- **Processing**: Extracted from dataset.csv with:
  - `composition` or `formula`: Alloy formula
  - `concat_text`: String representation of descriptors (for BERT tokenization)
  - 11 elemental descriptor columns (avg_Atomic_Radius, avg_Pauling_Electronegativity, etc.)
  - `hardness`: Target variable (Vickers hardness)

### mlm_pretraining_dataset.csv
- **Size**: 150,000 HEA composition samples
- **Purpose**: Pre-training MatSci BERT with masked language modeling (15% masking probability)
- **Format**: Concatenated elemental descriptor strings for tokenization

## Model Weights

### MatSci BERT Models
- **Original**: Auto-downloaded from Hugging Face Hub (`m3rg-iitd/matscibert`)
- **Fine-tuned**: Located in `models/finetuned_matscibert_weights/`
  - Trained on 150K HEA samples
  - 15% token masking probability
  - 40 epochs, batch size 16, learning rate 5e-5

### Trained Regression Heads
Located in `models/saved_models/`:
- `lgbm_model.pkl`: Best classical model (LightGBM)
- `regressor_head.pt`: Language model regression head
- Other transformer checkpoints

## Key Features & Performance Metrics

### Feature Importance (from LightGBM & SHAP)
1. Valence Electron Count (avg)
2. Bond Strength (avg)
3. Atomic Radius (avg)
4. Cohesive Energy (avg)
5. Lattice Constant (avg)

### Model Comparison (Test Set)

| Model | R² | MAE (HV) | RMSE (HV) |
|-------|-----|----------|-----------|
| Lasso | 0.389 | 130.9 | 26,956 |
| Gradient Boosting | 0.748 | 78.0 | 11,141 |
| LightGBM | 0.773 | 73.9 | 10,038 |
| Transformer (Mean Pool) | 0.742 | 87.6 | 106.7 |
| Language Model (Frozen) | 0.586 | 114.3 | 153.64 |
| Language Model (MLM FT) | 0.601 | 106.9 | 137.16 |
| **Language Model (Skip)** | **0.762** | **83.4** | **102.67** |

## Reproducing Results

### 1. Train All Models

```bash
cd src
python train_for_hardness.py              # Classical models
python train_hardness_regressor_scheduler.py  # Transformer frozen
python train_hardness_regressor_scheduler_skip.py  # Language model (best)
```

### 2. Evaluate Feature Importance

```bash
python model_shap_explainer.py            # SHAP analysis
python analysis_self_attention.py         # Attention visualization
```

### 3. Screen Virtual Candidates

```bash
python bo_language_model.py               # Bayesian Optimization
# Output: Top candidates for experimental validation
```

## Configuration

Edit `configs/config.py` to customize:
- Model hyperparameters (learning rate, batch size, epochs)
- Training data paths
- Device selection (GPU/CPU)
- Random seeds for reproducibility

## System Requirements

### Minimum
- CPU: 4 cores
- RAM: 8 GB
- Storage: 50 GB (for pre-trained weights)
- Python 3.9+

### Recommended
- CPU: 8+ cores
- RAM: 16+ GB
- GPU: NVIDIA GPU with 8+ GB VRAM (CUDA 11.0+)
- Storage: 100 GB

### Training Time (GPU)
- Classical models: ~5 minutes
- Transformer models: ~30 minutes
- Language models: ~45 minutes
- Virtual screening: ~2 hours

## Troubleshooting

### GPU Out of Memory
- Reduce `BATCH_SIZE` in `configs/config.py`
- Use CPU: Set `DEVICE = "cpu"`

### Model Download Issues
- MatSci BERT is auto-downloaded from Hugging Face Hub
- If network issues, download manually: `m3rg-iitd/matscibert`

### Import Errors
Ensure all packages installed:
```bash
pip install -r requirements.txt --upgrade
```

### Data Not Found
Verify paths in `configs/config.py` match your local setup

## Citation

If you use this code or data, please cite:

```bibtex
@article{your_paper_2025,
  title={From Data to Alloys: Predicting and Screening High-Entropy Alloys for High Hardness Using Machine Learning},
  author={Rahul Bouri},
  journal={TBD},
  year={2025}
}
```

## References

1. Yeh, J. W., et al. (2004). Nanostructured high-entropy alloys with multiple principal elements. Advanced Engineering Materials, 6(5), 299-303.

2. Gludovatz, B., et al. (2014). A fracture-resistant high-entropy alloy for cryogenic applications. Science, 345(6201), 1153-1158.

3. Borg, C. K., et al. (2020). Expanded dataset of mechanical properties and observed phases of multi-principal element alloys. Scientific Data, 7(1), 430.

4. Chaudhari, A., et al. (2024). AlloyBERT: Alloy property prediction with large language models. Computational Materials Science, 244, 113256.

5. Kamnis, S., & Delibasis, K. (2025). High entropy alloy property predictions using a transformer-based language model. Scientific Reports, 15, 8157.

6. Kristiadi, A., et al. (2024). A sober look at LLMs for material discovery: Are they actually good for Bayesian optimization over molecules? arXiv preprint arXiv:2402.05015.

## License

This project is released under the MIT License. See LICENSE file for details.

## Contributing

Contributions are welcome! Please submit issues or pull requests for:
- Bug fixes
- Performance improvements
- Additional analysis notebooks
- Documentation improvements

## Contact

For questions or issues, please open a GitHub issue or contact the authors directly.

---

**Last Updated**: May 2025
