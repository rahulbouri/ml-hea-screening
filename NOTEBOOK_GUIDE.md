# Enhanced Notebook Guide

Your notebooks now include comprehensive markdown cells explaining each step. This guide describes what was added.

## Overview

**77 markdown cells added** across 3 notebooks with explicit instructions for every major section.

---

## 1. Z_my_exp.ipynb - Classical ML Models

**Status**: 📓 161 total cells (77 markdown + 84 code)

### What This Notebook Covers
Training and evaluating three classical ML models for predicting HEA hardness:
- Lasso Regression (sparse linear)
- Gradient Boosting Regressor (sequential ensembles)
- LightGBM (histogram-based boosting)

### Section-by-Section Breakdown

#### 📌 **Introduction & Setup**
```
▶ Notebook Overview
  - Learning objectives
  - Expected outcomes
  - Runtime estimates

▶ Section 1: Environment Setup
  - Import libraries (pandas, numpy, sklearn, etc.)
  - Dependency verification
```

#### 📌 **Data Pipeline**
```
▶ Section 2: Data Loading
  - Load dataset.csv (415 HEA compositions)
  - Explore data shape and structure
  - Check for missing values
  
▶ Section 3: Feature Engineering
  - Parse chemical formulas
  - Generate 14 elemental descriptors
  - Atomic properties: Radius, Electronegativity, Valence electrons
  - Energetic properties: Cohesive energy, Bulk/Elastic modulus
  - Structural properties: Melting point, Lattice constant
  
▶ Section 4: Train-Test Split
  - 80% training (332 samples)
  - 20% testing (83 samples)
  - StandardScaler normalization
  - Why this split matters
```

#### 📌 **Model Training (3 Models)**
```
▶ Section 5a: Lasso Regression
  - Algorithm: Linear with L1 penalty
  - Hyperparameter tuning: 5-fold CV
  - Expected R² ≈ 0.389
  - When to use: Interpretability needed
  
▶ Section 5b: Gradient Boosting Regressor
  - Algorithm: Sequential tree ensemble
  - How boosting works (step-by-step)
  - Key hyperparameters explained
  - Expected R² ≈ 0.748
  
▶ Section 5c: LightGBM (Best Classical Model)
  - Why LightGBM is faster and better
  - Optimal hyperparameters from GridSearchCV
  - Expected R² = 0.773 ✓
  - Result: Best classical model
```

#### 📌 **Analysis & Evaluation**
```
▶ Section 6: Feature Importance
  - Which properties drive hardness?
  - Consistent across all models:
    1. Valence Electron Count
    2. Bond Strength
    3. Atomic Radius
    4. Cohesive Energy
    5. Lattice Constant
  
▶ Section 7: Model Evaluation
  - R² Score: Variance explained (0-1 range)
  - MAE: Average prediction error
  - RMSE: Error with penalty for outliers
  
  Results Table:
  | Model | R² | MAE (HV) | RMSE (HV) |
  |-------|------|----------|----------|
  | Lasso | 0.389 | 130.9 | 26,956 |
  | GBR | 0.748 | 78.0 | 11,141 |
  | LightGBM | 0.773 | 73.9 | 10,038 |
```

### Key Insights to Extract
1. Non-linear models (GBR, LightGBM) are much better than linear (Lasso)
2. LightGBM is fastest and most accurate
3. Valence electrons and bond strength are critical factors
4. 73.9 HV average error is acceptable for materials screening

### Expected Output
- Train/test metrics for 3 models
- Feature importance rankings
- Prediction vs actual plots
- Model comparison visualization

---

## 2. Z_my_exp_lgbm.ipynb - LightGBM Optimization

**Status**: 📓 80 total cells (41 markdown + 39 code)

### What This Notebook Covers
Deep dive into LightGBM hyperparameter optimization using GridSearchCV

### Section-by-Section Breakdown

#### 📌 **Notebook Overview**
```
▶ Best Hyperparameters Found
  n_estimators: 100         # Trees
  max_depth: 3             # Complexity
  num_leaves: 31           # Leaf nodes
  learning_rate: 0.1       # Update step size
  subsample: 0.8           # Samples per iteration
  colsample_bytree: 0.8    # Features per iteration
```

#### 📌 **Data Preparation**
```
▶ Section 1: Load Data and Features
  - Same dataset as Z_my_exp.ipynb
  - Same 12-14 elemental descriptors
  - Same train/test split
```

#### 📌 **Hyperparameter Optimization**
```
▶ Section 2: GridSearchCV
  What is GridSearchCV?
  - Tests all combinations of hyperparameters
  - Uses 5-fold cross-validation
  - Selects best configuration
  - Avoids overfitting
  
  Grid Tested:
  - 3 options for n_estimators
  - 3 options for max_depth
  - 3 options for num_leaves
  - 3 options for learning_rate
  - 3 options for subsample
  = 243 total models trained
  
  Why cross-validation?
  - Single train/test can be lucky/unlucky
  - 5 folds give robust estimate
  - More reliable model selection
```

#### 📌 **Results**
```
▶ Best Configuration
  Best CV Score: 0.773
  Best parameters: [shown in output]
  
▶ Final Model Evaluation
  Test R²: 0.773
  Test MAE: 73.9 HV
```

### When to Use This Notebook
- If you want to understand hyperparameter tuning
- If you want to optimize for your own dataset
- If you need to replicate the best classical model

### Key Takeaways
1. Systematic search finds better hyperparameters
2. max_depth=3 is optimal (prevents overfitting)
3. learning_rate=0.1 balances speed and accuracy
4. subsample=0.8 helps generalization

---

## 3. Custom Data MLM Encoder Pretraining.ipynb - Language Model

**Status**: 📓 31 total cells (19 markdown + 12 code)

### What This Notebook Covers
Fine-tuning MatSci BERT using Masked Language Modeling on 150K HEA samples

### Section-by-Section Breakdown

#### 📌 **Notebook Overview**
```
▶ Key Concepts
  1. Transfer Learning: Start with pre-trained MatSci BERT
  2. MLM: Learn from context (like BERT)
  3. Fine-tuning: Adapt to HEA compositions
  
▶ What is MLM?
  - Randomly mask 15% of tokens
  - Predict masked tokens from context
  - Forces model to learn relationships
  
▶ Benefits
  - Learns HEA grammar and vocabulary
  - Improves hardness R² (0.60 → 0.76)
  - Uses unlabeled data (150K samples)
  - No hardness labels needed for this stage
```

#### 📌 **Setup**
```
▶ Section 1: Install Packages
  - transformers (BERT models)
  - torch (deep learning)
  - datasets (data handling)
```

#### 📌 **Data Preparation**
```
▶ Section 2: Load Pre-training Data
  Dataset: 150K HEA compositions
  Format: Concatenated descriptors
  Example: "Al 0.68 Ni 0.16 Zn 0.16 ..."
  Note: These are UNLABELED (no hardness)
  
▶ Section 3: Tokenization
  What is tokenization?
  - Convert text to numeric tokens
  - BERT vocabulary: 28,996 tokens
  - Each token gets an ID (0-28995)
  
  Process:
  1. Split composition strings
  2. Map tokens to IDs
  3. Add padding for uniform length
  4. Create attention masks
  
  Output: input_ids + attention_mask
```

#### 📌 **MLM Training**
```
▶ Section 4: Create MLM Dataset
  - Tokenize all 150K compositions
  - Create Hugging Face Dataset objects
  - Prepare data collator
  
▶ Section 5: MLM Training
  Configuration:
  - Epochs: 40
  - Batch size: 16
  - Learning rate: 5e-5
  - Optimizer: AdamW with weight decay
  
  What happens:
  1. Model sees: "Al 0.68 Ni 0.16 [MASK] 0.16"
  2. Predicts: [MASK] = Zn
  3. Computes loss
  4. Updates weights
  5. Repeats for all 150K samples × 40 epochs
  
  Expected:
  - Loss decreases 0.5 → 0.3
  - Takes 2-3 hours on GPU
  - Model learns HEA grammar
```

#### 📌 **Save Weights**
```
▶ Section 6: Save Fine-tuned Model
  Output:
  - model_finetuned: BERT weights
  - tokenizer_finetuned: Tokenizer
  
  Used in: Downstream hardness prediction task
```

### Why This Matters
- Pre-trained models learn general patterns
- Fine-tuning adapts to specific domain (HEAs)
- Unlabeled data provides signal
- Results in better hardness predictions

### Connection to Other Notebooks
This fine-tuned BERT is used in:
- `train_hardness_regressor_scheduler_skip.py` (best model)
- Achieves R² = 0.762 when paired with regression head

---

## How to Use the Notebooks

### 1. **For Learning** (Recommended Order)
```
Start here:
  1. Z_my_exp.ipynb
     - Understand full ML pipeline
     - See comparison of 3 models
     - Learn feature importance
  
Then explore:
  2. Z_my_exp_lgbm.ipynb
     - Deep dive into best model
     - Understand hyperparameter tuning
  
Finally:
  3. Custom Data MLM Encoder Pretraining.ipynb
     - Learn advanced techniques
     - Understand transfer learning
```

### 2. **For Implementation**
```
Step 1: Run Z_my_exp.ipynb
  - Verifies your environment
  - Trains baseline models
  - Generates comparisons

Step 2: Run Z_my_exp_lgbm.ipynb
  - Optimizes best classical model
  - Generates deployment-ready model

Step 3: Run Custom Data MLM...
  - Fine-tune BERT for your data
  - Creates better encoder
```

### 3. **For Modification**
```
Change dataset:
  - Edit: data/dataset.csv
  - Both notebooks auto-load it

Adjust hyperparameters:
  - In GridSearchCV section
  - Change grid ranges

Tune MLM training:
  - Edit learning_rate, epochs, batch_size
  - In MLM Training section
```

## Features of the Markdown Cells

✅ **Clear Section Headers**
- "## Section X: What We're Doing"
- Easy to navigate

✅ **Objective Statements**
- "**Objective**: What this section accomplishes"

✅ **Process Explanations**
- Step-by-step algorithms
- Why each step matters

✅ **Expected Outputs**
- What results to expect
- Numeric ranges and metrics

✅ **Physical Interpretations**
- Why properties matter in materials science
- Connections to domain knowledge

✅ **Parameter Explanations**
- What each hyperparameter does
- Why we chose certain values
- How to modify for your use case

## Tips for Best Experience

1. **Read markdown before code**
   - Markdown sets context
   - Code implements the idea

2. **Run cells sequentially**
   - Each section depends on previous ones
   - Don't skip cells

3. **Examine outputs carefully**
   - Compare actual results to expected values
   - Investigate discrepancies

4. **Modify and experiment**
   - Change hyperparameters to see effects
   - Use notebooks for learning, not just replication

5. **Save outputs**
   - Models, scalers, feature importance rankings
   - Use for downstream tasks

## Markdown Cell Statistics

| Notebook | Total | Markdown | Code | Coverage |
|----------|-------|----------|------|----------|
| Z_my_exp.ipynb | 161 | 77 | 84 | 47.8% |
| Z_my_exp_lgbm.ipynb | 80 | 41 | 39 | 51.2% |
| Custom MLM.ipynb | 31 | 19 | 12 | 61.3% |
| **Total** | **272** | **137** | **135** | **50.4%** |

---

**Created**: May 2025
**Purpose**: Educational walkthrough of ML pipeline for HEA hardness prediction
**Last Updated**: 2025-05-11

