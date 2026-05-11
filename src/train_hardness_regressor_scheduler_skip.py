import os
os.environ['HF_HOME'] = '/Users/rahulbouri/Desktop/ml_hea/matscibert_weights/pretrained'

import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import traceback

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# Configuration - Set your root directory here
# ==========================================
ROOT_SAVE_DIR = '/Users/rahulbouri/Desktop/ml_hea/finetuned_matscibert_weights/language_regressor_with_skip_connections/'
os.makedirs(ROOT_SAVE_DIR, exist_ok=True)

# Test write permissions
try:
    test_file = os.path.join(ROOT_SAVE_DIR, 'test_write.txt')
    with open(test_file, 'w') as f:
        f.write('test')
    os.remove(test_file)
    print(f"✓ Write permissions confirmed for {ROOT_SAVE_DIR}")
except Exception as e:
    print(f"✗ Write permission error for {ROOT_SAVE_DIR}: {e}")
    exit(1)

model_path = "/Users/rahulbouri/Downloads/mlm_matsci_bert/model_finetuned"
tokenizer_path = "/Users/rahulbouri/Downloads/mlm_matsci_bert/tokenizer_finetuned"

# ==========================================
# 1. Data Loading & Preprocessing
# ==========================================
# Load your labeled dataset for regression
data = pd.read_csv('dataset_for_hardness_regression_language_model.csv')  # contains 'composition' and 'hardness'

# Define the numerical features for skip connections
numerical_features = [
    "avg_Atomic_Radius",
    "avg_Pauling_Electronegativity", 
    "avg_number_of_valence_electrons",
    "avg_Cohesive_energy_ev_atom",
    "avg_Bulk_modulus_RT_Gpa",
    "avg_Elastic_modulus_RT_Gpa",
    "avg_Melting_point_(K)",
    "avg_lattice_constant_A",
    "avg_BEC_percm3",
    "avg_Av.Valence_bond_strength_ev",
    "avg_EngelZ_e/a"
]

# Normalize numerical features
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

train_df, val_df = train_test_split(data, test_size=0.2, random_state=42)

class RegressionDataset(Dataset):
    def __init__(self, texts, numerical_features, targets, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts.tolist(), padding=True, truncation=True, max_length=max_length, return_tensors='pt'
        )
        self.numerical_features = torch.tensor(numerical_features.values, dtype=torch.float)
        self.targets = torch.tensor(targets.values, dtype=torch.float)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['numerical_features'] = self.numerical_features[idx]
        item['labels'] = self.targets[idx]
        return item

# ==========================================
# 3. Tokenize the data
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Create Datasets & DataLoaders
train_dataset = RegressionDataset(
    train_df['concat_text'], 
    train_df[numerical_features], 
    train_df['hardness'], 
    tokenizer
)
val_dataset = RegressionDataset(
    val_df['concat_text'], 
    val_df[numerical_features], 
    val_df['hardness'], 
    tokenizer
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# ==========================================
# 4. Model Definition: Frozen Encoder + FC Regression Head with Skip Connections
# ==========================================
class TransformerRegressor(nn.Module):
    def __init__(self, pretrained_model_name, hidden_size=768, num_numerical_features=11, fc_dims=[256, 128, 64, 8, 1]):
        super().__init__()
        # Load & freeze encoder
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Build regression head with skip connections
        layers = []
        in_dim = hidden_size + num_numerical_features  # Concatenate BERT output with numerical features
        
        for out_dim in fc_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != 1:
                layers.append(nn.ReLU())
            in_dim = out_dim
        
        self.regressor = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask, numerical_features):
        # Get BERT embeddings
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS]
        
        # Concatenate BERT embeddings with numerical features (skip connection)
        combined_features = torch.cat([cls_emb, numerical_features], dim=1)
        
        return self.regressor(combined_features).squeeze(-1)

# Instantiate model
model = TransformerRegressor(model_path, num_numerical_features=len(numerical_features))
model = model.to(device)

# ==========================================
# 4. Optimizer with Layer-wise Weight Decay on FC Head
# ==========================================
# 1) collect your trainable params
trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

# 2) specify any per-layer overrides here:
#    key = layer index (in the Sequential), value = weight_decay
per_layer_wd = {
    0: 0.02,   # first Linear's weight → 0.02
    2: 0.02,
    4: 0.02
}
default_wd = 0.01  # everything else (weights) falls back to this

# 3) bucket params by decay value
wd_to_params = defaultdict(list)
for name, param in trainable:
    # name is like 'regressor.0.weight' or 'regressor.0.bias'
    if not name.startswith('regressor.'):
        # (if you had other trainable parts you wanted to include)
        wd_to_params[default_wd].append(param)
        continue

    _, module_idx_str, param_type = name.split('.')  
    module_idx = int(module_idx_str)

    if param_type == 'bias':
        wd = 0.0
    else:  # 'weight'
        wd = per_layer_wd.get(module_idx, default_wd)

    wd_to_params[wd].append(param)

# 4) build your param_groups list
param_groups = [
    {'params': params, 'weight_decay': wd}
    for wd, params in wd_to_params.items()
]

# 5) instantiate optimizer with initial learning rate of 0.1
optimizer = torch.optim.AdamW(param_groups, lr=0.01)

# 6) Add learning rate scheduler
# Using ReduceLROnPlateau scheduler that reduces learning rate when validation loss plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',           # Monitor validation loss (minimize)
    factor=0.1,           # Reduce LR to tenth when plateauing
    patience=5,          # Wait 5 epochs before reducing LR
    min_lr=1e-7          # Minimum learning rate
)

# ==========================================
# 5. Training Loop with Statistics Tracking
# ==========================================
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    criterion = nn.MSELoss()
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        numerical_features = batch['numerical_features'].to(device)
        targets = batch['labels'].to(device)

        preds = model(input_ids=input_ids, attention_mask=attention_mask, numerical_features=numerical_features)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * input_ids.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds_list, labels_list = [], []
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        numerical_features = batch['numerical_features'].to(device)
        targets = batch['labels'].to(device)
        preds = model(input_ids=input_ids, attention_mask=attention_mask, numerical_features=numerical_features)
        preds_list.append(preds.cpu())
        labels_list.append(targets.cpu())
    preds_all = torch.cat(preds_list).numpy()
    labels_all = torch.cat(labels_list).numpy()
    return preds_all, labels_all

def save_training_stats(training_stats, save_dir):
    """Save training statistics to CSV with error handling"""
    try:
        training_stats_df = pd.DataFrame(training_stats)
        stats_file = os.path.join(save_dir, 'training_statistics.csv')
        training_stats_df.to_csv(stats_file, index=False)
        print(f"✓ Training statistics saved to {stats_file}")
        return True
    except Exception as e:
        print(f"✗ Error saving training statistics: {e}")
        return False

def save_final_metrics(final_mse, final_mae, final_r2, save_dir):
    """Save final metrics to CSV with error handling"""
    try:
        final_metrics = {
            'Metric': ['MSE', 'MAE', 'R2'],
            'Value': [final_mse, final_mae, final_r2]
        }
        final_metrics_df = pd.DataFrame(final_metrics)
        metrics_file = os.path.join(save_dir, 'final_metrics.csv')
        final_metrics_df.to_csv(metrics_file, index=False)
        print(f"✓ Final metrics saved to {metrics_file}")
        return True
    except Exception as e:
        print(f"✗ Error saving final metrics: {e}")
        return False

# Training hyperparams
epochs = 100
best_val_loss = float('inf')
previous_lr = optimizer.param_groups[0]['lr']  # Track initial learning rate

# Initialize training statistics tracking
training_stats = {
    'epoch': [],
    'train_loss': [],
    'val_loss': [],
    'val_mae': [],
    'val_r2': [],
    'learning_rate': []
}

print("Starting training...")
print(f"Training for {epochs} epochs")
print(f"Save directory: {ROOT_SAVE_DIR}")

try:
    for epoch in range(1, epochs + 1):
        try:
            train_loss = train_epoch(model, train_loader, optimizer, device)
            preds_val, labels_val = evaluate(model, val_loader, device)
            val_loss = mean_squared_error(labels_val, preds_val)
            val_mae = mean_absolute_error(labels_val, preds_val)
            val_r2 = r2_score(labels_val, preds_val)

            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Check if learning rate changed
            if current_lr != previous_lr:
                print(f"Epoch {epoch}: Learning rate updated from {previous_lr:.6f} to {current_lr:.6f}")
                previous_lr = current_lr
            
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val MSE = {val_loss:.4f}, Val MAE = {val_mae:.4f}, Val R2 = {val_r2:.4f}, LR = {current_lr:.6f}")

            # Track training statistics
            training_stats['epoch'].append(epoch)
            training_stats['train_loss'].append(train_loss)
            training_stats['val_loss'].append(val_loss)
            training_stats['val_mae'].append(val_mae)
            training_stats['val_r2'].append(val_r2)
            training_stats['learning_rate'].append(current_lr)

            # Save best head weights
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(ROOT_SAVE_DIR, 'regressor_head.pt'))
                print(f"--> New best model saved (MSE={val_loss:.4f})")

            # Update learning rate
            scheduler.step(val_loss)
            
            # Save intermediate statistics every 10 epochs
            if epoch % 10 == 0:
                save_training_stats(training_stats, ROOT_SAVE_DIR)
                
        except Exception as e:
            print(f"Error during epoch {epoch}: {e}")
            print(traceback.format_exc())
            # Save what we have so far
            save_training_stats(training_stats, ROOT_SAVE_DIR)
            break

    # ==========================================
    # 6. Final Evaluation & Save Statistics
    # ==========================================
    print("Training completed. Running final evaluation...")
    
    preds_val, labels_val = evaluate(model, val_loader, device)

    preds_val = preds_val.reshape(-1,1).squeeze()
    labels_val = labels_val.reshape(-1,1).squeeze()

    final_mse = mean_squared_error(labels_val, preds_val)
    final_mae = mean_absolute_error(labels_val, preds_val)
    final_r2 = r2_score(labels_val, preds_val)

    print("Final Metrics on Validation:")
    print("MSE:", final_mse)
    print("MAE:", final_mae)
    print("R2 :", final_r2)

    # Save training statistics
    save_training_stats(training_stats, ROOT_SAVE_DIR)

    # Save final metrics
    save_final_metrics(final_mse, final_mae, final_r2, ROOT_SAVE_DIR)

    # Save final tokenizer with proper naming and verification
    try:
        # Create a dedicated tokenizer directory
        tokenizer_save_dir = os.path.join(ROOT_SAVE_DIR, 'tokenizer')
        os.makedirs(tokenizer_save_dir, exist_ok=True)
        
        # Save the tokenizer
        tokenizer.save_pretrained(tokenizer_save_dir)
        
        # Verify the save by checking what files were created
        saved_files = os.listdir(tokenizer_save_dir)
        print(f"✓ Tokenizer saved successfully to {tokenizer_save_dir}")
        print(f"  Files saved: {saved_files}")
        
        # Test loading the saved tokenizer to ensure it works
        try:
            test_tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_dir)
            print("✓ Tokenizer verification successful - can be loaded back")
            
            # Test basic functionality
            test_text = "test composition"
            test_encoding = test_tokenizer(test_text, return_tensors='pt')
            print("✓ Tokenizer functionality verified")
            
        except Exception as verify_error:
            print(f"⚠ Warning: Tokenizer verification failed: {verify_error}")
            
    except Exception as e:
        print(f"✗ Error saving tokenizer: {e}")
        print(traceback.format_exc())
        
        # Fallback: try saving to root directory
        try:
            print("Attempting fallback save to root directory...")
            tokenizer.save_pretrained(ROOT_SAVE_DIR)
            print("✓ Tokenizer saved to root directory as fallback")
        except Exception as fallback_error:
            print(f"✗ Fallback save also failed: {fallback_error}")

    print("Training completed! Statistics and metrics saved.")
    print(f"All files saved to: {ROOT_SAVE_DIR}")
    print(f"Tokenizer saved to: {os.path.join(ROOT_SAVE_DIR, 'tokenizer')}")

except Exception as e:
    print(f"Critical error during training: {e}")
    print(traceback.format_exc())
    # Save what we have so far
    save_training_stats(training_stats, ROOT_SAVE_DIR)
    print("Partial results saved due to error.")
