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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = "/Users/rahulbouri/Downloads/mlm_matsci_bert/model_finetuned"
tokenizer_path = "/Users/rahulbouri/Downloads/mlm_matsci_bert/tokenizer_finetuned"
# model_path = "m3rg-iitd/matscibert"  # Original MatSci BERT model
# tokenizer_path = "m3rg-iitd/matscibert"  # Original MatSci BERT tokenizer

# ==========================================
# 1. Data Loading & Preprocessing
# ==========================================
# Load your labeled dataset for regression
data = pd.read_csv('dataset_for_hardness_regression_language_model.csv')  # contains 'composition' and 'hardness'

train_df, val_df = train_test_split(data, test_size=0.2, random_state=42)

class RegressionDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts.tolist(), padding=True, truncation=True, max_length=max_length, return_tensors='pt'
        )
        self.targets = torch.tensor(targets.values, dtype=torch.float)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.targets[idx]
        return item

# ==========================================
# 3. Tokenize the data
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Create Datasets & DataLoaders
train_dataset = RegressionDataset(train_df['concat_text'], train_df['hardness'], tokenizer)
val_dataset   = RegressionDataset(val_df['concat_text'],   val_df['hardness'], tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=16)

# ==========================================
# 4. Model Definition: Frozen Encoder + FC Regression Head
# ==========================================
class TransformerRegressor(nn.Module):
    def __init__(self, pretrained_model_name, hidden_size=768, fc_dims=[256, 128, 64, 8, 1]):
        super().__init__()
        # Load & freeze encoder
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        for param in self.encoder.parameters():
            param.requires_grad = False
        # Build regression head
        layers = []
        in_dim = hidden_size
        for out_dim in fc_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != 1:
                layers.append(nn.ReLU())
            in_dim = out_dim
        self.regressor = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS]
        return self.regressor(cls_emb).squeeze(-1)

# Instantiate model
model = TransformerRegressor(model_path)
model = model.to(device)

# ==========================================
# 4. Optimizer with Layer-wise Weight Decay on FC Head
# ==========================================
# 1) collect your trainable params
trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

# 2) specify any per-layer overrides here:
#    key = layer index (in the Sequential), value = weight_decay
per_layer_wd = {
    0: 0.02,   # first Linear’s weight → 0.02
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

# 5) instantiate optimizer
optimizer = torch.optim.AdamW(param_groups, lr=1e-4)

# ==========================================
# 5. Training Loop
# ==========================================
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    criterion = nn.MSELoss()
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['labels'].to(device)

        preds = model(input_ids=input_ids, attention_mask=attention_mask)
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
        targets = batch['labels'].to(device)
        preds = model(input_ids=input_ids, attention_mask=attention_mask)
        preds_list.append(preds.cpu())
        labels_list.append(targets.cpu())
    preds_all = torch.cat(preds_list).numpy()
    labels_all = torch.cat(labels_list).numpy()
    return preds_all, labels_all

# Training hyperparams
epochs = 200
best_val_loss = float('inf')

for epoch in range(1, epochs + 1):
    train_loss = train_epoch(model, train_loader, optimizer, device)
    preds_val, labels_val = evaluate(model, val_loader, device)
    val_loss = mean_squared_error(labels_val, preds_val)

    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val MSE = {val_loss:.4f}, Val MAE = {mean_absolute_error(labels_val, preds_val):.4f}, Val R2 = {r2_score(labels_val, preds_val):.4f}")

    # Save best head weights
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        os.makedirs('/Users/rahulbouri/Desktop/ml_hea/finetuned_matscibert_weights/trained_hardness_regressor_model/best_model_200_full_run', exist_ok=True)
        torch.save(model.state_dict(), '/Users/rahulbouri/Desktop/ml_hea/finetuned_matscibert_weights/trained_hardness_regressor_model/best_model_200_full_run/regressor_head.pt')
        print(f"--> New best model saved (MSE={val_loss:.4f})")

# ==========================================
# 6. Final Evaluation & Unscale
# ==========================================
preds_val, labels_val = evaluate(model, val_loader, device)

preds_val = preds_val.reshape(-1,1).squeeze()
labels_val = labels_val.reshape(-1,1).squeeze()

print("Final Metrics on Validation:")
print("MSE:", mean_squared_error(labels_val, preds_val))
print("MAE:", mean_absolute_error(labels_val, preds_val))
print("R2 :", r2_score(labels_val, preds_val))

# Save final tokenizer
tokenizer.save_pretrained('/Users/rahulbouri/Desktop/ml_hea/finetuned_matscibert_weights/trained_hardness_regressor_model_200_full_run')

