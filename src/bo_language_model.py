import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CANDIDATES_CSV = 'virtual_candidates.csv'    # your 9.12M‐row CSV
MODEL_HEAD_PATH = os.path.join(ROOT_SAVE_DIR, 'regressor_head.pt')
TOKENIZER_PATH   = tokenizer_path
PRETRAINED_MODEL = model_path

# Numerical feature names must match your CSV
NUM_FEATS = [
    "avg_Atomic_Radius",
    "avg_Pauling_Electronegativity", 
    "avg_number_of_valence_electrons",
    # … etc (11 total)
]

# ─────────────────────────────────────────────────────────────────────────────
# 2. Load candidates & scaler
# ─────────────────────────────────────────────────────────────────────────────
cands = pd.read_csv(CANDIDATES_CSV)
scaler = StandardScaler()
cands[NUM_FEATS] = scaler.fit_transform(cands[NUM_FEATS])  # or load your fitted scaler

# ─────────────────────────────────────────────────────────────────────────────
# 3. Define your model class and load weights
# ─────────────────────────────────────────────────────────────────────────────
class TransformerRegressor(nn.Module):
    def __init__(self, pretrained_model_name, hidden_size=768,
                 num_numerical_features=11, fc_dims=[256,128,64,8,1]):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        for p in self.encoder.parameters(): p.requires_grad = False

        layers, in_dim = [], hidden_size + num_numerical_features
        for out_dim in fc_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim!=1: layers.append(nn.ReLU())
            in_dim = out_dim
        self.regressor = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask, numerical_features):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask)\
                 .last_hidden_state[:,0,:]
        x = torch.cat([h, numerical_features], dim=1)
        return self.regressor(x).squeeze(-1)

# Instantiate & load
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = TransformerRegressor(PRETRAINED_MODEL, num_numerical_features=len(NUM_FEATS))
model.load_state_dict(torch.load(MODEL_HEAD_PATH, map_location=device), strict=False)
model.to(device).eval()

# ─────────────────────────────────────────────────────────────────────────────
# 4. Black-box function wrapping your surrogate
# ─────────────────────────────────────────────────────────────────────────────
def surrogate_predict(idx: float) -> float:
    """
    BO will supply a float between 0 and len(cands)-1;
    we round to an int index and predict hardness.
    """
    i = int(np.clip(idx, 0, len(cands)-1))
    row = cands.iloc[i]
    # tokenize text
    enc = tokenizer(
        row['concat_text'],
        truncation=True, padding='max_length',
        max_length=128, return_tensors='pt'
    ).to(device)
    num_feats = torch.tensor(row[NUM_FEATS].values, dtype=torch.float)\
                    .unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(input_ids=enc.input_ids,
                     attention_mask=enc.attention_mask,
                     numerical_features=num_feats)
    return float(pred.cpu().item())

# ─────────────────────────────────────────────────────────────────────────────
# 5. Run Bayesian Optimization over the index dimension
# ─────────────────────────────────────────────────────────────────────────────
pbounds = {'idx': (0, len(cands)-1)}
optimizer = BayesianOptimization(
    f=surrogate_predict,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

# initial random probes + BO iterations
optimizer.maximize(init_points=5, n_iter=45)

# extract the best composition
best_idx = int(optimizer.max['params']['idx'])
best_row = cands.iloc[best_idx]
print("► Best predicted hardness:", optimizer.max['target'])
print("► Composition:", best_row['concat_text'])
print("► Numerical feats:", best_row[NUM_FEATS].to_dict())
