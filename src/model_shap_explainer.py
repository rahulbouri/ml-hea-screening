#!/usr/bin/env python3
# explain_model.py

import os
import argparse
import numpy as np
import pandas as pd
import torch
import shap
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

# ─── USER CONFIG ────────────────────────────────────────────────────────────────

NUMERICAL_FEATURES = [
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
    "avg_EngelZ_e/a",
]

# ─── MODEL DEFINITION ───────────────────────────────────────────────────────────

class TransformerRegressor(nn.Module):
    def __init__(self, pretrained_model_name, hidden_size=768,
                 num_numerical_features=len(NUMERICAL_FEATURES),
                 fc_dims=[256, 128, 64, 8, 1]):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        for p in self.encoder.parameters():
            p.requires_grad = False

        layers = []
        in_dim = hidden_size + num_numerical_features
        for out_dim in fc_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != 1:
                layers.append(nn.ReLU())
            in_dim = out_dim
        self.regressor = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask, numerical_features):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = out.last_hidden_state[:, 0, :]
        combined = torch.cat([cls_emb, numerical_features], dim=1)
        return self.regressor(combined).squeeze(-1)

# ─── LOADING & PREDICTION ──────────────────────────────────────────────────────

def load_model(model_weights_path, tokenizer_dir, pretrained_model_dir, device):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = TransformerRegressor(pretrained_model_dir).to(device).eval()
    state = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(state)
    return tokenizer, model

def predict_fn(texts, numeric_array, tokenizer, model, device):
    # Ensure texts is a list of strings
    if isinstance(texts, np.ndarray):
        texts = texts.tolist()
    if isinstance(texts, str):
        texts = [texts]
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)
    num_feat = torch.tensor(numeric_array, dtype=torch.float32, device=device)
    with torch.no_grad():
        preds = model(enc["input_ids"], enc["attention_mask"], num_feat)
    return preds.cpu().numpy()

# ─── SHAP EXPLAINER ─────────────────────────────────────────────────────────────

def run_shap(text_example, num_example, tokenizer, model, device):
    masker = shap.maskers.Text(tokenizer, mask_token=tokenizer.pad_token)
    def f(texts):
        dummy = np.zeros((len(texts), num_example.shape[1]))
        return predict_fn(texts, dummy, tokenizer, model, device)
    explainer = shap.Explainer(f, masker=masker)
    shap_vals = explainer([text_example])
    return shap_vals[0]

# ─── LIME EXPLAINER ─────────────────────────────────────────────────────────────

def run_lime(text_example, num_example, tokenizer, model, device):
    print("Setting up LIME explainer...")
    def f(texts):
        dummy = np.zeros((len(texts), num_example.shape[1]))
        preds = predict_fn(texts, dummy, tokenizer, model, device)
        return np.array(preds).reshape(-1, 1)  # Ensure output is 2D for regression
    explainer = LimeTextExplainer(class_names=["hardness"])
    print("Running LIME explanation...")
    return explainer.explain_instance(
        text_example, f, num_features=15, num_samples=500, labels=[0]
    )

# ─── STRUCTURED PLOTTING ─────────────────────────────────────────────────────────

def plot_token_attributions(tokens, scores, title, save_path):
    import numpy as np
    import matplotlib.pyplot as plt
    norm = (np.array(scores) - np.min(scores)) / (np.ptp(scores) + 1e-8)
    colors = plt.cm.Reds(norm)
    plt.figure(figsize=(max(12, len(tokens) * 0.6), 2))
    for i, (tok, col) in enumerate(zip(tokens, colors)):
        plt.text(i, 0.5, tok, rotation=45, ha='right', va='center',
                 bbox=dict(facecolor=col, pad=2, edgecolor='none'),
                 fontsize=8)
    plt.axis("off")
    plt.title(title)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.85, bottom=0.15)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved → {save_path}")

# ─── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_csv",           required=True)
    p.add_argument("--model_weights",      required=True)
    p.add_argument("--tokenizer_dir",      required=True)
    p.add_argument("--pretrained_model_dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    # load
    df    = pd.read_csv(args.data_csv)
    texts = df["concat_text"].tolist()
    nums  = df[NUMERICAL_FEATURES].values

    # pick example
    idx = 0
    raw_text = texts[idx]
    num_ex   = nums[[idx]]

    # load model
    tokenizer, model = load_model(
        args.model_weights,
        args.tokenizer_dir,
        args.pretrained_model_dir,
        args.device
    )

    # SHAP
    print("Running SHAP…")
    sv = run_shap(raw_text, num_ex, tokenizer, model, args.device)
    shap_tokens = sv.data.tolist()
    shap_scores = sv.values.tolist()
    plot_token_attributions(
        shap_tokens, shap_scores,
        "SHAP Token Importances",
        "shap_structured.png"
    )

    # LIME
    print("Running LIME…")
    lime_exp = run_lime(raw_text, num_ex, tokenizer, model, args.device)
    lime_list = lime_exp.as_list(label=0)
    lime_tokens, lime_scores = zip(*lime_list)
    plot_token_attributions(
        lime_tokens, lime_scores,
        "LIME Token Importances",
        "lime_structured.png"
    )

if __name__ == "__main__":
    main()
