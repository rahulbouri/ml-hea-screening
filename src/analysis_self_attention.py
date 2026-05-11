import os
import random
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, BertConfig
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# 0. Reproducibility: same seed as training
# -----------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -----------------------------------------------------------------------------
# 1. Environment & Paths
# -----------------------------------------------------------------------------
os.environ['HF_HOME'] = '/Users/rahulbouri/Desktop/ml_hea/matscibert_weights/pretrained'

MODEL_PATH    = "/Users/rahulbouri/Downloads/mlm_matsci_bert/model_finetuned"
TOKENIZER_PATH= "/Users/rahulbouri/Downloads/mlm_matsci_bert/tokenizer_finetuned"
CSV_PATH      = "dataset_for_hardness_regression_language_model.csv" 
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------------------------------------------------------
# 2. Load data & split
# -----------------------------------------------------------------------------
data = pd.read_csv(CSV_PATH)  # must contain 'concat_text' and 'hardness'
train_df, test_df = train_test_split(data, test_size=0.2, random_state=SEED)

# pick a single test sample (e.g. the first one)
sample = test_df.reset_index(drop=True).loc[0]

# -----------------------------------------------------------------------------
# 3. Tokenizer
# -----------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

enc = tokenizer(
    [sample['concat_text']],
    padding='max_length',
    truncation=True,
    max_length=128,
    return_tensors='pt'
)

input_ids      = enc['input_ids'].to(DEVICE)
attention_mask = enc['attention_mask'].to(DEVICE)

# -----------------------------------------------------------------------------
# 4. Load encoder with attentions
#    (BERTConfig.output_attentions=True) :contentReference[oaicite:3]{index=3}
# -----------------------------------------------------------------------------
config = BertConfig.from_pretrained(
    MODEL_PATH,
    output_attentions=True,
    return_dict=False
)
encoder = AutoModel.from_pretrained(MODEL_PATH, config=config).to(DEVICE)
encoder.eval()

# -----------------------------------------------------------------------------
# 5. Forward pass: grab attentions
# -----------------------------------------------------------------------------
with torch.no_grad():
    last_hidden, pooler, attentions = encoder(
        input_ids=input_ids,
        attention_mask=attention_mask
    )
    # attentions: tuple of L layers, each (B, H, T, T)

# -----------------------------------------------------------------------------
# 6. Aggregate heads & pick first/last
# -----------------------------------------------------------------------------
# e.g. mean over heads
first_layer = attentions[0].mean(dim=1)[0].cpu().numpy()   # (T, T)
last_layer  = attentions[-1].mean(dim=1)[0].cpu().numpy()  # (T, T)

# -----------------------------------------------------------------------------
# 7. Map back to tokens
# -----------------------------------------------------------------------------
tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())

# -----------------------------------------------------------------------------
# 8. Save as CSV for heatmap plotting
# -----------------------------------------------------------------------------
df_first = pd.DataFrame(first_layer, index=tokens, columns=tokens)
df_last  = pd.DataFrame(last_layer,  index=tokens, columns=tokens)

os.makedirs("attention_outputs", exist_ok=True)
df_first.to_csv("attention_outputs/first_layer_attention.csv")
df_last.to_csv("attention_outputs/last_layer_attention.csv")

print("✅ Saved:")
print("  - attention_outputs/first_layer_attention.csv")
print("  - attention_outputs/last_layer_attention.csv")
