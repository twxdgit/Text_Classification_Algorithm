"""
src/6_bert.py
步骤6: BERT 微调（加分项）
  - 使用 HuggingFace bert-base-uncased
  - 对 AG News 4类分类进行 Fine-tune
运行: python src/6_bert.py
输出: models/bert_best, results/bert_results.csv
注意: 首次运行会自动下载 bert-base-uncased (~420 MB)
"""
import os, time
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          AdamW, get_linear_schedule_with_warmup)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, label_binarize)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = os.path.join(BASE, "models")
RES_DIR   = os.path.join(BASE, "results")
DATA_DIR  = os.path.join(BASE, "data")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RES_DIR,   exist_ok=True)

print(f"[info] device: {DEVICE}")

# ---------- 超参数 ----------
MODEL_NAME = "bert-base-uncased"
MAX_LEN    = 128
BATCH_SZ   = 16
EPOCHS     = 3
LR         = 2e-5
N_CLASSES  = 4

# ---------- 加载数据 ----------
def load_split(split):
    fname = "agnews_clean_train.csv" if split == "train" else "agnews_clean_test.csv"
    df = pd.read_csv(os.path.join(DATA_DIR, fname), encoding="utf-8")
    return df["clean"].fillna("").tolist(), df["label"].tolist()

print("[load] loading data ...")
X_tr, y_tr = load_split("train")
X_te, y_te = load_split("test")

from sklearn.model_selection import train_test_split
X_tr, X_va, y_tr, y_va = train_test_split(
    X_tr, y_tr, test_size=0.1, random_state=42, stratify=y_tr)

# ---------- Tokenizer ----------
print(f"[load] loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def encode(texts, labels):
    enc = tokenizer(texts, truncation=True, padding=True,
                    max_length=MAX_LEN, return_tensors="pt")
    return enc, torch.tensor(labels, dtype=torch.long)

tr_enc, tr_lbl = encode(X_tr, y_tr)
va_enc, va_lbl = encode(X_va, y_va)
te_enc, te_lbl = encode(X_te, y_te)

class BERTDataset(Dataset):
    def __init__(self, enc, labels):
        self.enc  = enc
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}, self.labels[i]

train_dl = DataLoader(BERTDataset(tr_enc, tr_lbl), batch_size=BATCH_SZ, shuffle=True)
val_dl   = DataLoader(BERTDataset(va_enc, va_lbl), batch_size=BATCH_SZ, shuffle=False)
test_dl  = DataLoader(BERTDataset(te_enc, te_lbl), batch_size=BATCH_SZ, shuffle=False)

# ---------- 模型 ----------
print(f"[load] loading model: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=N_CLASSES)
model = model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR)
total_steps = len(train_dl) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps)

# ---------- 训练 ----------
def run_epoch(dl, train=True):
    model.train() if train else model.eval()
    total_loss, total_correct, total = 0, 0, 0
    for batch, labels in dl:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        labels = labels.to(DEVICE)
        if train:
            optimizer.zero_grad()
        out = model(**batch, labels=labels)
        loss = out.loss
        if train:
            loss.backward()
            optimizer.step()
            scheduler.step()
        preds = out.logits.argmax(dim=1)
        total_loss    += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total         += labels.size(0)
    return total_loss / total, total_correct / total

best_val_acc = 0
best_state   = None
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_acc = run_epoch(train_dl, train=True)
    va_loss, va_acc = run_epoch(val_dl,   train=False)
    elapsed = time.time() - t0
    print(f"  Epoch {epoch}/{EPOCHS}  "
          f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
          f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}  "
          f"time={elapsed:.0f}s")
    if va_acc > best_val_acc:
        best_val_acc = va_acc
        best_state   = model.state_dict()
        save_path = os.path.join(MODEL_DIR, "bert_best")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"  [new best] val_acc={va_acc:.4f} → saved")

# ---------- 测试 ----------
model.load_state_dict(best_state)
model.eval()

all_preds, all_scores = [], []
with torch.no_grad():
    for batch, _ in test_dl:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        logits = model(**batch).logits
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_scores.append(torch.softmax(logits, dim=1).cpu().numpy())

y_pred  = np.concatenate(all_preds)
y_score = np.concatenate(all_scores)
y_true  = np.array(y_te, dtype=np.int64)

acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average="macro")
rec  = recall_score(y_true, y_pred, average="macro")
f1   = f1_score(y_true, y_pred, average="macro")
auc  = roc_auc_score(label_binarize(y_true, classes=range(N_CLASSES)),
                     y_score, average="macro", multi_class="ovr")

print(f"\n[Test] BERT   Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} AUC={auc:.4f}")

pd.DataFrame([{
    "model": "BERT", "vectorizer": "BERT-Tokenizer",
    "acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc
}]).to_csv(os.path.join(RES_DIR, "bert_results.csv"), index=False, encoding="utf-8")
print(f"[Done] 结果保存 → {os.path.join(RES_DIR, 'bert_results.csv')}")
