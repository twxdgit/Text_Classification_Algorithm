"""
src/5_bilstm.py
步骤5: BiLSTM 文本分类器
  - 词向量: 使用步骤3训练的 Word2Vec (100d)
  - 输入: clean 文本 → Word2Vec 词索引 → pad → Embedding → BiLSTM → FC
运行: python src/5_bilstm.py
输出: models/bilstm_best.pt, results/bilstm_results.csv
"""
import os, pickle, time
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from gensim.models import Word2Vec

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[info] using device: {DEVICE}")

# ---------- 超参数 ----------
MAX_LEN   = 200
BATCH_SZ  = 64
HIDDEN    = 128
EPOCHS    = 6
LR        = 1e-3
DROPOUT   = 0.3
N_CLASSES = 4

# ---------- 路径 ----------
MODEL_DIR = os.path.join(BASE, "models")
RES_DIR   = os.path.join(BASE, "results")
DATA_DIR  = os.path.join(BASE, "data")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RES_DIR,   exist_ok=True)

# ---------- 加载数据 ----------
def load_split(split):
    fname = "agnews_clean_train.csv" if split == "train" else "agnews_clean_test.csv"
    df = pd.read_csv(os.path.join(DATA_DIR, fname), encoding="utf-8")
    return df["clean"].fillna("").tolist(), df["label"].tolist()

print("[load] loading clean data ...")
X_tr, y_tr = load_split("train")
X_te, y_te = load_split("test")

# 从训练集 9:1 切出验证集
from sklearn.model_selection import train_test_split
X_tr, X_va, y_tr, y_va = train_test_split(
    X_tr, y_tr, test_size=0.1, random_state=42, stratify=y_tr)

# ---------- 词表 & 词向量矩阵 ----------
print("[load] loading Word2Vec ...")
w2v = Word2Vec.load(os.path.join(MODEL_DIR, "w2v_agnews.model"))
DIM = w2v.vector_size

# 构建词表: 0=pad, 1=unk, 2..N=word
word2idx = {"<pad>": 0, "<unk>": 1}
for w in w2v.wv.index_to_key:
    word2idx[w] = len(word2idx)

# 构建 embedding 矩阵
embed_matrix = np.zeros((len(word2idx), DIM), dtype=np.float32)
for w, idx in word2idx.items():
    if w in ("<pad>", "<unk>"):
        continue
    embed_matrix[idx] = w2v.wv[w]

print(f"  词表大小: {len(word2idx):,}, 词向量维度: {DIM}")

# ---------- Dataset ----------
class TextDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len):
        self.max_len = max_len
        self.labels  = np.array(labels, dtype=np.int64)
        self.ids = []
        for t in texts:
            tok = str(t).split()
            ids = [word2idx.get(w, 1) for w in tok][:max_len]
            self.ids.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.ids[i], self.labels[i]

def collate_fn(batch):
    seqs, labels = zip(*batch)
    seqs  = pad_sequence(seqs, batch_first=True, padding_value=0)
    mask  = (seqs != 0).long()
    return seqs.to(DEVICE), mask.to(DEVICE), torch.tensor(labels).to(DEVICE)

train_ds = TextDataset(X_tr, y_tr, word2idx, MAX_LEN)
val_ds   = TextDataset(X_va, y_va, word2idx, MAX_LEN)
test_ds  = TextDataset(X_te, y_te, word2idx, MAX_LEN)

train_dl = DataLoader(train_ds, batch_size=BATCH_SZ, shuffle=True,  collate_fn=collate_fn)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SZ, shuffle=False, collate_fn=collate_fn)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SZ, shuffle=False, collate_fn=collate_fn)

# ---------- 模型 ----------
class BiLSTMClassifier(nn.Module):
    def __init__(self, embed_matrix, hidden=HIDDEN, num_classes=N_CLASSES, dropout=DROPOUT):
        super().__init__()
        num_emb, dim = embed_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embed_matrix), padding_idx=0, freeze=False)
        self.lstm = nn.LSTM(dim, hidden, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.fc   = nn.Linear(2 * hidden, num_classes)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.embedding(x)                    # (B, L, D)
        lengths = mask.sum(dim=1).cpu()
        packed  = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        # h: (2, B, H) → 拼接前向和后向
        h = torch.cat([h[-2], h[-1]], dim=1)     # (B, 2H)
        h = self.drop(h)
        return self.fc(h)

model = BiLSTMClassifier(embed_matrix=embed_matrix).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---------- 训练 ----------
def run_epoch(dl, train=True):
    model.train() if train else model.eval()
    total_loss, total_correct, total = 0, 0, 0
    for seqs, mask, labels in dl:
        if train:
            optimizer.zero_grad()
        logits = model(seqs, mask)
        loss   = criterion(logits, labels)
        if train:
            loss.backward()
            optimizer.step()
        preds = logits.argmax(dim=1)
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
    print(f"  Epoch {epoch:2d}/{EPOCHS}  "
          f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
          f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}  "
          f"time={elapsed:.1f}s")
    if va_acc > best_val_acc:
        best_val_acc = va_acc
        best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        torch.save(best_state, os.path.join(MODEL_DIR, "bilstm_best.pt"))
        print(f"  [new best] val_acc={va_acc:.4f} → saved")

print(f"\n[Best val acc: {best_val_acc:.4f}]")

# ---------- 测试集评估 ----------
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, label_binarize)

model.load_state_dict(best_state)
model.eval()

all_preds, all_scores = [], []
with torch.no_grad():
    for seqs, mask, _ in test_dl:
        logits = model(seqs, mask)
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

print(f"\n[Test] BiLSTM  Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} AUC={auc:.4f}")

pd.DataFrame([{
    "model": "BiLSTM", "vectorizer": "Word2Vec",
    "acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc
}]).to_csv(os.path.join(RES_DIR, "bilstm_results.csv"), index=False, encoding="utf-8")
print(f"[Done] 结果保存 → {os.path.join(RES_DIR, 'bilstm_results.csv')}")
