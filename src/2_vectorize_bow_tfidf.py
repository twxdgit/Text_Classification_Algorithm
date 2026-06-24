"""
src/2_vectorize_bow_tfidf.py
步骤2: 用 CountVectorizer / TfidfVectorizer 向量化，保存稀疏矩阵供分类器使用
运行: python src/2_vectorize_bow_tfidf.py
输出: models/bow_*.npz, models/tfidf_*.npz, models/bow_vectorizer.pkl, models/tfidf_vectorizer.pkl
"""
import os, pickle, joblib
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

DATA_DIR = os.path.join(BASE, "data")
TRAIN = os.path.join(DATA_DIR, "agnews_clean_train.csv")
TEST  = os.path.join(DATA_DIR, "agnews_clean_test.csv")

def load():
    tr = pd.read_csv(TRAIN, encoding="utf-8")
    te = pd.read_csv(TEST,  encoding="utf-8")
    return tr["clean"].fillna("").tolist(), tr["label"].tolist(), \
           te["clean"].fillna("").tolist(),  te["label"].tolist()

X_tr, y_tr, X_te, y_te = load()

# ---------- 词袋 ----------
bow = CountVectorizer(max_features=50000, ngram_range=(1, 2), min_df=5)
X_tr_bow = bow.fit_transform(X_tr)
X_te_bow  = bow.transform(X_te)

sparse.save_npz(os.path.join(MODEL_DIR, "bow_train.npz"), X_tr_bow)
sparse.save_npz(os.path.join(MODEL_DIR, "bow_test.npz"),  X_te_bow)
with open(os.path.join(MODEL_DIR, "bow_vectorizer.pkl"), "wb") as f:
    pickle.dump(bow, f)

# ---------- TF-IDF ----------
tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                        min_df=5, sublinear_tf=True)
X_tr_tf = tfidf.fit_transform(X_tr)
X_te_tf  = tfidf.transform(X_te)

sparse.save_npz(os.path.join(MODEL_DIR, "tfidf_train.npz"), X_tr_tf)
sparse.save_npz(os.path.join(MODEL_DIR, "tfidf_test.npz"),  X_te_tf)
with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(tfidf, f)

# 保存标签
import numpy as np
np.save(os.path.join(MODEL_DIR, "y_train.npy"), np.array(y_tr))
np.save(os.path.join(MODEL_DIR, "y_test.npy"),  np.array(y_te))

print("[Done] BOW shape:", X_tr_bow.shape, "TF-IDF shape:", X_tr_tf.shape)
print("  词袋词汇量:", len(bow.vocabulary_))
print("  TF-IDF词汇量:", len(tfidf.vocabulary_))
