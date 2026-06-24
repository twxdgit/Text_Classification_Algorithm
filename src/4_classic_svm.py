"""
src/4_classic_svm.py
步骤4: Linear SVM — 在 BOW / TF-IDF / Word2Vec 三种向量化上各训练一次
运行: python src/4_classic_svm.py
输出: models/svm_bow.pkl, models/svm_tfidf.pkl, models/svm_w2v.pkl, results/svm_results.csv
"""
import os, pickle, numpy as np
import pandas as pd
from scipy import sparse
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, label_binarize)
from sklearn.calibration import CalibratedClassifierCV
from gensim.models import Word2Vec

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, "models")
RES_DIR   = os.path.join(BASE, "results")
DATA_DIR  = os.path.join(BASE, "data")
os.makedirs(RES_DIR, exist_ok=True)

y_tr = np.load(os.path.join(MODEL_DIR, "y_train.npy"))
y_te = np.load(os.path.join(MODEL_DIR, "y_test.npy"))
N_CLASSES = 4

def eval_report(name, y_true, y_pred, y_score=None):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro")
    rec  = recall_score(y_true, y_pred, average="macro")
    f1   = f1_score(y_true, y_pred, average="macro")
    auc  = None
    if y_score is not None:
        try:
            auc = roc_auc_score(label_binarize(y_true, classes=range(N_CLASSES)),
                                y_score, average="macro", multi_class="ovr")
        except Exception as e:
            print(f"  [warn] AUC 计算失败: {e}")
    print(f"  {name:40s} Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} AUC={auc}")
    return {"vectorizer": name, "acc": acc, "prec": prec,
            "rec": rec, "f1": f1, "auc": auc}

def load_w2v_doc_vectors(split: str):
    w2v = Word2Vec.load(os.path.join(MODEL_DIR, "w2v_agnews.model"))
    df = pd.read_csv(os.path.join(DATA_DIR,
                      "agnews_clean_train.csv" if split == "train"
                      else "agnews_clean_test.csv"), encoding="utf-8")
    dim = w2v.vector_size
    vecs = []
    for text in df["clean"].fillna(""):
        tok = str(text).split()
        v = [w2v.wv[w] for w in tok if w in w2v.wv]
        vecs.append(np.mean(v, axis=0) if v else np.zeros(dim))
    return np.vstack(vecs)

results = []

# 1) BOW — LinearSVC + CalibratedClassifierCV 以获得 predict_proba
print("[SVM + BOW]")
X_tr_bow = sparse.load_npz(os.path.join(MODEL_DIR, "bow_train.npz"))
X_te_bow = sparse.load_npz(os.path.join(MODEL_DIR, "bow_test.npz"))
base = LinearSVC(C=1.0)
clf = CalibratedClassifierCV(base, cv=3)
clf.fit(X_tr_bow, y_tr)
pred  = clf.predict(X_te_bow)
score = clf.predict_proba(X_te_bow)
results.append(eval_report("SVM+BOW", y_te, pred, score))
with open(os.path.join(MODEL_DIR, "svm_bow.pkl"), "wb") as f:
    pickle.dump(clf, f)

# 2) TF-IDF
print("[SVM + TF-IDF]")
X_tr_tf = sparse.load_npz(os.path.join(MODEL_DIR, "tfidf_train.npz"))
X_te_tf = sparse.load_npz(os.path.join(MODEL_DIR, "tfidf_test.npz"))
base = LinearSVC(C=1.0)
clf = CalibratedClassifierCV(base, cv=3)
clf.fit(X_tr_tf, y_tr)
pred  = clf.predict(X_te_tf)
score = clf.predict_proba(X_te_tf)
results.append(eval_report("SVM+TFIDF", y_te, pred, score))
with open(os.path.join(MODEL_DIR, "svm_tfidf.pkl"), "wb") as f:
    pickle.dump(clf, f)

# 3) Word2Vec
print("[SVM + Word2Vec]")
X_tr_w2v = load_w2v_doc_vectors("train")
X_te_w2v = load_w2v_doc_vectors("test")
base = LinearSVC(C=1.0)
clf = CalibratedClassifierCV(base, cv=3)
clf.fit(X_tr_w2v, y_tr)
pred  = clf.predict(X_te_w2v)
score = clf.predict_proba(X_te_w2v)
results.append(eval_report("SVM+Word2Vec", y_te, pred, score))
with open(os.path.join(MODEL_DIR, "svm_w2v.pkl"), "wb") as f:
    pickle.dump(clf, f)

pd.DataFrame(results).to_csv(
    os.path.join(RES_DIR, "svm_results.csv"), index=False, encoding="utf-8")
print(f"\n[Done] 结果保存 → {os.path.join(RES_DIR, 'svm_results.csv')}")
