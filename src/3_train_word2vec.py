"""
src/3_train_word2vec.py
步骤3: 在训练语料上训练 Word2Vec 词向量
运行: python src/3_train_word2vec.py
输出: models/w2v_agnews.model, models/w2v_agnews.vectors.npy
"""
import os, numpy as np
import pandas as pd
from gensim.models import Word2Vec

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, "models")
DATA_DIR  = os.path.join(BASE, "data")
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN = os.path.join(DATA_DIR, "agnews_clean_train.csv")

def main():
    print("[load] reading clean data ...")
    df = pd.read_csv(TRAIN, encoding="utf-8")
    sentences = [str(t).split() for t in df["clean"].fillna("")]

    print(f"[train] Word2Vec on {len(sentences):,} sentences ...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=5,
        workers=4,
        epochs=10,
        seed=42,
    )

    w2v_path = os.path.join(MODEL_DIR, "w2v_agnews.model")
    model.save(w2v_path)

    # 单独保存向量矩阵，方便后续直接用
    np.save(os.path.join(MODEL_DIR, "w2v_agnews.vectors.npy"),
            model.wv.vectors, allow_pickle=False)

    print("[Done] Word2Vec 训练完成")
    print(f"  词汇量 : {len(model.wv):,}")
    print(f"  向量维度: {model.vector_size}")
    print(f"  保存位置: {w2v_path}")

if __name__ == "__main__":
    main()
