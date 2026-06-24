"""
src/1_data_prepare.py
步骤1: 下载 AG News 数据 → 清洗 → 保存 clean CSV
运行: python src/1_data_prepare.py
"""
import os, re, nltk, pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------- 路径 ----------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
os.makedirs(DATA_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test.csv")
CLEAN_TRAIN = os.path.join(DATA_DIR, "agnews_clean_train.csv")
CLEAN_TEST  = os.path.join(DATA_DIR, "agnews_clean_test.csv")

# ---------- NLTK ----------
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)

STOP = set(stopwords.words('english'))
LEMMA = WordNetLemmatizer()

# ---------- 下载 ----------
TRAIN_URL = ("https://raw.githubusercontent.com/mhjabreel/"
             "CharCnn_Keras/master/data/ag_news_csv/train.csv")
TEST_URL  = ("https://raw.githubusercontent.com/mhjabreel/"
             "CharCnn_Keras/master/data/ag_news_csv/test.csv")

def download(url, dest):
    if os.path.exists(dest):
        print(f"[skip] {dest} already exists")
        return
    print(f"[download] {url}")
    df = pd.read_csv(url, header=None, names=["label", "title", "description"],
                     quoting=3, on_bad_lines="skip", encoding="utf-8")
    df.to_csv(dest, index=False, encoding="utf-8")
    print(f"[ok] saved {len(df):,} rows → {dest}")

# ---------- 清洗 ----------
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [LEMMA.lemmatize(w) for w in tokens
              if w not in STOP and len(w) > 2]
    return " ".join(tokens)

def clean_and_save(src, dest, limit=None):
    print(f"[clean] {src} → {dest}")
    df = pd.read_csv(src, encoding="utf-8")
    if limit:
        df = df.head(limit)
        print(f"       using first {limit:,} rows")
    df["text"] = (df["title"].astype(str) + " "
                  + df["description"].astype(str))
    df["clean"] = df["text"].apply(preprocess)
    df[["label", "clean"]].to_csv(dest, index=False, encoding="utf-8")
    print(f"[ok] {len(df):,} rows saved")

# ---------- main ----------
if __name__ == "__main__":
    download(TRAIN_URL, TRAIN_CSV)
    download(TEST_URL,  TEST_CSV)
    clean_and_save(TRAIN_CSV, CLEAN_TRAIN)   # 全量 120k
    clean_and_save(TEST_CSV,  CLEAN_TEST)    # 7.6k
    print("\n[Done] 预处理完成，输出文件:")
    print(" ", CLEAN_TRAIN)
    print(" ", CLEAN_TEST)
