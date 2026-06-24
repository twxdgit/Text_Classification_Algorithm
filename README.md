# AG News 多算法文本分类实验

《数据仓库与数据挖掘》期末大作业

## 一、项目概述

对同一份文本数据集，分别使用 **词袋 (Bag-of-Words)**、**TF-IDF**、**Word2Vec** 三种方式向量化，再用多种经典机器学习算法和深度学习算法进行 4 类新闻主题分类，最后对比各方法的分类效果。

## 二、数据集

| 项目 | 说明 |
|------|------|
| 数据集 | AG News（4 类英文新闻分类） |
| 来源 | https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv |
| 训练集 | 120,000 条 |
| 测试集 | 7,600 条 |
| 类别 | World (1) / Sports (2) / Business (3) / Sci/Tech (4) |
| 特征 | `title` + `description`（拼接为 `text`） |

### 数据格式示例

```
label,title,description
3,Wall St. Bears Claw Back Into the Black (Reuters),Reuters - Short-sellers, Wall Street's...
1,Palestinian says he met Israeli officials (Reuters),Reuters - A Palestinian official...
```

## 三、环境要求

### Python 版本

```
Python 3.9+
```

### 依赖库（`pip install` 安装）

```bash
pip install pandas numpy scikit-learn gensim nltk xgboost \
            matplotlib seaborn torch transformers tqdm
```

### NLTK 数据

需要手动下载 NLTK 数据包并放到 `C:\Users\<用户名>\AppData\Roaming\nltk_data\`：

1. 访问 https://github.com/nltk/nltk_data → Code → Download ZIP
2. 解压后，将 `packages/` 下的所有子目录移动到 `C:\Users\<用户名>\AppData\Roaming\nltk_data\`
3. 验证：

```powershell
python -c "import nltk; print(nltk.data.find('tokenizers/punkt_tab.zip')); print('ok')"
```

## 四、项目目录结构

```
源代码/
├── README.md                          # 本文件
├── 实验方案.md                         # 实验方案文档
├── 题目.txt
├── 评分标准.txt
├── src/                               # 源代码
│   ├── 1_data_prepare.py              # 下载 + 清洗数据
│   ├── 2_vectorize_bow_tfidf.py       # BOW / TF-IDF 向量化
│   ├── 3_train_word2vec.py            # 训练 Word2Vec 词向量
│   ├── 4_classic_lr.py                # Logistic Regression
│   ├── 4_classic_svm.py               # Linear SVM
│   ├── 5_bilstm.py                    # BiLSTM 深度学习模型
│   ├── 6_bert.py                      # BERT 微调（加分项）
│   └── 7_evaluate.py                  # 汇总结果 + 可视化
├── data/                              # 运行后生成
│   ├── train.csv
│   ├── test.csv
│   └── agnews_clean_train.csv
│   └── agnews_clean_test.csv
├── models/                            # 运行后生成
│   ├── bow_train.npz / bow_test.npz
│   ├── tfidf_train.npz / tfidf_test.npz
│   ├── w2v_agnews.model
│   ├── lr_bow.pkl / lr_tfidf.pkl / lr_w2v.pkl
│   ├── svm_bow.pkl / svm_tfidf.pkl / svm_w2v.pkl
│   ├── bilstm_best.pt
│   └── bert_best/                     # BERT 模型目录
├── results/                           # 运行后生成
│   ├── lr_results.csv
│   ├── svm_results.csv
│   ├── bilstm_results.csv
│   ├── bert_results.csv
│   ├── all_results.csv                # 汇总表
│   └── metrics_compare.png            # 对比柱状图
└── figures/
    └── metrics_compare.png
```

## 五、运行顺序

```powershell
cd "d:\大三\下\数据仓库和挖掘\大作业\源代码"

# 1. 下载并清洗数据（首次约需 1-2 分钟）
python src/1_data_prepare.py

# 2. 生成 BOW / TF-IDF 向量矩阵
python src/2_vectorize_bow_tfidf.py

# 3. 训练 Word2Vec 词向量（约 30 秒）
python src/3_train_word2vec.py

# 4. 经典机器学习分类
python src/4_classic_lr.py       # Logistic Regression（3 组向量化）
python src/4_classic_svm.py      # Linear SVM（3 组向量化）

# 5. 深度学习分类
python src/5_bilstm.py           # BiLSTM（约 10-20 分钟）

# 6. 可选：BERT 微调（约 30-60 分钟，首次需下载 420MB 模型）
python src/6_bert.py

# 7. 汇总所有结果并生成对比图
python src/7_evaluate.py
```

## 六、各脚本说明

### 1_data_prepare.py

从 GitHub 下载 AG News 原始 CSV，然后对每条文本执行：
- 转小写 → 去除非字母字符 → NLTK 分词 → 去停用词 → WordNet 词形还原

输出 `agnews_clean_train.csv` 和 `agnews_clean_test.csv`。

### 2_vectorize_bow_tfidf.py

用 `sklearn.feature_extraction.text` 将清洗后的文本转换为稀疏矩阵：

| 向量化方式 | 参数 |
|-----------|------|
| CountVectorizer | max_features=50000, ngram_range=(1,2), min_df=5 |
| TfidfVectorizer | 同上 + sublinear_tf=True |

输出稀疏矩阵 `.npz` 和对应 Vectorizer 对象 `.pkl`。

### 3_train_word2vec.py

使用 `gensim.Word2Vec` 在训练集语料上训练 100 维词向量。

| 参数 | 值 |
|------|----|
| vector_size | 100 |
| window | 5 |
| min_count | 5 |
| epochs | 10 |

### 4_classic_lr.py / 4_classic_svm.py

两种经典分类器，分别在 BOW / TF-IDF / Word2Vec 三种向量化上训练，输出 Accuracy / Precision / Recall / F1 / AUC。

- **Logistic Regression**: `sklearn.linear_model.LogisticRegression`
- **Linear SVM**: `sklearn.svm.LinearSVC` + `CalibratedClassifierCV`（用于获得概率预测以计算 AUC）

### 5_bilstm.py

基于 Word2Vec 预训练词向量，构建 BiLSTM 文本分类器：

- 词向量嵌入层（100 维，fine-tune）
- 双向 LSTM（隐藏层 128 维）
- Dropout 0.3 + 全连接层
- 训练集 9:1 划分验证集，保留最佳模型

### 6_bert.py

使用 HuggingFace `bert-base-uncased` 进行微调：

- 序列长度 128，batch_size 16，学习率 2e-5
- 3 个 epoch，warmup 比例 10%
- 输出指标同上 5 项

### 7_evaluate.py

读取 `results/` 下所有 `*_results.csv`，合并为对比表，绘制柱状图，并输出各指标的最佳模型。

## 七、评估指标

| 指标 | 含义 | 公式 |
|------|------|------|
| Accuracy | 整体正确率 | TP+TN / Total |
| Precision | 精确率（macro 平均） | TP / (TP+FP) |
| Recall | 召回率（macro 平均） | TP / (TP+FN) |
| F1-Score | F1 值（macro 平均） | 2·P·R / (P+R) |
| ROC-AUC | ROC 曲线下面积（ovr macro） | sklearn.roc_auc_score |

## 八、实验设计总结

| 向量化方式 | 机器学习算法 | 深度学习算法 |
|-----------|------------|------------|
| Bag-of-Words | Logistic Regression, Linear SVM | - |
| TF-IDF | Logistic Regression, Linear SVM | - |
| Word2Vec | Logistic Regression, Linear SVM | BiLSTM |

> BERT 作为加分项，使用 Transformer 架构微调，不使用上述三种向量化方式。

## 九、常见问题

**Q: `1_data_prepare.py` 报网络错误？**  
确保可访问 GitHub（raw.githubusercontent.com），或手动下载后放入 `data/train.csv`。

**Q: `5_bilstm.py` 训练太慢？**  
减小 `EPOCHS`（6→3）或 `BATCH_SZ`（64→32）。本实验使用 CPU 训练，约 10-20 分钟。

**Q: `6_bert.py` 首次运行报错？**  
首次运行会下载 `bert-base-uncased`（约 420 MB），需联网且磁盘空间充足。可跳过该步骤。

## 十、引用

```
@misc{agnews2026,
  title  = {AG News Text Classification Experiment},
  author = {Data Mining Course Project},
  year   = {2026},
  url    = {https://github.com/mhjabreel/CharCnn_Keras}
}
```
