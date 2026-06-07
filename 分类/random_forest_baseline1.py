import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import jieba
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")


# ====================== 1. 数据处理（与深度学习模型完全一致）=====================
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab=None, max_len=64, vectorizer=None):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        self.vectorizer = vectorizer  # 随机森林专用：TF-IDF向量化器

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        try:
            text = str(self.texts[idx]).strip()
            label = int(self.labels[idx])

            # 随机森林返回TF-IDF特征
            if self.vectorizer is not None:
                return text, label

            # 深度学习模型返回token ids
            tokens = jieba.lcut(text)
            ids = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
            if len(ids) > self.max_len:
                ids = ids[:self.max_len]
            else:
                ids += [self.vocab["<PAD>"]] * (self.max_len - len(ids))
            return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"❌ 第{idx}条数据出错：{e}")
            if self.vectorizer is not None:
                return "", 0
            return torch.zeros(self.max_len, dtype=torch.long), torch.tensor(0, dtype=torch.long)


def load_data(file_path, min_freq=1, test_size=0.2):
    print(f"正在加载数据：{file_path}")
    df = pd.read_csv(file_path, sep="\t", header=0, names=["sentence", "label"], on_bad_lines="skip")
    print(f"原始数据行数：{len(df)}")

    df = df.dropna(subset=["sentence", "label"])
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    df = df[df["sentence"].str.strip() != ""]
    print(f"过滤后有效数据行数：{len(df)}")

    if len(df) < 5:
        print("\n⚠️ 警告：数据量太少！至少需要5条数据才能正常训练")
        exit()

    texts = df["sentence"].tolist()
    raw_labels = df["label"].tolist()

    # 标签映射（与深度学习模型完全一致）
    unique_labels = sorted(list(set(raw_labels)))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}
    labels = [label2id[label] for label in raw_labels]
    num_classes = len(unique_labels)
    print(f"\n检测到类别数：{num_classes}")
    print(f"标签分布：{Counter(raw_labels)}")
    print(f"标签映射：{label2id}")

    # 划分训练集验证集
    from sklearn.model_selection import train_test_split
    train_texts, dev_texts, train_labels, dev_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )

    print(f"\n训练集大小：{len(train_texts)}")
    print(f"验证集大小：{len(dev_texts)}")

    # 构建词汇表（用于深度学习模型）
    all_tokens = []
    for text in train_texts:
        all_tokens.extend(jieba.lcut(str(text).strip()))
    counter = Counter(all_tokens)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)
    print(f"词汇表大小：{len(vocab)}")

    return (train_texts, train_labels, dev_texts, dev_labels,
            vocab, label2id, id2label, num_classes)


# ====================== 2. 随机森林文本分类模型 =====================
class RandomForestTextCls:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, random_state=42):
        # TF-IDF向量化器：使用jieba分词
        self.vectorizer = TfidfVectorizer(
            tokenizer=jieba.lcut,
            max_features=10000,  # 保留最常见的10000个词
            ngram_range=(1, 2)  # 同时考虑1-gram和2-gram
        )
        # 随机森林分类器
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1  # 使用所有CPU核心
        )
        self.label2id = None
        self.id2label = None

    def train(self, train_texts, train_labels, dev_texts, dev_labels):
        print("\n开始训练随机森林模型...")

        # 训练TF-IDF向量化器
        print("  训练TF-IDF向量化器...")
        train_features = self.vectorizer.fit_transform(train_texts)
        dev_features = self.vectorizer.transform(dev_texts)

        print(f"  TF-IDF特征维度：{train_features.shape[1]}")

        # 训练随机森林
        print("  训练随机森林分类器...")
        self.model.fit(train_features, train_labels)

        # 评估
        train_pred = self.model.predict(train_features)
        dev_pred = self.model.predict(dev_features)

        train_acc = accuracy_score(train_labels, train_pred)
        dev_acc = accuracy_score(dev_labels, dev_pred)

        print(f"\n训练完成！")
        print(f"训练准确率：{train_acc:.4f}")
        print(f"验证准确率：{dev_acc:.4f}")

        return train_acc, dev_acc

    def predict(self, text):
        # 单句预测
        features = self.vectorizer.transform([text])
        pred_id = self.model.predict(features)[0]
        prob = self.model.predict_proba(features)[0][pred_id]
        return pred_id, prob

    def save(self, path="best_rf_cls_model.pkl"):
        import joblib
        joblib.dump({
            "vectorizer": self.vectorizer,
            "model": self.model,
            "label2id": self.label2id,
            "id2label": self.id2label
        }, path)
        print(f"\n✅ 随机森林模型已保存为 {path}")

    @classmethod
    def load(cls, path="best_rf_cls_model.pkl"):
        import joblib
        checkpoint = joblib.load(path)
        rf = cls()
        rf.vectorizer = checkpoint["vectorizer"]
        rf.model = checkpoint["model"]
        rf.label2id = checkpoint["label2id"]
        rf.id2label = checkpoint["id2label"]
        return rf


# ====================== 3. 主函数 =====================
if __name__ == "__main__":
    # 配置
    DATA_PATH = "/data/train.txt"
    TEST_SIZE = 0.2

    # 随机森林超参数
    N_ESTIMATORS = 200  # 决策树数量
    MAX_DEPTH = None  # 树的最大深度，None表示不限制
    MIN_SAMPLES_SPLIT = 2

    # 加载数据
    try:
        (train_texts, train_labels, dev_texts, dev_labels,
         vocab, label2id, id2label, num_classes) = load_data(DATA_PATH, test_size=TEST_SIZE)
    except Exception as e:
        print(f"❌ 数据加载失败：{e}")
        exit()

    # 初始化并训练随机森林
    rf_model = RandomForestTextCls(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT
    )
    rf_model.label2id = label2id
    rf_model.id2label = id2label

    train_acc, dev_acc = rf_model.train(train_texts, train_labels, dev_texts, dev_labels)

    # 保存模型
    # rf_model.save()

    # 测试预测
    test_texts = [
        "中华女子学院：本科层次仅1专业招男生",
        "两天价网站背后重重迷雾：做个网站究竟要多少钱",
        "东5环海棠公社230-290平2居准现房98折优惠"
    ]

    print("\n预测测试：")
    for text in test_texts:
        pred_id, prob = rf_model.predict(text)
        pred_label = id2label[pred_id]
        print(f"文本：{text}")
        print(f"预测类别：{pred_label}，置信度：{prob:.4f}\n")
