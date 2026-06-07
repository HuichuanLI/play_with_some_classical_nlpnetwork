import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import jieba
import os
import numpy as np
from elmoformanylangs import Embedder
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# ========== 配置 ==========
DATA_ROOT = "/Users/lhc456/Desktop/nlp课程/play_with_some_classical_nlpnetwork/data"
CODE_ROOT = "/Users/lhc456/Desktop/nlp课程/play_with_some_classical_nlpnetwork"
ELMO_MODEL_PATH = os.path.join(CODE_ROOT, "zhs.model")

TRAIN_PATH = os.path.join(DATA_ROOT, "train.txt")
DEV_PATH = os.path.join(DATA_ROOT, "dev.txt")
TEST_PATH = os.path.join(DATA_ROOT, "test.txt")

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
MAX_LEN = 64
HIDDEN_DIM = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 1. 加载 ELMo 模型 ==========
print("正在加载中文 ELMo 模型...")
embedder = Embedder(ELMO_MODEL_PATH, batch_size=64)  # batch_size 指内部处理的 batch 大小
print("ELMo 模型加载完成")


# ========== 2. 数据加载与分词 ==========
def load_data(train_path, dev_path=None, test_path=None):
    """读取 tsv 文件并返回原始文本和标签"""
    train_df = pd.read_csv(train_path, sep="\t", header=0,
                           names=["sentence", "label"], on_bad_lines="skip")
    train_df = train_df.dropna(subset=["sentence", "label"])
    train_df["label"] = pd.to_numeric(train_df["label"], errors="coerce").fillna(0).astype(int)
    train_df = train_df[train_df["sentence"].str.strip() != ""]

    texts = train_df["sentence"].tolist()
    labels = train_df["label"].tolist()

    # 构造标签映射
    unique_labels = sorted(set(labels))
    label2id = {l: i for i, l in enumerate(unique_labels)}
    id2label = {i: l for l, i in label2id.items()}
    labels = [label2id[l] for l in labels]

    # 若提供 dev/test 则类似处理，这里简化直接使用划分
    return texts, labels, label2id, id2label


texts, labels, label2id, id2label = load_data(TRAIN_PATH, DEV_PATH, TEST_PATH)
num_classes = len(label2id)
print(f"类别数: {num_classes}, 样本数: {len(texts)}")


# 分词并截断
def tokenize(text):
    return jieba.lcut(text.strip())[:MAX_LEN]


all_tokens = [tokenize(t) for t in tqdm(texts, desc="分词中")]


# ========== 3. 提取 ELMo 特征 ==========
# ELMoForManyLangs 的 sents2elmo 接收分词后的句子列表，返回 numpy 数组 (sent_len, 1024)
# 为适应变长，我们取平均池化得到句子向量
def extract_elmo_features(tokenized_sentences, batch_size=64):
    features = []
    for i in tqdm(range(0, len(tokenized_sentences), batch_size), desc="提取 ELMo 特征"):
        batch = tokenized_sentences[i:i + batch_size]
        # 获取 ELMo 向量列表（每个句子是一个 [seq_len, 1024] 的 ndarray）
        vecs = embedder.sents2elmo(batch)
        # 对每个句子取平均池化
        pooled = [v.mean(axis=0) if len(v) > 0 else np.zeros(1024) for v in vecs]
        features.append(np.array(pooled))
    return np.concatenate(features, axis=0)


print("开始提取训练特征...")
X = extract_elmo_features(all_tokens)
y = np.array(labels)
print(f"特征矩阵形状: {X.shape}")

# 划分训练 / 验证集（这里简单用 train_test_split）
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ========== 4. 构建分类器 ==========
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, num_classes=2, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


model = MLPClassifier(1024, HIDDEN_DIM, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# 转换为 Tensor
train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
val_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ========== 5. 训练 & 评估 ==========
for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    train_acc = correct / total

    # 验证
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            pred = logits.argmax(1)
            val_correct += (pred == yb).sum().item()
            val_total += yb.size(0)
    val_acc = val_correct / val_total
    print(
        f"Epoch {epoch + 1:2d} | Loss: {total_loss / len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

# ========== 6. 测试集评估（如果有） ==========
if os.path.exists(TEST_PATH):
    test_df = pd.read_csv(TEST_PATH, sep="\t", header=0, names=["sentence", "label"], on_bad_lines="skip")
    test_df = test_df.dropna(subset=["sentence", "label"])
    test_texts = test_df["sentence"].tolist()
    test_labels = [label2id.get(l, 0) for l in test_df["label"].tolist()]
    test_tokens = [tokenize(t) for t in test_texts]
    X_test = extract_elmo_features(test_tokens)
    y_test = np.array(test_labels)

    test_set = torch.utils.data.TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            all_preds.extend(logits.argmax(1).cpu().numpy())
    print("\n测试集报告:")
    print(classification_report(y_test, all_preds, target_names=[str(id2label[i]) for i in range(num_classes)]))


# ========== 7. 单句预测 ==========
def predict(text):
    tokens = tokenize(text)
    vec = embedder.sents2elmo([tokens])[0]
    pooled = vec.mean(axis=0) if len(vec) > 0 else np.zeros(1024)
    x = torch.FloatTensor(pooled).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        pred_id = logits.argmax(1).item()
        prob = torch.softmax(logits, dim=1)[0, pred_id].item()
    return id2label[pred_id], prob


print("\n预测示例:")
for text in ["今天天气真好", "这部电影太糟糕了", "机器学习很有意思"]:
    label, prob = predict(text)
    print(f"文本：{text}  →  类别：{label} (置信度：{prob:.4f})")
