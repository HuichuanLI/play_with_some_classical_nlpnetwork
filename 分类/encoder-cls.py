import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import jieba
from collections import Counter
import math
import copy
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


# ====================== 1. 数据处理（新增测试集支持）=====================
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=64):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        try:
            text = str(self.texts[idx]).strip()
            label = int(self.labels[idx])
            tokens = jieba.lcut(text)
            ids = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
            if len(ids) > self.max_len:
                ids = ids[:self.max_len]
            else:
                ids += [self.vocab["<PAD>"]] * (self.max_len - len(ids))
            return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"❌ 第{idx}条数据出错：{e}")
            return torch.zeros(self.max_len, dtype=torch.long), torch.tensor(0, dtype=torch.long)


def load_data(train_path, dev_path=None, test_path=None, min_freq=1):
    """
    加载数据：
    - 如果提供dev_path，使用独立验证集
    - 如果提供test_path，使用独立测试集
    - 词汇表和标签映射仅从训练集构建
    """
    print(f"正在加载训练集：{train_path}")
    train_df = pd.read_csv(train_path, sep="\t", header=0, names=["sentence", "label"], on_bad_lines="skip")
    train_df = train_df.dropna(subset=["sentence", "label"])
    train_df["label"] = pd.to_numeric(train_df["label"], errors="coerce").fillna(0).astype(int)
    train_df = train_df[train_df["sentence"].str.strip() != ""]
    print(f"训练集有效行数：{len(train_df)}")

    # 仅从训练集构建标签映射
    train_raw_labels = train_df["label"].tolist()
    unique_labels = sorted(list(set(train_raw_labels)))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    print(f"\n检测到类别数：{num_classes}")
    print(f"标签映射：{label2id}")

    # 转换标签
    train_labels = [label2id[label] for label in train_raw_labels]
    train_texts = train_df["sentence"].tolist()

    # 仅从训练集构建词汇表
    print("\n构建词汇表...")
    all_tokens = []
    for text in train_texts:
        all_tokens.extend(jieba.lcut(str(text).strip()))
    counter = Counter(all_tokens)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)
    print(f"词汇表大小：{len(vocab)}")

    # 加载验证集
    dev_texts, dev_labels = [], []
    if dev_path:
        print(f"\n正在加载验证集：{dev_path}")
        dev_df = pd.read_csv(dev_path, sep="\t", header=0, names=["sentence", "label"], on_bad_lines="skip")
        dev_df = dev_df.dropna(subset=["sentence", "label"])
        dev_df["label"] = pd.to_numeric(dev_df["label"], errors="coerce").fillna(0).astype(int)
        dev_df = dev_df[dev_df["sentence"].str.strip() != ""]
        print(f"验证集有效行数：{len(dev_df)}")

        dev_texts = dev_df["sentence"].tolist()
        dev_raw_labels = dev_df["label"].tolist()
        # 使用训练集的标签映射
        dev_labels = [label2id.get(label, 0) for label in dev_raw_labels]

    # 加载测试集
    test_texts, test_labels = [], []
    if test_path:
        print(f"\n正在加载测试集：{test_path}")
        test_df = pd.read_csv(test_path, sep="\t", header=0, names=["sentence", "label"], on_bad_lines="skip")
        test_df = test_df.dropna(subset=["sentence", "label"])
        test_df["label"] = pd.to_numeric(test_df["label"], errors="coerce").fillna(0).astype(int)
        test_df = test_df[test_df["sentence"].str.strip() != ""]
        print(f"测试集有效行数：{len(test_df)}")

        test_texts = test_df["sentence"].tolist()
        test_raw_labels = test_df["label"].tolist()
        # 使用训练集的标签映射
        test_labels = [label2id.get(label, 0) for label in test_raw_labels]

    return (train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels,
            vocab, label2id, id2label, num_classes)


# ====================== 2. Transformer Encoder核心组件（完全不变）=====================
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn_weights = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn_weights = dropout(attn_weights)
    return torch.matmul(attn_weights, value), attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)

        query, key, value = [
            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears[:3], (query, key, value))
        ]

        output, _ = scaled_dot_product_attention(query, key, value, mask, self.dropout)

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.linears[-1](output)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout=0.1):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model, padding_idx=0)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ====================== 3. Transformer Encoder分类模型（完全不变）=====================
class TransformerEncoderCls(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, d_ff=512, dropout=0.1, num_classes=2):
        super().__init__()
        self.embedding = Embeddings(d_model, vocab_size)
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        attn = MultiHeadAttention(nhead, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)

        encoder_layer = EncoderLayer(d_model, attn, ff, dropout)
        self.encoder = Encoder(encoder_layer, num_layers)

        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len = x.size()
        mask = (x != 0).unsqueeze(1)

        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.encoder(x, mask)
        x = torch.mean(x, dim=1)
        x = self.dropout(x)
        return self.fc(x)


# ====================== 4. 训练&评估（新增完整测试函数）=====================
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    print("\n开始训练...")
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        pred = torch.argmax(outputs, dim=1)
        batch_correct = (pred == y).sum().item()
        batch_acc = batch_correct / len(y)

        total_loss += loss.item()
        total_correct += batch_correct
        total_samples += len(y)

        print(f"  批次 {batch_idx + 1}/{len(dataloader)} | 损失：{loss.item():.4f} | 准确率：{batch_acc:.4f}")

    epoch_loss = total_loss / len(dataloader)
    epoch_acc = total_correct / total_samples if total_samples > 0 else 0
    print(f"\n训练汇总 | 平均损失：{epoch_loss:.4f} | 平均准确率：{epoch_acc:.4f}")
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device, dataset_name="验证集"):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    print(f"\n开始{dataset_name}评估...")
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)

            pred = torch.argmax(outputs, dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            total_loss += loss.item()

            print(f"  处理批次 {batch_idx + 1}/{len(dataloader)}")

    epoch_loss = total_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    print(f"\n{dataset_name}汇总 | 平均损失：{epoch_loss:.4f} | 准确率：{epoch_acc:.4f}")
    return epoch_loss, epoch_acc, all_preds, all_labels


def test_model(model, test_loader, criterion, device, id2label, save_report=True):
    """完整的测试集评估，输出详细报告和混淆矩阵"""
    test_loss, test_acc, all_preds, all_labels = evaluate(
        model, test_loader, criterion, device, dataset_name="测试集"
    )

    # 生成分类报告
    target_names = [str(id2label[i]) for i in range(len(id2label))]
    report = classification_report(
        all_labels, all_preds, target_names=target_names, digits=4
    )

    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    print("\n" + "=" * 60)
    print("测试集详细评估报告")
    print("=" * 60)
    print(report)
    print("混淆矩阵：")
    print(cm)
    print("=" * 60)

    # 保存报告和混淆矩阵
    if save_report:
        with open("test_report.txt", "w", encoding="utf-8") as f:
            f.write(f"测试集准确率：{test_acc:.4f}\n\n")
            f.write("分类报告：\n")
            f.write(report)
            f.write("\n混淆矩阵：\n")
            f.write(str(cm))
        print("\n✅ 测试报告已保存为 test_report.txt")

        # 绘制并保存混淆矩阵图
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=target_names, yticklabels=target_names)
        plt.xlabel("预测标签")
        plt.ylabel("真实标签")
        plt.title("测试集混淆矩阵")
        plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
        print("✅ 混淆矩阵图已保存为 confusion_matrix.png")

    return test_loss, test_acc, report, cm


# ====================== 5. 主函数（新增测试集流程）=====================
if __name__ == "__main__":
    # 配置
    TRAIN_PATH = "/Users/lhc456/Desktop/nlp课程/play_with_some_classical_nlpnetwork/data/train.txt"
    DEV_PATH = "/Users/lhc456/Desktop/nlp课程/play_with_some_classical_nlpnetwork/data/dev.txt"
    TEST_PATH = "/Users/lhc456/Desktop/nlp课程/play_with_some_classical_nlpnetwork/data/test.txt"  # 你的测试集路径

    BATCH_SIZE = 1000
    EPOCHS = 5
    LR = 5e-4
    MAX_LEN = 64

    # Transformer Encoder超参数
    D_MODEL = 128
    NHEAD = 4
    NUM_LAYERS = 2
    D_FF = 512
    DROPOUT = 0.1

    # 设备配置
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用 NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("使用 Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("使用 CPU")

    # 加载数据（包含训练集、验证集、测试集）
    try:
        (train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels,
         vocab, label2id, id2label, num_classes) = load_data(TRAIN_PATH, DEV_PATH, TEST_PATH)
    except Exception as e:
        print(f"❌ 数据加载失败：{e}")
        exit()

    # 构建数据集和加载器
    print("\n构建数据集...")
    train_dataset = TextDataset(train_texts, train_labels, vocab, MAX_LEN)
    dev_dataset = TextDataset(dev_texts, dev_labels, vocab, MAX_LEN)
    test_dataset = TextDataset(test_texts, test_labels, vocab, MAX_LEN)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=False
    )

    print(f"训练集批次数量：{len(train_loader)}")
    print(f"验证集批次数量：{len(dev_loader)}")
    print(f"测试集批次数量：{len(test_loader)}")

    # 初始化模型
    print("\n初始化模型...")
    model = TransformerEncoderCls(
        len(vocab), D_MODEL, NHEAD, NUM_LAYERS, D_FF, DROPOUT, num_classes
    ).to(device)
    print(f"模型参数量：{sum(p.numel() for p in model.parameters())}")

    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    best_dev_acc = 0
    print("\n" + "=" * 60)
    print("开始训练循环")
    print("=" * 60)

    for epoch in range(EPOCHS):
        print(f"\n{'=' * 25} Epoch {epoch + 1}/{EPOCHS} {'=' * 25}")
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        dev_loss, dev_acc, _, _ = evaluate(model, dev_loader, criterion, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab": vocab,
                "label2id": label2id,
                "id2label": id2label,
                "config": {
                    "D_MODEL": D_MODEL, "NHEAD": NHEAD, "NUM_LAYERS": NUM_LAYERS,
                    "D_FF": D_FF, "DROPOUT": DROPOUT, "MAX_LEN": MAX_LEN
                }
            }, "best_transformer_encoder_cls_model.pth")
            print("\n✅ 保存最佳模型")

    print("\n" + "=" * 60)
    print(f"训练完成！最佳验证准确率：{best_dev_acc:.4f}")
    print("=" * 60)

    # 加载最佳模型进行测试集评估
    print("\n" + "=" * 60)
    print("开始测试集评估")
    print("=" * 60)

    checkpoint = torch.load("best_transformer_encoder_cls_model.pth")
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_report, test_cm = test_model(
        model, test_loader, criterion, device, id2label
    )


    # 单句预测
    def predict(text):
        tokens = jieba.lcut(text.strip())
        ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
        if len(ids) > MAX_LEN:
            ids = ids[:MAX_LEN]
        else:
            ids += [vocab["<PAD>"]] * (MAX_LEN - len(ids))
        x = torch.tensor([ids], dtype=torch.long).to(device)

        with torch.no_grad():
            output = model(x)
            pred_id = torch.argmax(output, dim=1).item()
            prob = F.softmax(output, dim=1)[0][pred_id].item()
        return id2label[pred_id], prob


    # 测试预测
    test_texts = [
        "中华女子学院：本科层次仅1专业招男生",
        "两天价网站背后重重迷雾：做个网站究竟要多少钱",
        "东5环海棠公社230-290平2居准现房98折优惠"
    ]

    print("\n预测测试：")
    for text in test_texts:
        pred, prob = predict(text)
        print(f"文本：{text}")
        print(f"预测类别：{pred}，置信度：{prob:.4f}\n")
