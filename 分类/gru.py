import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import jieba
from collections import Counter
import warnings

warnings.filterwarnings("ignore")


# ====================== 1. 数据处理 =====================
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


def load_data(file_path, min_freq=1):
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

    unique_labels = sorted(list(set(raw_labels)))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}
    labels = [label2id[label] for label in raw_labels]
    num_classes = len(unique_labels)
    print(f"\n检测到类别数：{num_classes}")
    print(f"标签分布：{Counter(raw_labels)}")
    print(f"标签映射：{label2id}")

    train_texts, dev_texts = texts, texts[:3]
    train_labels, dev_labels = labels, labels[:3]

    print(f"\n训练集大小：{len(train_texts)}")
    print(f"验证集大小：{len(dev_texts)}")

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


# ====================== 2. GRU模型 =====================
class GRUTextCls(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64, num_layers=1, dropout=0.1, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embed = self.dropout(self.embedding(x))
        out, _ = self.gru(embed)
        out = torch.mean(out, dim=1)
        out = self.dropout(out)
        return self.fc(out)


# ====================== 3. 训练&评估（新增批次级准确率打印）====================
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

        # 计算当前批次的准确率
        pred = torch.argmax(outputs, dim=1)
        batch_correct = (pred == y).sum().item()
        batch_acc = batch_correct / len(y)

        # 累加全局统计
        total_loss += loss.item()
        total_correct += batch_correct
        total_samples += len(y)

        # 实时打印批次信息
        print(f"  批次 {batch_idx + 1}/{len(dataloader)} | 损失：{loss.item():.4f} | 准确率：{batch_acc:.4f}")

    epoch_loss = total_loss / len(dataloader)
    epoch_acc = total_correct / total_samples if total_samples > 0 else 0
    print(f"\n训练汇总 | 平均损失：{epoch_loss:.4f} | 平均准确率：{epoch_acc:.4f}")
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    print("\n开始验证...")
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)

            # 计算当前批次的准确率
            pred = torch.argmax(outputs, dim=1)
            batch_correct = (pred == y).sum().item()
            batch_acc = batch_correct / len(y)

            # 累加全局统计
            total_loss += loss.item()
            total_correct += batch_correct
            total_samples += len(y)

            # 实时打印批次信息
            print(f"  批次 {batch_idx + 1}/{len(dataloader)} | 损失：{loss.item():.4f} | 准确率：{batch_acc:.4f}")

    epoch_loss = total_loss / len(dataloader)
    epoch_acc = total_correct / total_samples if total_samples > 0 else 0
    print(f"\n验证汇总 | 平均损失：{epoch_loss:.4f} | 平均准确率：{epoch_acc:.4f}")
    return epoch_loss, epoch_acc


# ====================== 4. 主函数 =====================
if __name__ == "__main__":
    # 配置
    DATA_PATH = "/data/train.txt"  # 替换成你的数据文件路径
    BATCH_SIZE = 1000
    EPOCHS = 5
    LR = 5e-4
    MAX_LEN = 64
    EMBED_DIM = 32
    HIDDEN_DIM = 64
    NUM_LAYERS = 1
    DROPOUT = 0.1

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    # 加载数据
    try:
        (train_texts, train_labels, dev_texts, dev_labels,
         vocab, label2id, id2label, num_classes) = load_data(DATA_PATH)
    except Exception as e:
        print(f"❌ 数据加载失败：{e}")
        exit()

    # 构建数据集和加载器（macOS强制单进程）
    print("\n构建数据集...")
    train_dataset = TextDataset(train_texts, train_labels, vocab, MAX_LEN)
    dev_dataset = TextDataset(dev_texts, dev_labels, vocab, MAX_LEN)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # 关键！macOS必须设为0
        pin_memory=False
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    print(f"训练集批次数量：{len(train_loader)}")
    print(f"验证集批次数量：{len(dev_loader)}")

    # 初始化模型
    print("\n初始化模型...")
    model = GRUTextCls(len(vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, num_classes).to(device)
    print(f"模型参数量：{sum(p.numel() for p in model.parameters())}")

    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    best_acc = 0
    print("\n" + "=" * 60)
    print("开始训练循环")
    print("=" * 60)

    for epoch in range(EPOCHS):
        print(f"\n{'=' * 25} Epoch {epoch + 1}/{EPOCHS} {'=' * 25}")
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        dev_loss, dev_acc = evaluate(model, dev_loader, criterion, device)

        if dev_acc > best_acc:
            best_acc = dev_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab": vocab,
                "label2id": label2id,
                "id2label": id2label,
                "config": {
                    "EMBED_DIM": EMBED_DIM,
                    "HIDDEN_DIM": HIDDEN_DIM,
                    "NUM_LAYERS": NUM_LAYERS,
                    "DROPOUT": DROPOUT,
                    "MAX_LEN": MAX_LEN
                }
            }, "best_gru_cls_model.pth")
            print("\n✅ 保存最佳模型")

    print("\n" + "=" * 60)
    print(f"训练完成！最佳验证准确率：{best_acc:.4f}")
    print("=" * 60)


    # 单句预测
    def predict(text):
        checkpoint = torch.load("best_gru_cls_model.pth")
        vocab = checkpoint["vocab"]
        id2label = checkpoint["id2label"]
        config = checkpoint["config"]
        num_classes = len(id2label)

        model = GRUTextCls(
            len(vocab),
            config["EMBED_DIM"],
            config["HIDDEN_DIM"],
            config["NUM_LAYERS"],
            config["DROPOUT"],
            num_classes
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        tokens = jieba.lcut(text.strip())
        ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
        if len(ids) > config["MAX_LEN"]:
            ids = ids[:config["MAX_LEN"]]
        else:
            ids += [vocab["<PAD>"]] * (config["MAX_LEN"] - len(ids))
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
