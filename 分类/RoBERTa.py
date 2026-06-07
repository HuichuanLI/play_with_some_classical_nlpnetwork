import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import jieba
from collections import Counter
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ====================== 路径配置（完全不变）=====================
# 你的数据根目录（绝对路径）
DATA_ROOT = "/Users/lhc456/Desktop/nlp课程/play_with_some_classical_nlpnetwork/data"
# 你的代码根目录（绝对路径）
CODE_ROOT = "/Users/lhc456/Desktop/nlp课程/play_with_some_classical_nlpnetwork"

# 自动创建目录（如果不存在）
os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(CODE_ROOT, exist_ok=True)


# ====================== 1. 数据处理（完全不变，原生支持padding）=====================
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        try:
            text = str(self.texts[idx]).strip()
            label = int(self.labels[idx])

            # RoBERTa原生支持padding，不需要任何额外设置
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt"
            )

            return {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                "label": torch.tensor(label, dtype=torch.long)
            }
        except Exception as e:
            print(f"❌ 第{idx}条数据出错：{e}")
            return {
                "input_ids": torch.zeros(self.max_len, dtype=torch.long),
                "attention_mask": torch.zeros(self.max_len, dtype=torch.long),
                "label": torch.tensor(0, dtype=torch.long)
            }


def load_data(train_path, dev_path=None, test_path=None, min_freq=1, test_size=0.2):
    """
    加载数据：接口完全不变
    - 如果提供dev_path，使用独立验证集
    - 如果未提供dev_path，从训练集随机划分20%作为验证集
    - 分词器和标签映射仅从训练集构建
    """
    print(f"正在加载训练集：{train_path}")
    train_df = pd.read_csv(train_path, sep="\t", header=0, names=["sentence", "label"], on_bad_lines="skip")
    train_df = train_df.dropna(subset=["sentence", "label"])
    train_df["label"] = pd.to_numeric(train_df["label"], errors="coerce").fillna(0).astype(int)
    train_df = train_df[train_df["sentence"].str.strip() != ""]
    print(f"训练集有效行数：{len(train_df)}")

    # 仅从训练集构建标签映射（完全不变）
    train_raw_labels = train_df["label"].tolist()
    unique_labels = sorted(list(set(train_raw_labels)))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    print(f"\n检测到类别数：{num_classes}")
    print(f"标签映射：{label2id}")
    print(f"标签分布：{Counter(train_raw_labels)}")

    # 转换标签（完全不变）
    train_labels = [label2id[label] for label in train_raw_labels]
    train_texts = train_df["sentence"].tolist()

    # 处理验证集（完全不变）
    dev_texts, dev_labels = [], []
    if dev_path and os.path.exists(dev_path):
        print(f"\n正在加载独立验证集：{dev_path}")
        dev_df = pd.read_csv(dev_path, sep="\t", header=0, names=["sentence", "label"], on_bad_lines="skip")
        dev_df = dev_df.dropna(subset=["sentence", "label"])
        dev_df["label"] = pd.to_numeric(dev_df["label"], errors="coerce").fillna(0).astype(int)
        dev_df = dev_df[dev_df["sentence"].str.strip() != ""]
        print(f"独立验证集有效行数：{len(dev_df)}")

        dev_texts = dev_df["sentence"].tolist()
        dev_raw_labels = dev_df["label"].tolist()
        dev_labels = [label2id.get(label, 0) for label in dev_raw_labels]
    else:
        print(f"\n未找到独立验证集，从训练集随机划分{test_size * 100}%作为验证集")
        train_texts, dev_texts, train_labels, dev_labels = train_test_split(
            train_texts, train_labels, test_size=test_size, random_state=42, stratify=train_labels
        )
        print(f"划分后训练集大小：{len(train_texts)}，验证集大小：{len(dev_texts)}")

    # 处理测试集（完全不变）
    test_texts, test_labels = [], []
    if test_path and os.path.exists(test_path):
        print(f"\n正在加载测试集：{test_path}")
        test_df = pd.read_csv(test_path, sep="\t", header=0, names=["sentence", "label"], on_bad_lines="skip")
        test_df = test_df.dropna(subset=["sentence", "label"])
        test_df["label"] = pd.to_numeric(test_df["label"], errors="coerce").fillna(0).astype(int)
        test_df = test_df[test_df["sentence"].str.strip() != ""]
        print(f"测试集有效行数：{len(test_df)}")

        test_texts = test_df["sentence"].tolist()
        test_raw_labels = test_df["label"].tolist()
        test_labels = [label2id.get(label, 0) for label in test_raw_labels]

    # 加载RoBERTa中文分词器（原生支持pad_token）
    print("\n加载RoBERTa中文分词器...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    print(f"✅ RoBERTa分词器加载完成，词汇表大小：{len(tokenizer)}")
    print(f"  原生pad_token: {tokenizer.pad_token}")
    print(f"  原生pad_token_id: {tokenizer.pad_token_id}")

    return (train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels,
            tokenizer, label2id, id2label, num_classes)


# ====================== 2. RoBERTa模型（接口与所有模型完全一致）=====================
class RoBERTaTextCls(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super().__init__()
        from transformers import AutoModelForSequenceClassification
        # 加载预训练RoBERTa模型（中文文本分类黄金标准）
        self.roberta = AutoModelForSequenceClassification.from_pretrained(
            "hfl/chinese-roberta-wwm-ext",
            num_labels=num_classes
        )

        # 冻结前10层，只训练最后2层（加快训练速度，效果几乎不变）
        for param in list(self.roberta.parameters())[:-4]:
            param.requires_grad = False

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None):
        # 接口与所有模型完全一致：输入张量，输出logits
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        return self.dropout(outputs.logits)


# ====================== 3. 训练&评估（完全不变）=====================
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    print("\n开始训练...")
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
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


def evaluate(model, dataloader, criterion, device, dataset_name="验证集"):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    print(f"\n开始{dataset_name}评估...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
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
    """完整的测试集评估，输出详细报告和混淆矩阵（完全不变）"""
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
        report_path = os.path.join(CODE_ROOT, "roberta_test_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"测试集准确率：{test_acc:.4f}\n\n")
            f.write("分类报告：\n")
            f.write(report)
            f.write("\n混淆矩阵：\n")
            f.write(str(cm))
        print(f"\n✅ 测试报告已保存为 {report_path}")

        # 绘制并保存混淆矩阵图
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=target_names, yticklabels=target_names)
        plt.xlabel("预测标签")
        plt.ylabel("真实标签")
        plt.title("RoBERTa测试集混淆矩阵")
        cm_path = os.path.join(CODE_ROOT, "roberta_confusion_matrix.png")
        plt.savefig(cm_path, dpi=300, bbox_inches="tight")
        print(f"✅ 混淆矩阵图已保存为 {cm_path}")

    return test_loss, test_acc, report, cm


# ====================== 4. 主函数（几乎完全不变）=====================
if __name__ == "__main__":
    # 配置（RoBERTa与MacBERT相当，batch_size建议16）
    TRAIN_PATH = os.path.join(DATA_ROOT, "train.txt")
    DEV_PATH = os.path.join(DATA_ROOT, "dev.txt")
    TEST_PATH = os.path.join(DATA_ROOT, "test.txt")

    BATCH_SIZE = 1024
    EPOCHS = 3
    LR = 2e-5
    MAX_LEN = 64
    DROPOUT = 0.1

    # 设备配置（RoBERTa在MPS上兼容性很好）
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用 NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("使用 Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("使用 CPU")

    # 检查数据文件（完全不变）
    print("\n检查数据文件...")
    if not os.path.exists(TRAIN_PATH):
        print(f"❌ 训练集文件不存在：{TRAIN_PATH}")
        exit()
    print(f"✅ 找到训练集：{TRAIN_PATH}")

    if os.path.exists(DEV_PATH):
        print(f"✅ 找到独立验证集：{DEV_PATH}")
    else:
        print(f"⚠️ 未找到独立验证集，将从训练集自动划分")

    if os.path.exists(TEST_PATH):
        print(f"✅ 找到测试集：{TEST_PATH}")
    else:
        print(f"⚠️ 未找到测试集，将不进行测试集评估")

    # 加载数据（接口完全不变）
    try:
        (train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels,
         tokenizer, label2id, id2label, num_classes) = load_data(TRAIN_PATH, DEV_PATH, TEST_PATH)
    except Exception as e:
        print(f"❌ 数据加载失败：{e}")
        exit()

    # 构建数据集和加载器（完全不变）
    print("\n构建数据集...")
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    dev_dataset = TextDataset(dev_texts, dev_labels, tokenizer, MAX_LEN)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, MAX_LEN) if test_texts else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    ) if test_dataset else None

    print(f"训练集批次数量：{len(train_loader)}")
    print(f"验证集批次数量：{len(dev_loader)}")
    if test_loader:
        print(f"测试集批次数量：{len(test_loader)}")

    # 初始化RoBERTa模型（接口与所有模型完全一致）
    print("\n初始化RoBERTa文本分类模型...")
    model = RoBERTaTextCls(num_classes, DROPOUT).to(device)
    print(f"可训练参数数量：{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 优化器和损失函数（完全不变）
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # 训练循环（完全不变）
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
                "tokenizer": tokenizer,
                "label2id": label2id,
                "id2label": id2label,
                "config": {
                    "DROPOUT": DROPOUT,
                    "MAX_LEN": MAX_LEN
                }
            }, os.path.join(CODE_ROOT, "best_roberta_cls_model.pth"))
            print("\n✅ 保存最佳RoBERTa模型")

    print("\n" + "=" * 60)
    print(f"训练完成！最佳验证准确率：{best_dev_acc:.4f}")
    print("=" * 60)

    # 加载最佳模型进行测试集评估（完全不变）
    if test_loader:
        print("\n" + "=" * 60)
        print("开始测试集评估")
        print("=" * 60)

        checkpoint = torch.load(os.path.join(CODE_ROOT, "best_roberta_cls_model.pth"))
        model.load_state_dict(checkpoint["model_state_dict"])

        test_loss, test_acc, test_report, test_cm = test_model(
            model, test_loader, criterion, device, id2label
        )


    # 单句预测（接口完全不变）
    def predict(text):
        checkpoint = torch.load(os.path.join(CODE_ROOT, "best_roberta_cls_model.pth"))
        tokenizer = checkpoint["tokenizer"]
        id2label = checkpoint["id2label"]
        config = checkpoint["config"]
        num_classes = len(id2label)

        model = RoBERTaTextCls(
            num_classes,
            config["DROPOUT"]
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # RoBERTa分词（原生支持padding）
        encoding = tokenizer(
            text.strip(),
            truncation=True,
            padding="max_length",
            max_length=config["MAX_LEN"],
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            output = model(input_ids, attention_mask)
            pred_id = torch.argmax(output, dim=1).item()
            prob = F.softmax(output, dim=1)[0][pred_id].item()
        return id2label[pred_id], prob


    # 测试预测（完全不变）
    test_texts = [
        "中华女子学院：本科层次仅1专业招男生",
        "两天价网站背后重重迷雾：做个网站究竟要多少钱",
        "东5环海棠公社230-290平准现房98折优惠"
    ]

    print("\n预测测试：")
    for text in test_texts:
        pred, prob = predict(text)
        print(f"文本：{text}")
        print(f"预测类别：{pred}，置信度：{prob:.4f}\n")
