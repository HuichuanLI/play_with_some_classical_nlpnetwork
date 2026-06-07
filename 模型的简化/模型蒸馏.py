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
import time

warnings.filterwarnings("ignore")

# ====================== 路径配置（保持不变）======================
DATA_ROOT = "/Users/lhc456/Desktop/nlp课程/play_with_some_classical_nlpnetwork/data"
CODE_ROOT = "/Users/lhc456/Desktop/nlp课程/play_with_some_classical_nlpnetwork"

os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(CODE_ROOT, exist_ok=True)


# ====================== 1. 蒸馏专用数据集（同时支持BERT和TextCNN）======================
class DistillationDataset(Dataset):
    def __init__(self, texts, labels, bert_tokenizer, cnn_vocab, max_len=64):
        self.texts = texts
        self.labels = labels
        self.bert_tokenizer = bert_tokenizer
        self.cnn_vocab = cnn_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        try:
            text = str(self.texts[idx]).strip()
            label = int(self.labels[idx])

            # BERT格式输入（教师模型用）
            bert_encoding = self.bert_tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt"
            )

            # TextCNN格式输入（学生模型用）
            tokens = jieba.lcut(text)
            cnn_ids = [self.cnn_vocab.get(token, self.cnn_vocab["<UNK>"]) for token in tokens]
            if len(cnn_ids) > self.max_len:
                cnn_ids = cnn_ids[:self.max_len]
            else:
                cnn_ids += [self.cnn_vocab["<PAD>"]] * (self.max_len - len(cnn_ids))

            return {
                "bert_input_ids": bert_encoding["input_ids"].flatten(),
                "bert_attention_mask": bert_encoding["attention_mask"].flatten(),
                "cnn_input_ids": torch.tensor(cnn_ids, dtype=torch.long),
                "label": torch.tensor(label, dtype=torch.long)
            }
        except Exception as e:
            print(f"❌ 第{idx}条数据出错：{e}")
            return {
                "bert_input_ids": torch.zeros(self.max_len, dtype=torch.long),
                "bert_attention_mask": torch.zeros(self.max_len, dtype=torch.long),
                "cnn_input_ids": torch.zeros(self.max_len, dtype=torch.long),
                "label": torch.tensor(0, dtype=torch.long)
            }


def load_data(train_path, dev_path=None, test_path=None, min_freq=1, test_size=0.2):
    print(f"正在加载训练集：{train_path}")
    train_df = pd.read_csv(train_path, sep="\t", header=0, names=["sentence", "label"], on_bad_lines="skip")
    train_df = train_df.dropna(subset=["sentence", "label"])
    train_df["label"] = pd.to_numeric(train_df["label"], errors="coerce").fillna(0).astype(int)
    train_df = train_df[train_df["sentence"].str.strip() != ""]
    print(f"训练集有效行数：{len(train_df)}")

    train_raw_labels = train_df["label"].tolist()
    unique_labels = sorted(list(set(train_raw_labels)))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    print(f"\n检测到类别数：{num_classes}")
    print(f"标签映射：{label2id}")
    print(f"标签分布：{Counter(train_raw_labels)}")

    train_labels = [label2id[label] for label in train_raw_labels]
    train_texts = train_df["sentence"].tolist()

    # 处理验证集
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

    # 处理测试集
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

    # 构建TextCNN词汇表
    print("\n构建TextCNN词汇表...")
    all_tokens = []
    for text in train_texts:
        all_tokens.extend(jieba.lcut(str(text).strip()))
    counter = Counter(all_tokens)
    cnn_vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, freq in counter.items():
        if freq >= min_freq:
            cnn_vocab[token] = len(cnn_vocab)
    print(f"TextCNN词汇表大小：{len(cnn_vocab)}")

    # 加载BERT分词器
    print("\n加载BERT分词器...")
    from transformers import AutoTokenizer
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    print(f"✅ BERT分词器加载完成")

    return (train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels,
            bert_tokenizer, cnn_vocab, label2id, id2label, num_classes)


# ====================== 2. 教师模型（BERT）======================
class BERTTextCls(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super().__init__()
        from transformers import AutoModelForSequenceClassification
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-chinese",
            num_labels=num_classes
        )

        # 冻结前10层，只训练最后2层
        for param in list(self.bert.parameters())[:-4]:
            param.requires_grad = False

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.dropout(outputs.logits)


# ====================== 3. 学生模型（TextCNN）======================
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_filters=128, filter_sizes=[2, 3, 4], dropout=0.1, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=num_filters,
                kernel_size=fs
            ) for fs in filter_sizes
        ])
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embed = self.embedding(x)
        embed = embed.permute(0, 2, 1)

        conv_outs = []
        for conv in self.convs:
            out = F.relu(conv(embed))
            out = F.max_pool1d(out, out.size(2))
            conv_outs.append(out.squeeze(2))

        out = torch.cat(conv_outs, dim=1)
        out = self.dropout(out)
        return self.fc(out)


# ====================== 4. 🔥 核心：蒸馏损失函数 ======================
class DistillationLoss(nn.Module):
    """
    知识蒸馏损失函数：硬标签损失 + 软标签损失
    temperature: 温度系数，越高软标签越平滑
    alpha: 硬标签损失的权重，(1-alpha)是软标签损失的权重
    """

    def __init__(self, temperature=5.0, alpha=0.3):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        # 硬标签损失（和真实标签的交叉熵）
        hard_loss = self.ce_loss(student_logits, labels)

        # 软标签损失（和教师模型输出的KL散度）
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.temperature ** 2)

        # 总损失：加权求和
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        return total_loss


# ====================== 5. 蒸馏训练函数 ======================
def distill_train(teacher_model, student_model, dataloader, optimizer, criterion, device):
    teacher_model.eval()  # 教师模型设置为评估模式
    student_model.train()  # 学生模型设置为训练模式

    total_loss = 0
    total_correct = 0
    total_samples = 0

    print("\n开始蒸馏训练...")
    for batch_idx, batch in enumerate(dataloader):
        bert_input_ids = batch["bert_input_ids"].to(device)
        bert_attention_mask = batch["bert_attention_mask"].to(device)
        cnn_input_ids = batch["cnn_input_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        # 教师模型前向传播（不计算梯度）
        with torch.no_grad():
            teacher_logits = teacher_model(bert_input_ids, bert_attention_mask)

        # 学生模型前向传播
        student_logits = student_model(cnn_input_ids)

        # 计算蒸馏损失
        loss = criterion(student_logits, teacher_logits, labels)
        loss.backward()
        optimizer.step()

        # 计算准确率
        pred = torch.argmax(student_logits, dim=1)
        batch_correct = (pred == labels).sum().item()
        batch_acc = batch_correct / len(labels)

        total_loss += loss.item()
        total_correct += batch_correct
        total_samples += len(labels)

        if batch_idx % 10 == 0:
            print(f"  批次 {batch_idx + 1}/{len(dataloader)} | 损失：{loss.item():.4f} | 准确率：{batch_acc:.4f}")

    epoch_loss = total_loss / len(dataloader)
    epoch_acc = total_correct / total_samples if total_samples > 0 else 0
    print(f"\n蒸馏训练汇总 | 平均损失：{epoch_loss:.4f} | 平均准确率：{epoch_acc:.4f}")
    return epoch_loss, epoch_acc


# ====================== 6. 评估函数 ======================
def evaluate(model, dataloader, criterion, device, model_type="student", dataset_name="验证集"):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    print(f"\n开始{dataset_name}评估...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if model_type == "teacher":
                input_ids = batch["bert_input_ids"].to(device)
                attention_mask = batch["bert_attention_mask"].to(device)
                outputs = model(input_ids, attention_mask)
            else:  # student
                input_ids = batch["cnn_input_ids"].to(device)
                outputs = model(input_ids)

            labels = batch["label"].to(device)
            loss = criterion(outputs, labels)

            pred = torch.argmax(outputs, dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"  处理批次 {batch_idx + 1}/{len(dataloader)} | 损失：{loss.item():.4f}")

    epoch_loss = total_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    print(f"\n{dataset_name}汇总 | 平均损失：{epoch_loss:.4f} | 准确率：{epoch_acc:.4f}")
    return epoch_loss, epoch_acc, all_preds, all_labels


# ====================== 7. 速度对比测试 ======================
def benchmark_inference(model, dataloader, device, model_type="student", num_runs=100):
    print(f"\n开始{model_type}模型推理速度测试...")
    model.eval()

    # 取第一个批次作为测试
    batch = next(iter(dataloader))
    if model_type == "teacher":
        input_ids = batch["bert_input_ids"].to(device)
        attention_mask = batch["bert_attention_mask"].to(device)
    else:
        input_ids = batch["cnn_input_ids"].to(device)

    # 预热
    with torch.no_grad():
        for _ in range(10):
            if model_type == "teacher":
                _ = model(input_ids, attention_mask)
            else:
                _ = model(input_ids)

    # 正式测试
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            if model_type == "teacher":
                _ = model(input_ids, attention_mask)
            else:
                _ = model(input_ids)
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs * 1000  # 转换为毫秒
    print(f"✅ 平均推理时间：{avg_time:.2f} ms/批次")
    print(f"✅ 每秒可处理：{1000 / avg_time * len(input_ids):.2f} 条")
    return avg_time


# ====================== 8. 主函数（完整蒸馏流水线）======================
if __name__ == "__main__":
    # 可调整参数
    TRAIN_PATH = os.path.join(DATA_ROOT, "train.txt")
    DEV_PATH = os.path.join(DATA_ROOT, "dev.txt")
    TEST_PATH = os.path.join(DATA_ROOT, "test.txt")

    BATCH_SIZE = 64  # TextCNN可以用大batch_size
    TEACHER_EPOCHS = 2  # 教师模型训练轮数
    DISTILL_EPOCHS = 5  # 蒸馏训练轮数
    TEMPERATURE = 5.0  # 温度系数，4-6效果最好
    ALPHA = 0.3  # 硬标签权重，0.2-0.4效果最好
    LR = 1e-4  # 蒸馏学习率，比从头训练低1个量级
    MAX_LEN = 64
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

    # 加载数据
    try:
        (train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels,
         bert_tokenizer, cnn_vocab, label2id, id2label, num_classes) = load_data(TRAIN_PATH, DEV_PATH, TEST_PATH)
    except Exception as e:
        print(f"❌ 数据加载失败：{e}")
        exit()

    # 构建数据集和加载器
    print("\n构建数据集...")
    train_dataset = DistillationDataset(train_texts, train_labels, bert_tokenizer, cnn_vocab, MAX_LEN)
    dev_dataset = DistillationDataset(dev_texts, dev_labels, bert_tokenizer, cnn_vocab, MAX_LEN)
    test_dataset = DistillationDataset(test_texts, test_labels, bert_tokenizer, cnn_vocab,
                                       MAX_LEN) if test_texts else None

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

    # ====================== 第一步：训练教师模型（BERT）======================
    print("\n" + "=" * 60)
    print("第一步：训练教师模型（BERT）")
    print("=" * 60)

    teacher_model = BERTTextCls(num_classes, DROPOUT).to(device)
    print(f"教师模型参数量：{sum(p.numel() for p in teacher_model.parameters()):,}")

    teacher_optimizer = torch.optim.AdamW(teacher_model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    best_teacher_dev_acc = 0
    for epoch in range(TEACHER_EPOCHS):
        print(f"\n{'=' * 25} Epoch {epoch + 1}/{TEACHER_EPOCHS} {'=' * 25}")
        # 训练教师模型（复用之前的train函数，需要稍微修改输入）
        teacher_model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["bert_input_ids"].to(device)
            attention_mask = batch["bert_attention_mask"].to(device)
            y = batch["label"].to(device)

            teacher_optimizer.zero_grad()
            outputs = teacher_model(input_ids, attention_mask)
            loss = criterion(outputs, y)
            loss.backward()
            teacher_optimizer.step()

            pred = torch.argmax(outputs, dim=1)
            batch_correct = (pred == y).sum().item()
            total_loss += loss.item()
            total_correct += batch_correct
            total_samples += len(y)

            if batch_idx % 10 == 0:
                print(f"  批次 {batch_idx + 1}/{len(train_loader)} | 损失：{loss.item():.4f}")

        train_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_samples
        print(f"\n训练汇总 | 平均损失：{train_loss:.4f} | 平均准确率：{train_acc:.4f}")

        dev_loss, dev_acc, _, _ = evaluate(teacher_model, dev_loader, criterion, device, model_type="teacher")

        if dev_acc > best_teacher_dev_acc:
            best_teacher_dev_acc = dev_acc
            torch.save({
                "model_state_dict": teacher_model.state_dict(),
                "tokenizer_name": "bert-base-chinese",
                "label2id": label2id,
                "id2label": id2label,
                "config": {
                    "DROPOUT": DROPOUT,
                    "MAX_LEN": MAX_LEN
                }
            }, os.path.join(CODE_ROOT, "teacher_bert_model.pth"))
            print("\n✅ 保存最佳教师模型")

    print(f"\n教师模型最佳验证准确率：{best_teacher_dev_acc:.4f}")

    # ====================== 第二步：蒸馏训练学生模型（TextCNN）======================
    print("\n" + "=" * 60)
    print("第二步：蒸馏训练学生模型（TextCNN）")
    print("=" * 60)

    # 加载最佳教师模型
    checkpoint = torch.load(os.path.join(CODE_ROOT, "teacher_bert_model.pth"), weights_only=False)
    teacher_model.load_state_dict(checkpoint["model_state_dict"])

    # 初始化学生模型
    student_model = TextCNN(
        len(cnn_vocab),
        embed_dim=128,
        num_filters=128,
        filter_sizes=[2, 3, 4],
        dropout=DROPOUT,
        num_classes=num_classes
    ).to(device)
    print(f"学生模型参数量：{sum(p.numel() for p in student_model.parameters()):,}")
    print(
        f"模型压缩比：{sum(p.numel() for p in teacher_model.parameters()) / sum(p.numel() for p in student_model.parameters()):.1f}x")

    # 蒸馏损失函数
    distill_criterion = DistillationLoss(temperature=TEMPERATURE, alpha=ALPHA)
    student_optimizer = torch.optim.Adam(student_model.parameters(), lr=LR)

    best_student_dev_acc = 0
    for epoch in range(DISTILL_EPOCHS):
        print(f"\n{'=' * 25} Epoch {epoch + 1}/{DISTILL_EPOCHS} {'=' * 25}")
        train_loss, train_acc = distill_train(teacher_model, student_model, train_loader, student_optimizer,
                                              distill_criterion, device)
        dev_loss, dev_acc, _, _ = evaluate(student_model, dev_loader, criterion, device, model_type="student")

        if dev_acc > best_student_dev_acc:
            best_student_dev_acc = dev_acc
            torch.save({
                "model_state_dict": student_model.state_dict(),
                "vocab": cnn_vocab,
                "label2id": label2id,
                "id2label": id2label,
                "config": {
                    "EMBED_DIM": 128,
                    "NUM_FILTERS": 128,
                    "FILTER_SIZES": [2, 3, 4],
                    "DROPOUT": DROPOUT,
                    "MAX_LEN": MAX_LEN
                }
            }, os.path.join(CODE_ROOT, "distilled_textcnn_model.pth"))
            print("\n✅ 保存最佳蒸馏学生模型")

    print(f"\n蒸馏后学生模型最佳验证准确率：{best_student_dev_acc:.4f}")
    print(f"精度损失：{(best_teacher_dev_acc - best_student_dev_acc) * 100:.2f}%")

    # ====================== 第三步：测试所有模型并对比 ======================
    if test_loader:
        print("\n" + "=" * 60)
        print("测试教师模型（BERT）")
        print("=" * 60)
        teacher_test_loss, teacher_test_acc, _, _ = evaluate(
            teacher_model, test_loader, criterion, device, model_type="teacher", dataset_name="测试集"
        )

        print("\n" + "=" * 60)
        print("测试蒸馏后学生模型（TextCNN）")
        print("=" * 60)
        checkpoint = torch.load(os.path.join(CODE_ROOT, "distilled_textcnn_model.pth"), weights_only=False)
        student_model.load_state_dict(checkpoint["model_state_dict"])
        student_test_loss, student_test_acc, _, _ = evaluate(
            student_model, test_loader, criterion, device, model_type="student", dataset_name="测试集"
        )

        # ====================== 第四步：速度对比测试 ======================
        print("\n" + "=" * 60)
        print("推理速度对比测试")
        print("=" * 60)

        print("\n教师模型（BERT）：")
        teacher_speed = benchmark_inference(teacher_model, test_loader, device, model_type="teacher")

        print("\n学生模型（TextCNN）：")
        student_speed = benchmark_inference(student_model, test_loader, device, model_type="student")

        # 输出最终对比结果
        print("\n" + "=" * 60)
        print("最终效果对比")
        print("=" * 60)
        print(f"{'模型':<20} {'准确率':<10} {'参数量':<12} {'推理速度':<15} {'加速比':<10}")
        print("-" * 60)
        print(f"{'BERT（教师）':<20} {teacher_test_acc:.4f}{'':<6} {'110M':<12} {teacher_speed:.2f}ms{'':<7} {'1x':<10}")
        print(
            f"{'蒸馏TextCNN（学生）':<20} {student_test_acc:.4f}{'':<6} {'10M':<12} {student_speed:.2f}ms{'':<7} {teacher_speed / student_speed:.1f}x")
        print("=" * 60)


    # ====================== 单句预测（蒸馏TextCNN）======================
    def predict_distilled(text):
        checkpoint = torch.load(os.path.join(CODE_ROOT, "distilled_textcnn_model.pth"), weights_only=False)
        vocab = checkpoint["vocab"]
        id2label = checkpoint["id2label"]
        config = checkpoint["config"]
        num_classes = len(id2label)

        model = TextCNN(
            len(vocab),
            config["EMBED_DIM"],
            config["NUM_FILTERS"],
            config["FILTER_SIZES"],
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
        "东5环海棠公社230-290平准现房98折优惠"
    ]

    print("\n蒸馏模型预测测试：")
    for text in test_texts:
        pred, prob = predict_distilled(text)
        print(f"文本：{text}")
        print(f"预测类别：{pred}，置信度：{prob:.4f}\n")
