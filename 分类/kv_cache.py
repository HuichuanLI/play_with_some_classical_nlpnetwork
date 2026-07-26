import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import Counter
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

# ====================== 路径配置（与之前完全一致）======================
DATA_ROOT = "/Users/lhc456/Desktop/nlp课程/play_with_some_classical_nlpnetwork/data"
CODE_ROOT = "/Users/lhc456/Desktop/nlp课程/play_with_some_classical_nlpnetwork"

os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(CODE_ROOT, exist_ok=True)


# ====================== 1. 数据处理（与之前完全一致）======================
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


def load_data(train_path, dev_path=None, test_path=None, test_size=0.2):
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
    print(f"标签分布：{Counter(train_raw_labels)}")

    train_labels = [label2id[label] for label in train_raw_labels]
    train_texts = train_df["sentence"].tolist()

    train_texts, dev_texts, train_labels, dev_labels = train_test_split(
        train_texts, train_labels, test_size=test_size, random_state=42
    )
    print(f"划分后训练集大小：{len(train_texts)}，验证集大小：{len(dev_texts)}")

    print("\n加载BERT中文分词器...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    print(f"✅ BERT分词器加载完成，词汇表大小：{len(tokenizer)}")

    return (train_texts, train_labels, dev_texts, dev_labels, [], [],
            tokenizer, label2id, id2label, num_classes)


# ====================== 2. 带KV Cache的自注意力核心模块 ======================
class CachedSelfAttention(nn.Module):
    """
    带KV缓存的多头自注意力，复现文档核心实现
    - 分类训练/推理：use_cache=False，全序列计算
    - 生成式场景：use_cache=True，逐token复用历史KV
    """

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # KV 缓存
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, use_cache=False):
        batch_size, seq_len, embed_dim = x.shape

        # 计算当前步 Q/K/V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 拼接历史缓存
        if use_cache and self.cache_k is not None:
            k = torch.cat([self.cache_k, k], dim=-2)
            v = torch.cat([self.cache_v, v], dim=-2)

        # 更新缓存
        if use_cache:
            self.cache_k = k
            self.cache_v = v

        # 缩放点积注意力
        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)

        output = attn_probs @ v
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out_proj(output)

    def reset_cache(self):
        """重置KV缓存，新序列前调用"""
        self.cache_k = None
        self.cache_v = None


# ====================== 3. 自注意力文本分类模型 ======================
class SelfAttentionClassifier(nn.Module):
    """
    结构：词嵌入 + 可学习位置编码 → 多层自注意力 → 均值池化 → 分类头
    完整保留 KV Cache 能力，兼容分类与生成两种场景
    """

    def __init__(self, vocab_size, num_classes, embed_dim=128, num_heads=4, num_layers=2, max_len=128):
        super().__init__()
        self.embed_dim = embed_dim

        # 词嵌入 + 位置编码
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)

        # 多层自注意力
        self.attn_layers = nn.ModuleList([
            CachedSelfAttention(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])

        # 分类头
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        batch_size, seq_len = input_ids.shape

        # 嵌入层
        x = self.token_emb(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = x + self.pos_emb(positions)

        # 多层自注意力 + 残差归一化
        for attn, norm in zip(self.attn_layers, self.layer_norms):
            attn_out = attn(x, use_cache=use_cache)
            x = norm(x + attn_out)

        # 均值池化得到句子向量（可替换为取第一个token）
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        x = self.dropout(x)
        logits = self.classifier(x)
        return logits

    def reset_all_cache(self):
        """重置所有层的KV缓存"""
        for layer in self.attn_layers:
            layer.reset_cache()


# ====================== 4. 训练 & 评估（与之前结构完全一致）======================
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
        # 分类任务关闭KV Cache
        logits = model(input_ids, attention_mask, use_cache=False)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        pred = torch.argmax(logits, dim=1)
        batch_correct = (pred == y).sum().item()

        total_loss += loss.item()
        total_correct += batch_correct
        total_samples += len(y)

        print(
            f"  批次 {batch_idx + 1}/{len(dataloader)} | 损失：{loss.item():.4f} | 准确率：{batch_correct / len(y):.4f}")

    epoch_loss = total_loss / len(dataloader)
    epoch_acc = total_correct / total_samples if total_samples > 0 else 0
    print(f"\n训练汇总 | 平均损失：{epoch_loss:.4f} | 平均准确率：{epoch_acc:.4f}")
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    print("\n开始验证...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["label"].to(device)

            logits = model(input_ids, attention_mask, use_cache=False)
            loss = criterion(logits, y)

            pred = torch.argmax(logits, dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            total_loss += loss.item()

            print(f"  处理批次 {batch_idx + 1}/{len(dataloader)}")

    epoch_loss = total_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    print(f"\n验证汇总 | 平均损失：{epoch_loss:.4f} | 准确率：{epoch_acc:.4f}")
    return epoch_loss, epoch_acc, all_preds, all_labels


# ====================== 5. 主函数 ======================
if __name__ == "__main__":
    TRAIN_PATH = os.path.join(DATA_ROOT, "train.txt")

    # 超参数
    BATCH_SIZE = 1024
    EPOCHS = 5
    LR = 1e-3
    MAX_LEN = 64
    EMBED_DIM = 128
    NUM_HEADS = 4
    NUM_LAYERS = 2

    # 设备选择
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用 NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("使用 Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("使用 CPU")

    print("\n检查数据文件...")
    if not os.path.exists(TRAIN_PATH):
        print(f"❌ 训练集文件不存在：{TRAIN_PATH}")
        exit()
    print(f"✅ 找到训练集：{TRAIN_PATH}")

    try:
        (train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels,
         tokenizer, label2id, id2label, num_classes) = load_data(TRAIN_PATH)
    except Exception as e:
        print(f"❌ 数据加载失败：{e}")
        exit()

    print("\n构建数据集...")
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    dev_dataset = TextDataset(dev_texts, dev_labels, tokenizer, MAX_LEN)

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

    print(f"训练集批次数量：{len(train_loader)}")
    print(f"验证集批次数量：{len(dev_loader)}")

    print("\n初始化自注意力分类模型...")
    model = SelfAttentionClassifier(
        vocab_size=tokenizer.vocab_size,
        num_classes=num_classes,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_len=MAX_LEN
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数量：{trainable_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print("\n" + "=" * 60)
    print("开始训练循环")
    print("=" * 60)

    for epoch in range(EPOCHS):
        print(f"\n{'=' * 25} Epoch {epoch + 1}/{EPOCHS} {'=' * 25}")
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        dev_loss, dev_acc, _, _ = evaluate(model, dev_loader, criterion, device)

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)


    # 单句预测
    def predict(text):
        model.eval()
        encoding = tokenizer(
            text.strip(),
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask, use_cache=False)
            pred_id = torch.argmax(logits, dim=1).item()
            prob = F.softmax(logits, dim=1)[0][pred_id].item()
        return id2label[pred_id], prob


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
