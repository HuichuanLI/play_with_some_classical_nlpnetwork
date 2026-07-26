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

# ====================== 路径配置（完全不变）======================
DATA_ROOT = "/Users/lhc456/Desktop/nlp课程/play_with_some_classical_nlpnetwork/data"
CODE_ROOT = "/Users/lhc456/Desktop/nlp课程/play_with_some_classical_nlpnetwork"

os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(CODE_ROOT, exist_ok=True)


# ====================== 1. 数据处理（完全不变）======================
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


# ====================== 2. MoE 核心模块 ======================
class Expert(nn.Module):
    """单个专家网络：两层全连接 + 激活 + Dropout"""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class TopKRouter(nn.Module):
    """Top-K 门控路由：输出选中专家权重、索引，以及全量概率（用于负载均衡）"""

    def __init__(self, input_dim, num_experts, top_k=2):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.top_k = top_k

    def forward(self, x):
        logits = self.gate(x)
        gate_probs = F.softmax(logits, dim=1)  # 全量分布，计算辅助损失用
        top_k_logits, indices = logits.topk(self.top_k, dim=1)
        routing_weights = F.softmax(top_k_logits, dim=1)  # top-k 归一化权重
        return routing_weights, indices, gate_probs


class SparseMoE(nn.Module):
    """稀疏混合专家模块：门控选专家 + 批量稀疏计算"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = TopKRouter(input_dim, num_experts, top_k)
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)
        ])

    def forward(self, x):
        batch_size = x.shape[0]
        routing_weights, selected_experts_indices, gate_probs = self.router(x)

        expert_output_dim = self.experts[0].net[-1].out_features
        final_output = torch.zeros(batch_size, expert_output_dim, device=x.device)

        # 稀疏计算：遍历专家，批量处理分配给该专家的样本
        for i in range(self.num_experts):
            batch_idx, index_expert = torch.where(selected_experts_indices == i)
            if batch_idx.numel() == 0:
                continue  # 无样本选中则跳过，实现稀疏加速

            expert_inputs = x[batch_idx]
            expert_out = self.experts[i](expert_inputs)
            weights = routing_weights[batch_idx, index_expert].unsqueeze(1)
            final_output.index_add_(0, batch_idx, expert_out * weights)

        return final_output, gate_probs


def compute_load_balance_loss(gate_probs, aux_weight=0.01):
    """
    负载均衡辅助损失：MoE 训练必备
    防止所有样本路由到少数专家，避免专家塌缩
    """
    expert_avg_probs = gate_probs.mean(dim=0)
    balance_loss = torch.sum(expert_avg_probs ** 2) * gate_probs.shape[1]
    return aux_weight * balance_loss


# ====================== 3. BERT + MoE 文本分类模型 ======================
class BERTMoECls(nn.Module):
    """结构：BERT 特征提取 → MoE 稀疏变换 → 分类头"""

    def __init__(self, num_classes, num_experts=8, top_k=2, moe_hidden_dim=1024):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained("bert-base-chinese")
        self.bert_hidden = 768  # bert-base 固定隐藏层维度

        # MoE 特征变换层
        self.moe = SparseMoE(
            input_dim=self.bert_hidden,
            hidden_dim=moe_hidden_dim,
            output_dim=self.bert_hidden,
            num_experts=num_experts,
            top_k=top_k
        )

        # 分类头
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert_hidden, num_classes)

        # 冻结 BERT 大部分参数，仅训练最后几层 + MoE + 分类头
        all_params = list(self.bert.parameters())
        freeze_num = int(len(all_params) * 0.85)
        for param in all_params[:freeze_num]:
            param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        # 取   token 作为句子级特征
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_feat = bert_out.last_hidden_state[:, 0, :]
        cls_feat = self.dropout(cls_feat)

        # MoE 稀疏变换
        moe_out, gate_probs = self.moe(cls_feat)

        # 分类输出 + 路由概率（用于辅助损失）
        logits = self.classifier(moe_out)
        return logits, gate_probs


# ====================== 4. 训练 & 评估（适配 MoE 双输出）======================
def train(model, dataloader, optimizer, criterion, device, aux_weight=0.01):
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
        logits, gate_probs = model(input_ids, attention_mask)

        # 主损失 + 负载均衡辅助损失
        cls_loss = criterion(logits, y)
        balance_loss = compute_load_balance_loss(gate_probs, aux_weight)
        loss = cls_loss + balance_loss

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


def evaluate(model, dataloader, criterion, device, aux_weight=0.01):
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

            logits, gate_probs = model(input_ids, attention_mask)
            cls_loss = criterion(logits, y)
            balance_loss = compute_load_balance_loss(gate_probs, aux_weight)
            loss = cls_loss + balance_loss

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
    BATCH_SIZE = 64
    EPOCHS = 5
    LR = 2e-5
    MAX_LEN = 64
    NUM_EXPERTS = 8
    TOP_K = 2
    AUX_WEIGHT = 0.01  # 负载均衡损失权重

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

    print("\n初始化 BERT+MoE 文本分类模型...")
    model = BERTMoECls(
        num_classes=num_classes,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        moe_hidden_dim=1024
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量：{total_params:,}")
    print(f"可训练参数量：{trainable_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print("\n" + "=" * 60)
    print("开始训练循环")
    print("=" * 60)

    for epoch in range(EPOCHS):
        print(f"\n{'=' * 25} Epoch {epoch + 1}/{EPOCHS} {'=' * 25}")
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, AUX_WEIGHT)
        dev_loss, dev_acc, _, _ = evaluate(model, dev_loader, criterion, device, AUX_WEIGHT)

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
            logits, _ = model(input_ids, attention_mask)
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
