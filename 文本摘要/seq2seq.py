import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import jieba
import re
import os
import time
from multiprocessing import cpu_count, Pool
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ==================================================
# 1. 全局配置
# ==================================================
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_PATH, "data")
MODEL_DIR = os.path.join(ROOT_PATH, "models")
RESULT_DIR = os.path.join(ROOT_PATH, "results")

for d in [DATA_DIR, MODEL_DIR, RESULT_DIR]:
    os.makedirs(d, exist_ok=True)

# 数据文件路径
TRAIN_RAW = os.path.join(DATA_DIR, "train.csv")
TEST_RAW = os.path.join(DATA_DIR, "test.csv")
STOP_WORDS_FILE = os.path.join(DATA_DIR, "stopwords.txt")

# 预处理产物路径
TRAIN_X_NPY = os.path.join(DATA_DIR, "train_X.npy")
TRAIN_Y_NPY = os.path.join(DATA_DIR, "train_Y.npy")
TEST_X_NPY = os.path.join(DATA_DIR, "test_X.npy")
VOCAB_FILE = os.path.join(DATA_DIR, "vocab.txt")
REVERSE_VOCAB_FILE = os.path.join(DATA_DIR, "reverse_vocab.txt")

# 模型超参数（平衡速度与效果）
CONFIG = {
    "max_enc_len": 200,
    "max_dec_len": 40,
    "batch_size": 32,
    "epochs": 10,
    "embed_size": 256,
    "enc_units": 256,
    "dec_units": 256,
    "attn_units": 16,
    "learning_rate": 0.001
}


# ==================================================
# 2. 工具函数
# ==================================================
def parallelize(df, func):
    """多进程并行处理DataFrame"""
    cores = cpu_count()
    data_split = np.array_split(df, cores)
    pool = Pool(cores)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data


def save_vocab(path, word2id):
    """保存词汇表"""
    with open(path, "w", encoding="utf-8") as f:
        for w, idx in word2id.items():
            f.write(f"{w}\t{idx}\n")


def load_vocab(vocab_path, reverse_path):
    """加载词汇表（鲁棒版，自动跳过脏行）"""
    word2id, id2word = {}, {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            w, idx = parts
            try:
                word2id[w] = int(idx)
            except ValueError:
                continue

    with open(reverse_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            idx, w = parts
            try:
                id2word[int(idx)] = w
            except ValueError:
                continue

    return word2id, id2word


# ==================================================
# 3. 数据预处理
# ==================================================
def load_stop_words(path):
    """加载停用词"""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


STOP_WORDS = load_stop_words(STOP_WORDS_FILE)


def clean_text(text):
    """文本清洗"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\D(\d\.)\D", "", text)
    text = re.sub(r"[(（]进口[)）]|\(海外\)", "", text)
    text = re.sub(r"车主说|技师说|语音|图片|你好|您好", "", text)
    text = re.sub(r"[^,!?。.\-\u4e00-\u9fa5_a-zA-Z0-9]", "", text)
    return text


def sentence_process(text):
    """单句处理：清洗→分词→去停用词"""
    text = clean_text(text)
    words = jieba.cut(text)
    words = [w for w in words if w and w not in STOP_WORDS]
    return " ".join(words)


def batch_process(df):
    """批量处理DataFrame列"""
    for col in ["Brand", "Model", "Question", "Dialogue"]:
        if col in df.columns:
            df[col] = df[col].apply(sentence_process)
    if "Report" in df.columns:
        df["Report"] = df["Report"].apply(sentence_process)
    return df


def get_max_len(series):
    """计算合适的最大长度（均值+2倍标准差）"""
    lens = series.apply(lambda x: x.count(" ") + 1)
    return int(np.mean(lens) + 2 * np.std(lens))


def pad_sentence(sentence, max_len, word2id):
    """填充特殊标记并截断"""
    words = sentence.strip().split(" ")[:max_len]
    words = [w if w in word2id else "<UNK>" for w in words]
    words = ["<START>"] + words + ["<STOP>"]
    words += ["<PAD>"] * (max_len + 2 - len(words))
    return " ".join(words)


def sent2ids(sentence, word2id):
    """句子转id序列"""
    return [word2id.get(w, word2id["<UNK>"]) for w in sentence.split(" ")]


def build_dataset():
    """完整数据预处理流程（仅首次运行）"""
    print("=" * 60)
    print("开始构建数据集（首次运行）")
    print("=" * 60)

    # 1. 加载原始数据
    print("\n1. 加载原始数据")
    train_df = pd.read_csv(TRAIN_RAW, engine="python", encoding="utf-8")
    test_df = pd.read_csv(TEST_RAW, engine="python", encoding="utf-8")
    print(f"训练集: {len(train_df)} 条 | 测试集: {len(test_df)} 条")

    # 2. 去空值
    print("\n2. 去除空值")
    train_df.dropna(subset=["Question", "Dialogue", "Report"], how="any", inplace=True)
    test_df.dropna(subset=["Question", "Dialogue"], how="any", inplace=True)
    print(f"去空后训练集: {len(train_df)} 条")

    # 3. 多进程预处理
    print("\n3. 多进程文本预处理")
    train_df = parallelize(train_df, batch_process)
    test_df = parallelize(test_df, batch_process)

    # 4. 构建词汇表
    print("\n4. 构建词汇表")
    all_text = pd.concat([
        train_df["Question"] + " " + train_df["Dialogue"] + " " + train_df["Report"],
        test_df["Question"] + " " + test_df["Dialogue"]
    ])

    word_count = {}
    for text in all_text:
        for w in text.strip().split(" "):
            word_count[w] = word_count.get(w, 0) + 1

    # 过滤低频词（出现≥5次）
    word2id = {}
    for w, freq in word_count.items():
        if freq >= 5:
            word2id[w] = len(word2id)

    # 添加特殊标记
    for token in ["<PAD>", "<UNK>", "<START>", "<STOP>"]:
        if token not in word2id:
            word2id[token] = len(word2id)

    id2word = {v: k for k, v in word2id.items()}
    print(f"词汇表大小: {len(word2id)}")

    # 5. 构造输入输出
    print("\n5. 构造输入输出并填充")
    train_df["X"] = train_df["Question"] + " " + train_df["Dialogue"]
    test_df["X"] = test_df["Question"] + " " + test_df["Dialogue"]

    x_max_len = min(get_max_len(pd.concat([train_df["X"], test_df["X"]])), CONFIG["max_enc_len"])
    y_max_len = min(get_max_len(train_df["Report"]), CONFIG["max_dec_len"])
    print(f"输入最大长度: {x_max_len} | 输出最大长度: {y_max_len}")

    train_df["X"] = train_df["X"].apply(lambda x: pad_sentence(x, x_max_len, word2id))
    test_df["X"] = test_df["X"].apply(lambda x: pad_sentence(x, x_max_len, word2id))
    train_df["Y"] = train_df["Report"].apply(lambda x: pad_sentence(x, y_max_len, word2id))

    # 6. 转id并保存
    print("\n6. 转换为numpy数组并保存")
    train_X = np.array(train_df["X"].apply(lambda x: sent2ids(x, word2id)).tolist())
    train_Y = np.array(train_df["Y"].apply(lambda x: sent2ids(x, word2id)).tolist())
    test_X = np.array(test_df["X"].apply(lambda x: sent2ids(x, word2id)).tolist())

    np.save(TRAIN_X_NPY, train_X)
    np.save(TRAIN_Y_NPY, train_Y)
    np.save(TEST_X_NPY, test_X)
    save_vocab(VOCAB_FILE, word2id)
    save_vocab(REVERSE_VOCAB_FILE, id2word)

    print(f"\ntrain_X shape: {train_X.shape}")
    print(f"train_Y shape: {train_Y.shape}")
    print(f"test_X shape: {test_X.shape}")
    print("\n✅ 数据集构建完成")
    return word2id, id2word


def get_dataloader(mode="train", batch_size=32):
    """获取数据加载器"""
    if mode == "train":
        X = np.load(TRAIN_X_NPY)[:, :CONFIG["max_enc_len"]]
        Y = np.load(TRAIN_Y_NPY)[:, :CONFIG["max_dec_len"]]
        dataset = TensorDataset(torch.LongTensor(X), torch.LongTensor(Y))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    else:
        X = np.load(TEST_X_NPY)[:, :CONFIG["max_enc_len"]]
        dataset = TensorDataset(torch.LongTensor(X))
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)


# ==================================================
# 4. 模型层（无维度bug，完全对齐）
# ==================================================
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=1, batch_first=True)

    def forward(self, x, h0):
        # x: (batch, seq_len) | h0: (1, batch, hidden_dim)
        x = self.embedding(x)
        output, hn = self.gru(x, h0)
        return output, hn.squeeze(0)  # hn: (batch, hidden_dim)

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.gru.hidden_size)


class Attention(nn.Module):
    def __init__(self, enc_dim, dec_dim, attn_dim):
        super().__init__()
        self.w_enc = nn.Linear(enc_dim, attn_dim)
        self.w_dec = nn.Linear(dec_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1)

    def forward(self, dec_hidden, enc_output):
        """
        兼容单步推理和全序列训练
        dec_hidden: (batch, dec_dim) 推理模式 或 (batch, dec_len, dec_dim) 训练模式
        enc_output: (batch, enc_len, enc_dim)
        """
        if dec_hidden.dim() == 2:
            # 推理模式：单步
            dec_proj = self.w_dec(dec_hidden).unsqueeze(1)  # (batch, 1, attn_dim)
        else:
            # 训练模式：全序列并行
            dec_proj = self.w_dec(dec_hidden)  # (batch, dec_len, attn_dim)

        enc_proj = self.w_enc(enc_output)  # (batch, enc_len, attn_dim)
        # 广播计算得分
        score = self.v(torch.tanh(enc_proj.unsqueeze(1) + dec_proj.unsqueeze(2)))
        attn_weights = F.softmax(score, dim=2)
        # 加权求和
        context = torch.sum(attn_weights * enc_output.unsqueeze(1), dim=2)

        if dec_hidden.dim() == 2:
            context = context.squeeze(1)

        return context, attn_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim + hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward_train(self, x, context, h0):
        """训练模式：全序列并行输入"""
        x = self.embedding(x)
        x = torch.cat([x, context], dim=-1)
        output, hn = self.gru(x, h0)
        output = torch.cat([output, context], dim=-1)
        pred = self.fc(output)
        return pred, hn.squeeze(0)

    def forward_step(self, x, context, h0):
        """推理模式：单步输入"""
        x = self.embedding(x)
        x = torch.cat([x, context.unsqueeze(1)], dim=-1)
        output, hn = self.gru(x, h0.unsqueeze(0))
        output = output.squeeze(1)
        context = context.squeeze(1) if context.dim() == 3 else context
        output = torch.cat([output, context], dim=-1)
        pred = self.fc(output)
        return pred, hn.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(vocab_size, config["embed_size"], config["enc_units"])
        self.attention = Attention(config["enc_units"], config["dec_units"], config["attn_units"])
        self.decoder = Decoder(vocab_size, config["embed_size"], config["dec_units"])

    def forward(self, enc_input, dec_target, start_idx):
        """训练阶段：全并行计算，仅返回预测结果"""
        batch_size = enc_input.shape[0]
        device = enc_input.device

        # 编码器编码
        h0 = self.encoder.init_hidden(batch_size).to(device)
        enc_output, enc_hidden = self.encoder(enc_input, h0)

        # 构造解码器输入：<START> + 前n-1个真实标签
        start_tokens = torch.full((batch_size, 1), start_idx, dtype=torch.long, device=device)
        dec_input = torch.cat([start_tokens, dec_target[:, :-1]], dim=1)

        # 初始上下文
        init_context = enc_hidden.unsqueeze(1).repeat(1, dec_input.shape[1], 1)

        # 解码器全序列并行
        dec_hiddens, _ = self.decoder.gru(
            torch.cat([self.decoder.embedding(dec_input), init_context], dim=-1),
            enc_hidden.unsqueeze(0)
        )

        # 并行计算所有时间步注意力
        context_vector, _ = self.attention(dec_hiddens, enc_output)

        # 拼接并预测
        output = torch.cat([dec_hiddens, context_vector], dim=-1)
        predictions = self.decoder.fc(output)

        return predictions

    def generate(self, enc_input, start_idx, stop_idx, max_len, device):
        """推理阶段：贪心解码，自回归生成"""
        batch_size = enc_input.shape[0]

        # 编码器编码
        h0 = self.encoder.init_hidden(batch_size).to(device)
        enc_output, dec_hidden = self.encoder(enc_input, h0)

        # 初始输入
        dec_input = torch.full((batch_size, 1), start_idx, dtype=torch.long, device=device)
        results = [[] for _ in range(batch_size)]
        finished = [False] * batch_size

        for _ in range(max_len):
            context, _ = self.attention(dec_hidden, enc_output)
            pred, dec_hidden = self.decoder.forward_step(dec_input, context, dec_hidden)

            # 贪心取最大概率
            pred_ids = torch.argmax(pred, dim=1)

            # 记录结果
            for i in range(batch_size):
                if not finished[i]:
                    if pred_ids[i].item() == stop_idx:
                        finished[i] = True
                    else:
                        results[i].append(pred_ids[i].item())

            if all(finished):
                break

            # 下一步输入
            dec_input = pred_ids.unsqueeze(1)

        return results


# ==================================================
# 5. 训练函数
# ==================================================
def train_model(model, word2id, config, device):
    epochs = config["epochs"]
    pad_idx = word2id["<PAD>"]
    start_idx = word2id["<START>"]

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    train_loader = get_dataloader("train", config["batch_size"])
    steps_per_epoch = len(train_loader)

    print(f"\n开始训练 | 共{epochs}轮 | 每轮{steps_per_epoch}个批次 | 设备: {device}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
        for batch_idx, (enc_x, dec_y) in enumerate(pbar):
            enc_x = enc_x.to(device)
            dec_y = dec_y.to(device)

            optimizer.zero_grad()
            pred = model(enc_x, dec_y, start_idx)

            # 计算损失
            loss = criterion(pred.transpose(1, 2), dec_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / (batch_idx + 1):.4f}"
            })

        avg_loss = total_loss / steps_per_epoch
        epoch_time = time.time() - start_time

        # 每2轮保存一次
        if (epoch + 1) % 2 == 0:
            save_path = os.path.join(MODEL_DIR, f"seq2seq_epoch_{epoch + 1}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"✅ 模型已保存: {save_path}")

        print(f"Epoch {epoch + 1} 完成 | 平均损失: {avg_loss:.4f} | 耗时: {epoch_time:.1f}s")
        print("-" * 60)


# ==================================================
# 6. 测试生成函数
# ==================================================
def generate_summary(model, word2id, id2word, config, device):
    model.eval()
    test_loader = get_dataloader("test", config["batch_size"])
    start_idx = word2id["<START>"]
    stop_idx = word2id["<STOP>"]

    results = []
    print("\n开始生成测试集摘要...")

    with torch.no_grad():
        for (enc_x,) in tqdm(test_loader, desc="生成中", unit="batch"):
            enc_x = enc_x.to(device)
            batch_results = model.generate(enc_x, start_idx, stop_idx, config["max_dec_len"], device)

            # id转文字
            for ids in batch_results:
                words = [id2word.get(i, "<UNK>") for i in ids]
                summary = "".join(words)
                results.append(summary)

    # 保存结果
    test_df = pd.read_csv(TEST_RAW)
    test_df["Prediction"] = results[:len(test_df)]

    save_name = f"result_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    save_path = os.path.join(RESULT_DIR, save_name)
    test_df[["QID", "Prediction"]].to_csv(save_path, index=False)

    print(f"\n✅ 结果已保存: {save_path}")
    return results


# ==================================================
# 7. 主函数
# ==================================================
if __name__ == "__main__":
    # 设备选择
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用设备: NVIDIA GPU ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("使用设备: Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("使用设备: CPU")

    # 加载或构建数据集
    if os.path.exists(VOCAB_FILE) and os.path.exists(TRAIN_X_NPY):
        print("\n加载已有数据集和词汇表...")
        word2id, id2word = load_vocab(VOCAB_FILE, REVERSE_VOCAB_FILE)
    else:
        word2id, id2word = build_dataset()

    vocab_size = len(word2id)
    print(f"词汇表大小: {vocab_size}")

    # 初始化模型
    print("\n初始化Seq2Seq模型...")
    model = Seq2Seq(vocab_size, CONFIG)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 训练
    train_model(model, word2id, CONFIG, device)

    # 加载最佳模型并生成结果
    best_model_path = os.path.join(MODEL_DIR, f"seq2seq_epoch_{CONFIG['epochs']}.pt")
    if os.path.exists(best_model_path):
        print("\n加载训练好的模型生成摘要...")
        # 兼容PyTorch 2.6，weights_only=False避免自定义对象报错
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=False))
        results = generate_summary(model, word2id, id2word, CONFIG, device)

        # 展示前5条结果
        print("\n前5条生成结果:")
        for i in range(min(5, len(results))):
            print(f"{i + 1}. {results[i]}")