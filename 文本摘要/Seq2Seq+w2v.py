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
from gensim.models.word2vec import Word2Vec, LineSentence
import warnings

warnings.filterwarnings("ignore")

# ==================================================
# 1. 全局配置（3.1基线 / 3.2预训练词向量 一键切换）
# ==================================================
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
RESULT_DIR = os.path.join(ROOT, "results")
WV_DIR = os.path.join(DATA_DIR, "wv")

for d in [DATA_DIR, MODEL_DIR, RESULT_DIR, WV_DIR]:
    os.makedirs(d, exist_ok=True)

# 数据文件路径
TRAIN_RAW = os.path.join(DATA_DIR, "train.csv")
TEST_RAW = os.path.join(DATA_DIR, "test.csv")
STOP_WORDS_FILE = os.path.join(DATA_DIR, "stopwords.txt")
USER_DICT = os.path.join(DATA_DIR, "user_dict.txt")

# 预处理缓存路径
TRAIN_SEG = os.path.join(DATA_DIR, "train_seg.csv")
TEST_SEG = os.path.join(DATA_DIR, "test_seg.csv")
MERGED_SEG = os.path.join(DATA_DIR, "merged_seg.csv")
TRAIN_X_PAD = os.path.join(DATA_DIR, "train_x_pad.csv")
TRAIN_Y_PAD = os.path.join(DATA_DIR, "train_y_pad.csv")
TEST_X_PAD = os.path.join(DATA_DIR, "test_x_pad.csv")
TRAIN_X_NPY = os.path.join(DATA_DIR, "train_X.npy")
TRAIN_Y_NPY = os.path.join(DATA_DIR, "train_Y.npy")
TEST_X_NPY = os.path.join(DATA_DIR, "test_X.npy")
VOCAB_FILE = os.path.join(WV_DIR, "vocab.txt")
REVERSE_VOCAB_FILE = os.path.join(WV_DIR, "reverse_vocab.txt")
WV_MODEL_PATH = os.path.join(WV_DIR, "word2vec.model")

# 超参数（完全对齐课程默认值）
CONFIG = {
    "max_enc_len": 300,
    "max_dec_len": 50,
    "batch_size": 64,
    "seq2seq_epochs": 20,
    "wv_train_epochs": 10,
    "embed_size": 500,
    "enc_units": 512,
    "dec_units": 512,
    "attn_units": 20,
    "learning_rate": 0.001,
    "use_pretrained_wv": True,  # True=3.2预训练词向量优化版；False=3.1基线版
    "min_word_count": 5
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
    """保存词汇表为txt"""
    with open(path, "w", encoding="utf-8") as f:
        for w, idx in word2id.items():
            f.write(f"{w}\t{idx}\n")


def load_vocab(vocab_path, reverse_path):
    """鲁棒加载词汇表"""
    word2id, id2word = {}, {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            w, idx = line.split("\t", 1)
            try:
                word2id[w] = int(idx)
            except ValueError:
                continue
    with open(reverse_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            idx, w = line.split("\t", 1)
            try:
                id2word[int(idx)] = w
            except ValueError:
                continue
    return word2id, id2word


# ==================================================
# 3. 数据预处理（完全对齐课程流程 + Word2Vec预训练）
# ==================================================
def load_stop_words(path):
    """加载停用词"""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


STOP_WORDS = load_stop_words(STOP_WORDS_FILE)
if os.path.exists(USER_DICT):
    jieba.load_userdict(USER_DICT)


def clean_sentence(text):
    """文本清洗"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\D(\d\.)\D", "", text)
    text = re.sub(r"[(（]进口[)）]|\(海外\)", "", text)
    text = re.sub(r"车主说|技师说|语音|图片|你好|您好", "", text)
    text = re.sub(r"[^,!?。.\-\u4e00-\u9fa5_a-zA-Z0-9]", "", text)
    return text


def filter_stopwords(seg_list):
    """过滤停用词"""
    words = [w for w in seg_list if w]
    return [w for w in words if w not in STOP_WORDS]


def sentence_proc(sentence):
    """单句处理：清洗→分词→去停用词"""
    sentence = clean_sentence(sentence)
    words = jieba.cut(sentence)
    words = filter_stopwords(words)
    return " ".join(words)


def sentences_proc(df):
    """批量处理DataFrame列"""
    for col in ["Brand", "Model", "Question", "Dialogue"]:
        if col in df.columns:
            df[col] = df[col].apply(sentence_proc)
    if "Report" in df.columns:
        df["Report"] = df["Report"].apply(sentence_proc)
    return df


def get_max_len(data):
    """计算合适的最大长度（均值+2倍标准差）"""
    max_lens = data.apply(lambda x: x.count(" ") + 1)
    return int(np.mean(max_lens) + 2 * np.std(max_lens))


def pad_proc(sentence, max_len, word_to_id):
    """填充START/STOP/PAD/UNK"""
    words = sentence.strip().split(" ")
    words = words[:max_len]
    sentence = [w if w in word_to_id else "<UNK>" for w in words]
    sentence = ["<START>"] + sentence + ["<STOP>"]
    # 补齐到 max_len + 2（START+STOP占2位）
    sentence += ["<PAD>"] * (max_len + 2 - len(sentence))
    return " ".join(sentence)


def transform_data(sentence, word_to_id):
    """句子转id序列"""
    words = sentence.split(" ")
    return [word_to_id[w] if w in word_to_id else word_to_id["<UNK>"] for w in words]


def build_dataset():
    """完整数据预处理 + Word2Vec预训练（仅首次运行）"""
    print("=" * 60)
    print("首次运行：构建数据集 + 预训练词向量")
    print("=" * 60)

    # 1. 加载原始数据
    print("\n1. 加载原始数据")
    train_df = pd.read_csv(TRAIN_RAW, engine="python", encoding="utf-8")
    test_df = pd.read_csv(TEST_RAW, engine="python", encoding="utf-8")
    print(f"原始训练集: {len(train_df)} 条 | 测试集: {len(test_df)} 条")

    # 2. 去空值
    print("\n2. 去除空值")
    train_df.dropna(subset=["Question", "Dialogue", "Report"], how="any", inplace=True)
    test_df.dropna(subset=["Question", "Dialogue"], how="any", inplace=True)
    print(f"去空后训练集: {len(train_df)} 条")

    # 3. 多进程预处理
    print("\n3. 多进程文本预处理")
    train_df = parallelize(train_df, sentences_proc)
    test_df = parallelize(test_df, sentences_proc)
    print("预处理完成")

    # 4. 合并数据用于构建词典/训练词向量
    print("\n4. 合并训练测试集")
    train_df["merged"] = train_df[["Question", "Dialogue", "Report"]].apply(lambda x: " ".join(x), axis=1)
    test_df["merged"] = test_df[["Question", "Dialogue"]].apply(lambda x: " ".join(x), axis=1)
    merged_df = pd.concat([train_df[["merged"]], test_df[["merged"]]], axis=0)
    print(f"训练集 {len(train_df)} 条 | 测试集 {len(test_df)} 条 | 合并集 {len(merged_df)} 条")

    # 5. 保存分词后数据
    print("\n5. 保存分词结果")
    train_df.drop(["merged"], axis=1, inplace=True)
    test_df.drop(["merged"], axis=1, inplace=True)
    train_df.to_csv(TRAIN_SEG, index=None, header=True)
    test_df.to_csv(TEST_SEG, index=None, header=True)
    merged_df.to_csv(MERGED_SEG, index=None, header=False)
    print("分词数据已保存")

    # ========== 3.2 新增：训练Word2Vec词向量 ==========
    if CONFIG["use_pretrained_wv"]:
        print("\n6. 预训练Word2Vec词向量")
        wv_model = Word2Vec(
            LineSentence(MERGED_SEG),
            vector_size=CONFIG["embed_size"],
            negative=5,
            workers=cpu_count(),
            epochs=CONFIG["wv_train_epochs"],
            window=3,
            min_count=CONFIG["min_word_count"]
        )
        print(f"初始词向量训练完成，词典大小: {len(wv_model.wv.key_to_index)}")

        # 构建初始词典
        word_to_id = {word: idx for idx, word in enumerate(wv_model.wv.key_to_index)}
    else:
        # 3.1 基线版：统计词频构建词典
        print("\n6. 统计词频构建词典")
        word_count = {}
        with open(MERGED_SEG, "r", encoding="utf-8") as f:
            for line in f:
                for w in line.strip().split(" "):
                    word_count[w] = word_count.get(w, 0) + 1
        # 过滤低频词
        word_to_id = {}
        for w, freq in word_count.items():
            if freq >= CONFIG["min_word_count"]:
                word_to_id[w] = len(word_to_id)
        print(f"过滤低频词后词典大小: {len(word_to_id)}")

    # 7. 构造输入输出并填充特殊标记
    print("\n7. 构造输入输出并填充特殊标记")
    train_df["X"] = train_df["Question"] + " " + train_df["Dialogue"]
    test_df["X"] = test_df["Question"] + " " + test_df["Dialogue"]

    x_max_len = min(get_max_len(pd.concat([train_df["X"], test_df["X"]])), CONFIG["max_enc_len"])
    y_max_len = min(get_max_len(train_df["Report"]), CONFIG["max_dec_len"])
    print(f"输入最大长度: {x_max_len} | 输出最大长度: {y_max_len}")

    train_df["X"] = train_df["X"].apply(lambda x: pad_proc(x, x_max_len, word_to_id))
    test_df["X"] = test_df["X"].apply(lambda x: pad_proc(x, x_max_len, word_to_id))
    train_df["Y"] = train_df["Report"].apply(lambda x: pad_proc(x, y_max_len, word_to_id))

    # 保存填充后数据
    train_df["X"].to_csv(TRAIN_X_PAD, index=None, header=False)
    train_df["Y"].to_csv(TRAIN_Y_PAD, index=None, header=False)
    test_df["X"].to_csv(TEST_X_PAD, index=None, header=False)
    print("填充后数据已保存")

    # ========== 3.2 新增：二次训练词向量，加入特殊标记 ==========
    if CONFIG["use_pretrained_wv"]:
        print("\n8. 二次训练词向量（加入特殊标记）")
        wv_model.build_vocab(LineSentence(TRAIN_X_PAD), update=True)
        wv_model.train(LineSentence(TRAIN_X_PAD), epochs=CONFIG["wv_train_epochs"],
                       total_examples=wv_model.corpus_count)
        print("  1/3 train_x 完成")

        wv_model.build_vocab(LineSentence(TRAIN_Y_PAD), update=True)
        wv_model.train(LineSentence(TRAIN_Y_PAD), epochs=CONFIG["wv_train_epochs"],
                       total_examples=wv_model.corpus_count)
        print("  2/3 train_y 完成")

        wv_model.build_vocab(LineSentence(TEST_X_PAD), update=True)
        wv_model.train(LineSentence(TEST_X_PAD), epochs=CONFIG["wv_train_epochs"], total_examples=wv_model.corpus_count)
        print("  3/3 test_x 完成")

        wv_model.save(WV_MODEL_PATH)
        print(f"词向量模型已保存，最终词典大小: {len(wv_model.wv.key_to_index)}")

        # 更新词典
        word_to_id = {word: idx for idx, word in enumerate(wv_model.wv.key_to_index)}
        id_to_word = {idx: word for idx, word in enumerate(wv_model.wv.key_to_index)}
    else:
        # 3.1 基线版：重新统计包含特殊标记的词典
        print("\n8. 更新词典（加入特殊标记）")
        word_to_id = {}
        count = 0
        for path in [TRAIN_X_PAD, TRAIN_Y_PAD, TEST_X_PAD]:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    for w in line.strip().split(" "):
                        if w not in word_to_id:
                            word_to_id[w] = count
                            count += 1
        id_to_word = {v: k for k, v in word_to_id.items()}
        print(f"最终词典大小: {len(word_to_id)}")

    # 9. 保存词汇表
    print("\n9. 保存词汇表")
    save_vocab(VOCAB_FILE, word_to_id)
    save_vocab(REVERSE_VOCAB_FILE, id_to_word)

    # 10. 转id并保存numpy
    print("\n10. 转换为numpy数组")
    train_ids_x = train_df["X"].apply(lambda x: transform_data(x, word_to_id))
    train_ids_y = train_df["Y"].apply(lambda x: transform_data(x, word_to_id))
    test_ids_x = test_df["X"].apply(lambda x: transform_data(x, word_to_id))

    train_X = np.array(train_ids_x.tolist())
    train_Y = np.array(train_ids_y.tolist())
    test_X = np.array(test_ids_x.tolist())

    np.save(TRAIN_X_NPY, train_X)
    np.save(TRAIN_Y_NPY, train_Y)
    np.save(TEST_X_NPY, test_X)

    print(f"train_X shape: {train_X.shape}")
    print(f"train_Y shape: {train_Y.shape}")
    print(f"test_X shape: {test_X.shape}")
    print("\n✅ 数据集构建完成")
    return word_to_id, id_to_word


def get_dataloader(mode="train", batch_size=64):
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
# 4. 模型层（支持预训练词向量加载，完全对齐课程架构）
# ==================================================
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size, embedding_matrix=None):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.enc_units = enc_units
        self.batch_size = batch_size

        # 词嵌入层：预训练/随机初始化二选一
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=enc_units,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x, h0):
        x = self.embedding(x)
        output, hn = self.gru(x, h0)
        # 对齐课程写法：(num_layers, batch, hidden) → (batch, 1, hidden)
        return output, hn.transpose(1, 0)

    def initialize_hidden_state(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return torch.zeros(1, batch_size, self.enc_units)


class Attention(nn.Module):
    def __init__(self, enc_units, dec_units, attn_units):
        super(Attention, self).__init__()
        self.enc_units = enc_units
        self.dec_units = dec_units
        self.attn_units = attn_units
        self.w1 = nn.Linear(enc_units, attn_units)
        self.w2 = nn.Linear(dec_units, attn_units)
        self.v = nn.Linear(attn_units, 1)

    def forward(self, query, value):
        # query: (batch, 1, dec_units)  解码器上一步隐藏状态
        # value: (batch, enc_seq_len, enc_units)  编码器全部输出
        score = self.v(torch.tanh(self.w1(value) + self.w2(query)))
        attention_weights = F.softmax(score, dim=1)
        context_vector = torch.sum(attention_weights * value, dim=1)
        return context_vector, attention_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size, embedding_matrix=None):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dec_units = dec_units
        self.batch_size = batch_size

        # 词嵌入层：预训练/随机初始化二选一
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.gru = nn.GRU(
            input_size=embedding_dim + dec_units,
            hidden_size=dec_units,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(dec_units, vocab_size)

    def forward(self, x, context_vector):
        x = self.embedding(x)
        # context_vector: (batch, dec_units) → 升维后与x拼接
        x = torch.cat([torch.unsqueeze(context_vector, 1), x], dim=-1)
        output, hn = self.gru(x)
        output = output.squeeze(1)
        prediction = self.fc(output)
        return prediction, hn.transpose(1, 0)


class Seq2Seq(nn.Module):
    def __init__(self, params, embedding_matrix=None):
        super(Seq2Seq, self).__init__()
        self.params = params

        self.encoder = Encoder(
            params["vocab_size"], params["embed_size"],
            params["enc_units"], params["batch_size"],
            embedding_matrix
        )
        self.attention = Attention(
            params["enc_units"], params["dec_units"], params["attn_units"]
        )
        self.decoder = Decoder(
            params["vocab_size"], params["embed_size"],
            params["dec_units"], params["batch_size"],
            embedding_matrix
        )

    def forward(self, dec_input, dec_hidden, enc_output, dec_target):
        predictions = []
        # 初始上下文向量
        context_vector, _ = self.attention(dec_hidden, enc_output)

        for t in range(dec_target.shape[1]):
            pred, dec_hidden = self.decoder(dec_input, context_vector)
            context_vector, _ = self.attention(dec_hidden, enc_output)
            # Teacher Forcing：使用真实标签作为下一步输入
            dec_input = dec_target[:, t].unsqueeze(1)
            predictions.append(pred)

        return torch.stack(predictions, 1), dec_hidden


# ==================================================
# 5. 训练函数（带进度条，对齐课程损失逻辑）
# ==================================================
def train_model(model, word_to_id, params, device):
    epochs = params["seq2seq_epochs"]
    batch_size = params["batch_size"]
    pad_index = word_to_id["<PAD>"]
    unk_index = word_to_id["<UNK>"]
    start_index = word_to_id["<START>"]

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # 对齐课程：掩码损失，忽略PAD和UNK
    def loss_function(pred, real):
        pad_mask = torch.eq(real, pad_index)
        unk_mask = torch.eq(real, unk_index)
        mask = torch.logical_not(torch.logical_or(pad_mask, unk_mask))
        pred = pred.transpose(2, 1)
        real = real * mask
        loss_ = criterion(pred, real)
        return torch.mean(loss_)

    train_loader = get_dataloader("train", batch_size)
    steps_per_epoch = len(train_loader)

    print(f"\n开始训练 | 共{epochs}轮 | 每轮{steps_per_epoch}个批次 | 设备: {device}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(device)
            targets = targets.to(device).type_as(inputs)

            optimizer.zero_grad()

            # 编码器前向
            initial_hidden = model.encoder.initialize_hidden_state(inputs.shape[0]).to(device)
            enc_output, enc_hidden = model.encoder(inputs, initial_hidden)

            # 解码器初始输入
            dec_input = torch.tensor([start_index] * inputs.shape[0]).unsqueeze(1).to(device)
            dec_hidden = enc_hidden

            # 解码器前向
            predictions, _ = model(dec_input, dec_hidden, enc_output, targets)

            # 计算损失
            loss = loss_function(predictions, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / (batch_idx + 1):.4f}"
            })

        avg_loss = total_loss / steps_per_epoch
        epoch_time = time.time() - start_time

        # 每2轮保存一次模型
        if (epoch + 1) % 2 == 0:
            save_path = os.path.join(MODEL_DIR, f"seq2seq_epoch_{epoch + 1}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"✅ 模型已保存: {save_path}")

        print(f"Epoch {epoch + 1} 完成 | 平均损失: {avg_loss:.4f} | 耗时: {epoch_time:.1f}s")
        print("-" * 60)


# ==================================================
# 6. 测试生成函数（贪心解码，带进度条）
# ==================================================
def greedy_decode(model, word_to_id, id_to_word, params, device):
    model.eval()
    test_loader = get_dataloader("test", params["batch_size"])
    start_idx = word_to_id["<START>"]
    stop_idx = word_to_id["<STOP>"]

    results = []
    print("\n开始生成测试集摘要...")

    with torch.no_grad():
        for (enc_x,) in tqdm(test_loader, desc="生成中", unit="batch"):
            enc_x = enc_x.to(device)
            batch_size = enc_x.shape[0]
            predicts = [""] * batch_size

            # 编码器编码
            initial_hidden = torch.zeros(1, batch_size, model.encoder.enc_units).to(device)
            enc_output, enc_hidden = model.encoder(enc_x, initial_hidden)

            dec_input = torch.tensor([start_idx] * batch_size).unsqueeze(1).to(device)
            dec_hidden = enc_hidden

            for t in range(params["max_dec_len"]):
                context_vector, _ = model.attention(dec_hidden, enc_output)
                predictions, dec_hidden = model.decoder(dec_input, context_vector)

                # 贪心解码
                predict_ids = torch.argmax(predictions, dim=1)
                for i, p_id in enumerate(predict_ids.cpu().numpy()):
                    predicts[i] += id_to_word[p_id] + " "

                dec_input = predict_ids.unsqueeze(1)

            # 后处理
            for pred in predicts:
                pred = pred.strip()
                if "<STOP>" in pred:
                    pred = pred[:pred.index("<STOP>")]
                pred = pred.replace(" ", "")
                results.append(pred)

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
    # 设备自动选择
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
    CONFIG["vocab_size"] = vocab_size
    print(f"词汇表大小: {vocab_size}")

    # 加载预训练词向量矩阵（3.2版本）
    embedding_matrix = None
    if CONFIG["use_pretrained_wv"] and os.path.exists(WV_MODEL_PATH):
        print("加载预训练Word2Vec词向量...")
        wv_model = Word2Vec.load(WV_MODEL_PATH)
        embedding_matrix = torch.from_numpy(wv_model.wv.vectors).float()

    # 初始化模型
    print("\n初始化Seq2Seq模型...")
    model = Seq2Seq(CONFIG, embedding_matrix)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 训练模型
    train_model(model, word2id, CONFIG, device)

    # 加载最后一轮模型生成结果
    last_model_path = os.path.join(MODEL_DIR, f"seq2seq_epoch_{CONFIG['seq2seq_epochs']}.pt")
    if os.path.exists(last_model_path):
        print("\n加载训练完成的模型生成摘要...")
        model.load_state_dict(torch.load(last_model_path, map_location=device, weights_only=False))
        results = greedy_decode(model, word2id, id2word, CONFIG, device)

        # 展示前10条结果
        print("\n前10条生成结果:")
        for i in range(min(10, len(results))):
            print(f"{i + 1}. {results[i]}")
