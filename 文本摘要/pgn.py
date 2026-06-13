import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
import pandas as pd
import numpy as np
import jieba
import re
import os
import time
import heapq
from collections import Counter
from multiprocessing import cpu_count, Pool
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec
import warnings

warnings.filterwarnings("ignore")

# ==================================================
# 1. 全局配置（完全对齐课程参数）
# ==================================================
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
WV_DIR = os.path.join(DATA_DIR, "wv")
MODEL_DIR = os.path.join(ROOT, "models")
RESULT_DIR = os.path.join(ROOT, "results")

for d in [DATA_DIR, WV_DIR, MODEL_DIR, RESULT_DIR]:
    os.makedirs(d, exist_ok=True)

# 原始数据路径
TRAIN_RAW = os.path.join(DATA_DIR, "train.csv")
TEST_RAW = os.path.join(DATA_DIR, "test.csv")
STOP_WORDS_FILE = os.path.join(DATA_DIR, "stopwords.txt")
USER_DICT = os.path.join(DATA_DIR, "user_dict.txt")

# 中间数据路径
TRAIN_SEG = os.path.join(DATA_DIR, "train_seg.csv")
TEST_SEG = os.path.join(DATA_DIR, "test_seg.csv")
TRAIN_TXT = os.path.join(DATA_DIR, "train.txt")
DEV_TXT = os.path.join(DATA_DIR, "dev.txt")
TEST_TXT = os.path.join(DATA_DIR, "test.txt")
WV_MODEL_PATH = os.path.join(WV_DIR, "word2vec_pad.model")
LOSS_PATH = os.path.join(DATA_DIR, "loss.txt")
LOG_PATH = os.path.join(DATA_DIR, "log_train.txt")

# 模型超参数
CONFIG = {
    # 网络结构参数
    "hidden_size": 512,
    "dec_hidden_size": 512,
    "embed_size": 512,
    "pointer": True,  # 开启指针生成机制
    "coverage": False,  # 关闭覆盖机制（baseline版本）

    # 词汇表参数
    "max_vocab_size": 20000,

    # 数据长度参数
    "max_enc_len": 300,
    "max_dec_len": 100,
    "truncate_enc": True,
    "truncate_dec": True,
    "min_dec_steps": 30,
    "max_dec_steps": 50,

    # 训练参数
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.001,
    "max_grad_norm": 2.0,
    "eps": 1e-31,
    "trunc_norm_init_std": 1e-4,
    "LAMBDA": 1,

    # 优化策略
    "fine_tune": False,
    "scheduled_sampling": False,
    "weight_tying": False,

    # 设备
    "device": torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"),
    "is_cuda": torch.cuda.is_available(),

    # 模型保存路径
    "model_save_path": os.path.join(MODEL_DIR, "pgn_model.pt"),
    "encoder_save_name": os.path.join(MODEL_DIR, "model_encoder.pt"),
    "decoder_save_name": os.path.join(MODEL_DIR, "model_decoder.pt"),
    "attention_save_name": os.path.join(MODEL_DIR, "model_attention.pt"),
    "reduce_state_save_name": os.path.join(MODEL_DIR, "model_reduce_state.pt"),
}


# ==================================================
# 2. 工具函数（对应 func_utils.py 全部功能）
# ==================================================
def timer(module):
    """函数耗时计时器装饰器"""

    def wrapper(func):
        def cal_time(*args, **kwargs):
            t1 = time.time()
            res = func(*args, **kwargs)
            t2 = time.time()
            print(f'{t2 - t1:.4f} secs used for {module}')
            return res

        return cal_time

    return wrapper


def simple_tokenizer(text):
    """按空格切分文本"""
    return text.split()


def count_words(counter, text):
    """统计单词词频"""
    for sentence in text:
        for word in sentence:
            counter[word] += 1


def sort_batch_by_len(data_batch):
    """按输入长度降序排列批次数据"""
    res = {'x': [], 'y': [], 'x_len': [], 'y_len': [], 'OOV': [], 'len_OOV': []}
    for i in range(len(data_batch)):
        res['x'].append(data_batch[i]['x'])
        res['y'].append(data_batch[i]['y'])
        res['x_len'].append(len(data_batch[i]['x']))
        res['y_len'].append(len(data_batch[i]['y']))
        res['OOV'].append(data_batch[i]['OOV'])
        res['len_OOV'].append(data_batch[i]['len_OOV'])

    sorted_indices = np.array(res['x_len']).argsort()[::-1].tolist()
    data_batch = {name: [_tensor[i] for i in sorted_indices] for name, _tensor in res.items()}
    return data_batch


def source2ids(source_words, vocab):
    """源文本映射为id，同时记录OOV词列表"""
    ids = []
    oovs = []
    unk_id = vocab.UNK
    for w in source_words:
        i = vocab[w]
        if i == unk_id:
            if w not in oovs:
                oovs.append(w)
            oov_num = oovs.index(w)
            ids.append(vocab.size() + oov_num)
        else:
            ids.append(i)
    return ids, oovs


def abstract2ids(abstract_words, vocab, source_oovs):
    """摘要文本映射为id，支持源文本OOV词映射"""
    ids = []
    unk_id = vocab.UNK
    for w in abstract_words:
        i = vocab[w]
        if i == unk_id:
            if w in source_oovs:
                vocab_idx = vocab.size() + source_oovs.index(w)
                ids.append(vocab_idx)
            else:
                ids.append(unk_id)
        else:
            ids.append(i)
    return ids


def outputids2words(id_list, source_oovs, vocab):
    """输出id映射回自然语言文本"""
    words = []
    for i in id_list:
        try:
            w = vocab.index2word[i]
        except IndexError:
            assert source_oovs is not None, "无法在词典中找到该ID，且无OOV列表"
            source_oov_idx = i - vocab.size()
            try:
                w = source_oovs[source_oov_idx]
            except IndexError:
                raise ValueError(
                    f'模型生成ID: {i}, 对应OOV索引: {source_oov_idx}, 但当前样本只有{len(source_oovs)}个OOV')
        words.append(w)
    return ' '.join(words)


def add2heap(heap, item, k):
    """小顶堆添加元素（用于束搜索，预留）"""
    if len(heap) < k:
        heapq.heappush(heap, item)
    else:
        heapq.heappushpop(heap, item)


def replace_oovs(in_tensor, vocab):
    """将张量中所有OOV词的id替换为UNK"""
    oov_token = torch.full(in_tensor.shape, vocab.UNK, dtype=torch.long).to(CONFIG["device"])
    out_tensor = torch.where(in_tensor > len(vocab) - 1, oov_token, in_tensor)
    return out_tensor


def config_info():
    """打印模型配置信息"""
    info = ('model_name = pgn_model, pointer = {}, coverage = {}, fine_tune = {}, '
            'scheduled_sampling = {}, weight_tying = {}, source = train')
    return info.format(CONFIG["pointer"], CONFIG["coverage"], CONFIG["fine_tune"],
                       CONFIG["scheduled_sampling"], CONFIG["weight_tying"])


# ==================================================
# 3. 词汇表类 Vocab（对应 vocab.py）
# ==================================================
class Vocab(object):
    PAD = 0
    SOS = 1
    EOS = 2
    UNK = 3

    def __init__(self):
        self.word2index = {}
        self.word2count = Counter()
        self.reserved = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        self.index2word = self.reserved[:]
        self.embedding_matrix = None

    def add_words(self, words):
        """向词汇表添加单词"""
        for word in words:
            if word not in self.word2index:
                self.word2index[word] = len(self.index2word)
                self.index2word.append(word)
        self.word2count.update(words)

    def load_embeddings(self, wv_model_path):
        """加载预训练词向量"""
        if not os.path.exists(wv_model_path):
            return
        wv_model = Word2Vec.load(wv_model_path)
        self.embedding_matrix = wv_model.wv.vectors

    def __getitem__(self, item):
        if type(item) is int:
            return self.index2word[item]
        return self.word2index.get(item, self.UNK)

    def __len__(self):
        return len(self.index2word)

    def size(self):
        return len(self.index2word)


# ==================================================
# 4. 数据预处理（生成 train.txt / dev.txt / test.txt）
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


def load_stop_words(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


STOP_WORDS = load_stop_words(STOP_WORDS_FILE)
if os.path.exists(USER_DICT):
    jieba.load_userdict(USER_DICT)


def clean_sentence(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\D(\d\.)\D", "", text)
    text = re.sub(r"[(（]进口[)）]|\(海外\)", "", text)
    text = re.sub(r"车主说|技师说|语音|图片|你好|您好", "", text)
    text = re.sub(r"[^,!?。.\-\u4e00-\u9fa5_a-zA-Z0-9]", "", text)
    return text


def filter_stopwords(seg_list):
    words = [w for w in seg_list if w]
    return [w for w in words if w not in STOP_WORDS]


def sentence_proc(sentence):
    sentence = clean_sentence(sentence)
    words = jieba.cut(sentence)
    words = filter_stopwords(words)
    return " ".join(words)


def sentences_proc(df):
    for col in ["Brand", "Model", "Question", "Dialogue"]:
        if col in df.columns:
            df[col] = df[col].apply(sentence_proc)
    if "Report" in df.columns:
        df["Report"] = df["Report"].apply(sentence_proc)
    return df


def build_raw_data():
    """从原始csv生成训练/验证/测试txt文件"""
    if os.path.exists(TRAIN_TXT) and os.path.exists(DEV_TXT) and os.path.exists(TEST_TXT):
        print("检测到已处理的txt数据，跳过预处理")
        return

    print("=" * 60)
    print("数据预处理：生成 train.txt / dev.txt / test.txt")
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

    # 4. 构造输入输出对
    print("\n4. 构造输入输出对")
    train_df["X"] = train_df["Question"] + " " + train_df["Dialogue"]
    train_df["Y"] = train_df["Report"]
    test_df["X"] = test_df["Question"] + " " + test_df["Dialogue"]

    # 5. 保存为csv并生成txt（用<SEP>分隔）
    print("\n5. 生成txt格式数据")
    # 训练集：70000条训练，剩余12871条验证
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_part = train_df.iloc[:70000]
    dev_part = train_df.iloc[70000:]

    # 写入训练集
    with open(TRAIN_TXT, "w", encoding="utf-8") as f:
        for _, row in train_part.iterrows():
            f.write(f"{row['X']}<SEP>{row['Y']}\n")

    # 写入验证集
    with open(DEV_TXT, "w", encoding="utf-8") as f:
        for _, row in dev_part.iterrows():
            f.write(f"{row['X']}<SEP>{row['Y']}\n")

    # 写入测试集（只有输入）
    with open(TEST_TXT, "w", encoding="utf-8") as f:
        for _, row in test_df.iterrows():
            f.write(f"{row['X']}\n")

    print(f"训练集: 70000 条 | 验证集: {len(dev_part)} 条 | 测试集: {len(test_df)} 条")
    print("✅ 原始数据预处理完成")


# ==================================================
# 5. 数据集类（对应 dataset.py：PairDataset + SampleDataset + collate_fn）
# ==================================================
class PairDataset(object):
    """读取txt文件，构建（编码器输入-解码器输出）文本对"""

    def __init__(self, filename, tokenize=simple_tokenizer, max_enc_len=None, max_dec_len=None,
                 truncate_enc=False, truncate_dec=False):
        print(f"Reading dataset {filename}...", end=' ', flush=True)
        self.filename = filename
        self.pairs = []

        with open(filename, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                pair = line.split('<SEP>')
                if len(pair) != 2:
                    continue

                enc = tokenize(pair[0])
                if max_enc_len and len(enc) > max_enc_len:
                    if truncate_enc:
                        enc = enc[:max_enc_len]

                dec = tokenize(pair[1])
                if max_dec_len and len(dec) > max_dec_len:
                    if truncate_dec:
                        dec = dec[:max_dec_len]

                self.pairs.append((enc, dec))

        print(f"{len(self.pairs)} pairs.")

    def build_vocab(self, embed_file=None):
        """构建词汇表"""
        word_counts = Counter()
        count_words(word_counts, [enc + dec for enc, dec in self.pairs])

        vocab = Vocab()
        vocab.load_embeddings(embed_file)

        # 按词频取前max_vocab_size个词
        for word, count in word_counts.most_common(CONFIG["max_vocab_size"]):
            vocab.add_words([word])

        return vocab


class SampleDataset(Dataset):
    """自定义数据集，返回PGN所需的6个字段"""

    def __init__(self, data_pair, vocab):
        self.src_sents = [x[0] for x in data_pair]
        self.trg_sents = [x[1] for x in data_pair]
        self.vocab = vocab
        self._len = len(data_pair)

    def __getitem__(self, index):
        x, oov = source2ids(self.src_sents[index], self.vocab)
        return {
            'x': [self.vocab.SOS] + x + [self.vocab.EOS],
            'OOV': oov,
            'len_OOV': len(oov),
            'y': [self.vocab.SOS] + abstract2ids(self.trg_sents[index], self.vocab, oov) + [self.vocab.EOS],
            'x_len': len(self.src_sents[index]),
            'y_len': len(self.trg_sents[index])
        }

    def __len__(self):
        return self._len


def collate_fn(batch):
    """自定义批次处理函数：按最大长度填充"""

    def padding(indice, max_length, pad_idx=0):
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    data_batch = sort_batch_by_len(batch)

    x = data_batch['x']
    x_max_length = max([len(t) for t in x])
    y = data_batch['y']
    y_max_length = max([len(t) for t in y])

    OOV = data_batch['OOV']
    len_OOV = torch.tensor(data_batch['len_OOV'])

    x_padded = padding(x, x_max_length)
    y_padded = padding(y, y_max_length)
    x_len = torch.tensor(data_batch['x_len'])
    y_len = torch.tensor(data_batch['y_len'])

    return x_padded, y_padded, x_len, y_len, OOV, len_OOV


# ==================================================
# 6. PGN 模型组件（对应 model.py 全部子层）
# ==================================================
class Encoder(nn.Module):
    """编码器：双向LSTM"""

    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded)
        return output, hidden


class Attention(nn.Module):
    """加性注意力（Bahdanau Attention）"""

    def __init__(self, hidden_units):
        super(Attention, self).__init__()
        self.Wh = nn.Linear(2 * hidden_units, 2 * hidden_units, bias=False)
        self.Ws = nn.Linear(2 * hidden_units, 2 * hidden_units)
        self.v = nn.Linear(2 * hidden_units, 1, bias=False)

    def forward(self, decoder_states, encoder_output, x_padding_masks):
        h_dec, c_dec = decoder_states
        # 拼接h和c得到解码器状态 s_t: (1, batch, 2*hidden)
        s_t = torch.cat([h_dec, c_dec], dim=2)
        # 转换为 (batch, 1, 2*hidden)
        s_t = s_t.transpose(0, 1)
        # 扩展到与编码器输出同形状: (batch, seq_len, 2*hidden)
        s_t = s_t.expand_as(encoder_output).contiguous()

        # 计算注意力得分 e_t
        encoder_features = self.Wh(encoder_output.contiguous())
        decoder_features = self.Ws(s_t)
        attn_inputs = encoder_features + decoder_features
        score = self.v(torch.tanh(attn_inputs)).squeeze(2)

        # softmax + padding mask
        attention_weights = F.softmax(score, dim=1)
        attention_weights = attention_weights * x_padding_masks
        # 重新归一化
        normalization_factor = attention_weights.sum(1, keepdim=True)
        attention_weights = attention_weights / normalization_factor

        # 计算上下文向量
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_output).squeeze(1)
        return context_vector, attention_weights


class Decoder(nn.Module):
    """解码器：单向LSTM + 词汇分布 + 生成概率p_gen"""

    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # 词汇分布全连接层
        self.W1 = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.W2 = nn.Linear(self.hidden_size, vocab_size)

        # 指针生成概率计算层
        if CONFIG["pointer"]:
            self.w_gen = nn.Linear(self.hidden_size * 4 + embed_size, 1)

    def forward(self, x_t, decoder_states, context_vector):
        decoder_emb = self.embedding(x_t)
        decoder_output, decoder_states = self.lstm(decoder_emb, decoder_states)

        # 拼接解码器输出和上下文向量
        decoder_output = decoder_output.view(-1, self.hidden_size)
        concat_vector = torch.cat([decoder_output, context_vector], dim=-1)

        # 计算词汇分布
        FF1_out = self.W1(concat_vector)
        FF2_out = self.W2(FF1_out)
        p_vocab = F.softmax(FF2_out, dim=1)

        # 构造解码器完整状态 s_t
        h_dec, c_dec = decoder_states
        s_t = torch.cat([h_dec, c_dec], dim=2)

        # 计算生成概率 p_gen
        p_gen = None
        if CONFIG["pointer"]:
            x_gen = torch.cat([
                context_vector,
                s_t.squeeze(0),
                decoder_emb.squeeze(1)
            ], dim=-1)
            p_gen = torch.sigmoid(self.w_gen(x_gen))

        return p_vocab, decoder_states, p_gen


class ReduceState(nn.Module):
    """将双向编码器的隐藏状态降维为单向，作为解码器初始状态"""

    def __init__(self):
        super(ReduceState, self).__init__()

    def forward(self, hidden):
        h, c = hidden
        h_reduced = torch.sum(h, dim=0, keepdim=True)
        c_reduced = torch.sum(c, dim=0, keepdim=True)
        return (h_reduced, c_reduced)


# ==================================================
# 7. PGN 完整模型
# ==================================================
class PGN(nn.Module):
    def __init__(self, vocab):
        super(PGN, self).__init__()
        self.vocab = vocab
        self.device = CONFIG["device"]

        self.attention = Attention(CONFIG["hidden_size"])
        self.encoder = Encoder(len(vocab), CONFIG["embed_size"], CONFIG["hidden_size"])
        self.decoder = Decoder(len(vocab), CONFIG["embed_size"], CONFIG["hidden_size"])
        self.reduce_state = ReduceState()

    def get_final_distribution(self, x, p_gen, p_vocab, attention_weights, max_oov):
        """计算最终扩展词汇分布（指针生成核心）"""
        if not CONFIG["pointer"]:
            return p_vocab

        batch_size = x.size()[0]
        # 裁剪p_gen，避免边界值
        p_gen = torch.clamp(p_gen, 0.001, 0.999)

        # 加权词汇分布
        p_vocab_weighted = p_gen * p_vocab
        # 加权注意力分布
        attention_weighted = (1 - p_gen) * attention_weights

        # 扩展词汇表维度，加入OOV位置
        extension = torch.zeros((batch_size, max_oov), dtype=torch.float).to(self.device)
        p_vocab_extended = torch.cat([p_vocab_weighted, extension], dim=1)

        # 将注意力权重累加到对应源单词位置
        final_distribution = p_vocab_extended.scatter_add_(dim=1, index=x, src=attention_weighted)
        return final_distribution

    def forward(self, x, x_len, y, len_oovs, batch=0, num_batches=0, teacher_forcing=True):
        """
        前向传播，返回批次平均损失
        x: 编码器输入 (batch, enc_len)
        y: 解码器目标 (batch, dec_len)
        """
        # 替换OOV为UNK，送入编码器
        x_copy = replace_oovs(x, self.vocab)
        x_padding_masks = torch.ne(x, 0).float()

        # 1. 编码器编码
        encoder_output, encoder_states = self.encoder(x_copy)
        # 降维编码器状态作为解码器初始状态
        decoder_states = self.reduce_state(encoder_states)

        step_losses = []
        # 初始化解码器输入：第一个时间步用SOS
        x_t = y[:, 0]

        # 2. 循环解码
        for t in range(y.shape[1] - 1):
            # Teacher Forcing：使用真实标签作为输入
            if teacher_forcing:
                x_t = y[:, t]

            x_t = replace_oovs(x_t, self.vocab)
            y_t = y[:, t + 1]  # 当前步的目标标签

            # 注意力计算
            context_vector, attention_weights = self.attention(
                decoder_states, encoder_output, x_padding_masks
            )

            # 解码器前向
            p_vocab, decoder_states, p_gen = self.decoder(
                x_t.unsqueeze(1), decoder_states, context_vector
            )

            # 计算最终扩展分布
            final_dist = self.get_final_distribution(
                x, p_gen, p_vocab, attention_weights, torch.max(len_oovs)
            )

            # 非指针模式下，目标标签也要替换OOV
            if not CONFIG["pointer"]:
                y_t = replace_oovs(y_t, self.vocab)

            # 取出目标位置的概率
            target_probs = torch.gather(final_dist, 1, y_t.unsqueeze(1)).squeeze(1)

            # 掩码损失，忽略PAD
            mask = torch.ne(y_t, 0).float()
            # 平滑处理，防止log(0)
            loss = -torch.log(target_probs + CONFIG["eps"])
            loss = loss * mask
            step_losses.append(loss)

            # 非Teacher Forcing模式下，用预测结果作为下一步输入
            if not teacher_forcing:
                x_t = torch.argmax(final_dist, dim=1).to(self.device)

        # 3. 计算批次平均损失
        sample_losses = torch.sum(torch.stack(step_losses, 1), dim=1)
        seq_len_mask = torch.ne(y, 0).float()
        batch_seq_len = torch.sum(seq_len_mask, dim=1)
        batch_loss = torch.mean(sample_losses / batch_seq_len)

        return batch_loss


# ==================================================
# 8. 评估函数
# ==================================================
def evaluate(model, val_data):
    """验证集评估，返回平均损失"""
    print('validating...')
    val_loss = []
    model.eval()

    val_dataloader = DataLoader(
        dataset=val_data,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn
    )

    with torch.no_grad():
        for batch, data in enumerate(tqdm(val_dataloader, desc="验证中")):
            x, y, x_len, y_len, oov, len_oovs = data
            x = x.to(CONFIG["device"])
            y = y.to(CONFIG["device"])
            x_len = x_len.to(CONFIG["device"])
            len_oovs = len_oovs.to(CONFIG["device"])

            loss = model(x, x_len, y, len_oovs, batch=batch, num_batches=len(val_dataloader), teacher_forcing=True)
            val_loss.append(loss.item())

    return np.mean(val_loss)


# ==================================================
# 9. 训练函数
# ==================================================
def train_model(dataset, val_dataset, vocab, start_epoch=0):
    device = CONFIG["device"]
    model = PGN(vocab).to(device)
    print("PGN模型初始化完成")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 构建数据集
    train_data = SampleDataset(dataset.pairs, vocab)
    val_data = SampleDataset(val_dataset.pairs, vocab)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    # 训练数据加载器
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collate_fn
    )

    val_losses = float('inf')
    num_batches = len(train_dataloader)
    teacher_forcing = True
    print(f'teacher_forcing = {teacher_forcing}')
    print(config_info())

    print(f"\n开始训练 | 共{CONFIG['epochs']}轮 | 每轮{num_batches}个批次 | 设备: {device}")
    print("-" * 60)

    for epoch in range(start_epoch, CONFIG["epochs"]):
        model.train()
        batch_losses = []
        start_time = time.time()

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']}", unit="batch")
        for batch, data in enumerate(pbar):
            x, y, x_len, y_len, oov, len_oovs = data
            x = x.to(device)
            y = y.to(device)
            x_len = x_len.to(device)
            len_oovs = len_oovs.to(device)

            optimizer.zero_grad()
            loss = model(x, x_len, y, len_oovs, batch=batch, num_batches=num_batches, teacher_forcing=teacher_forcing)
            loss.backward()

            # 梯度裁剪，防止爆炸
            clip_grad_norm_(model.encoder.parameters(), CONFIG["max_grad_norm"])
            clip_grad_norm_(model.decoder.parameters(), CONFIG["max_grad_norm"])
            clip_grad_norm_(model.attention.parameters(), CONFIG["max_grad_norm"])

            optimizer.step()
            batch_losses.append(loss.item())

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{np.mean(batch_losses):.4f}"
            })

        epoch_loss = np.mean(batch_losses)
        epoch_time = time.time() - start_time

        # 验证集评估
        avg_val_loss = evaluate(model, val_data)

        print(
            f"\nEpoch {epoch + 1} 完成 | 训练损失: {epoch_loss:.4f} | 验证损失: {avg_val_loss:.4f} | 耗时: {epoch_time:.1f}s")

        # 保存最优模型
        if avg_val_loss < val_losses:
            torch.save(model.state_dict(), CONFIG["model_save_path"])
            torch.save(model.encoder, CONFIG["encoder_save_name"])
            torch.save(model.decoder, CONFIG["decoder_save_name"])
            torch.save(model.attention, CONFIG["attention_save_name"])
            torch.save(model.reduce_state, CONFIG["reduce_state_save_name"])
            val_losses = avg_val_loss
            print(f"✅ 验证损失下降，模型已保存 | 最优验证损失: {val_losses:.4f}")

        print("-" * 60)

    print(f"\n训练完成 | 最优验证损失: {val_losses:.4f}")
    return model


# ==================================================
# 10. 预测类（贪心解码）
# ==================================================
class Predict(object):
    @timer(module="初始化预测器")
    def __init__(self, vocab):
        self.device = CONFIG["device"]
        self.vocab = vocab
        self.model = PGN(vocab)
        self.model.load_state_dict(torch.load(CONFIG["model_save_path"], map_location=self.device, weights_only=False))
        self.model.to(self.device)
        self.model.eval()

    def greedy_search(self, x, max_sum_len, len_oovs, x_padding_masks):
        """贪心解码生成摘要"""
        encoder_output, encoder_states = self.model.encoder(replace_oovs(x, self.vocab))
        decoder_states = self.model.reduce_state(encoder_states)

        # 初始输入为SOS
        x_t = torch.ones(1) * self.vocab.SOS
        x_t = x_t.to(self.device, dtype=torch.int64)
        summary = [self.vocab.SOS]

        # 循环解码
        while int(x_t.item()) != self.vocab.EOS and len(summary) < max_sum_len:
            context_vector, attention_weights = self.model.attention(
                decoder_states, encoder_output, x_padding_masks
            )
            p_vocab, decoder_states, p_gen = self.model.decoder(
                x_t.unsqueeze(1), decoder_states, context_vector
            )
            final_dist = self.model.get_final_distribution(
                x, p_gen, p_vocab, attention_weights, torch.max(len_oovs)
            )
            # 贪心取最大概率
            x_t = torch.argmax(final_dist, dim=1).to(self.device)
            decoder_word_idx = x_t.item()
            summary.append(decoder_word_idx)
            x_t = replace_oovs(x_t, self.vocab)

        return summary

    @timer(module="单条预测")
    def predict(self, text):
        """输入原始文本，输出摘要"""
        if isinstance(text, str):
            text = list(jieba.cut(clean_sentence(text)))

        x, oov = source2ids(text, self.vocab)
        x = torch.tensor([x]).to(self.device)
        len_oovs = torch.tensor([len(oov)]).to(self.device)
        x_padding_masks = torch.ne(x, 0).float()

        summary = self.greedy_search(
            x, CONFIG["max_dec_steps"], len_oovs, x_padding_masks
        )
        summary = outputids2words(summary, oov, self.vocab)
        # 移除特殊标记
        return summary.replace('<SOS>', '').replace('<EOS>', '').strip()


# ==================================================
# 11. 主函数
# ==================================================
if __name__ == "__main__":
    print(f"使用设备: {CONFIG['device']}")

    # 步骤1：原始数据预处理
    build_raw_data()

    # 步骤2：构建文本对数据集与词汇表
    print("\n" + "=" * 60)
    print("构建数据集与词汇表")
    print("=" * 60)

    train_dataset = PairDataset(
        TRAIN_TXT,
        max_enc_len=CONFIG["max_enc_len"],
        max_dec_len=CONFIG["max_dec_len"],
        truncate_enc=CONFIG["truncate_enc"],
        truncate_dec=CONFIG["truncate_dec"]
    )

    val_dataset = PairDataset(
        DEV_TXT,
        max_enc_len=CONFIG["max_enc_len"],
        max_dec_len=CONFIG["max_dec_len"],
        truncate_enc=CONFIG["truncate_enc"],
        truncate_dec=CONFIG["truncate_dec"]
    )

    vocab = train_dataset.build_vocab(embed_file=WV_MODEL_PATH)
    print(f"词汇表大小: {vocab.size()}")

    # 步骤3：训练模型
    print("\n" + "=" * 60)
    print("开始训练PGN模型")
    print("=" * 60)
    model = train_model(train_dataset, val_dataset, vocab)

    # 步骤4：预测示例
    print("\n" + "=" * 60)
    print("预测示例")
    print("=" * 60)

    predictor = Predict(vocab)

    # 从验证集随机选3条测试
    with open(DEV_TXT, "r", encoding="utf-8") as f:
        lines = f.readlines()
        import random

        samples = random.sample(lines, 3)

    for i, line in enumerate(samples):
        source, ref = line.strip().split('<SEP>')
        pred = predictor.predict(source)
        print(f"\n--- 示例 {i + 1} ---")
        print(f"原文: {source[:100]}...")
        print(f"参考摘要: {ref}")
        print(f"生成摘要: {pred}")
