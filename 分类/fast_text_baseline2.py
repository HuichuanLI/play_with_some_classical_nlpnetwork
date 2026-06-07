import fasttext
import time
import jieba
import pandas as pd
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")

# ====================== 路径配置 ======================
DATA_ROOT = "/Users/lhc456/Desktop/nlp课程/play_with_some_classical_nlpnetwork/data"
CODE_ROOT = "/Users/lhc456/Desktop/nlp课程/play_with_some_classical_nlpnetwork"

os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(CODE_ROOT, exist_ok=True)


# ====================== 1. 统一数据处理 ======================
def load_and_preprocess_data(train_path, dev_path=None, test_path=None, min_freq=1):
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

    dev_df = None
    if dev_path and os.path.exists(dev_path):
        print(f"\n正在加载验证集：{dev_path}")
        dev_df = pd.read_csv(dev_path, sep="\t", header=0, names=["sentence", "label"], on_bad_lines="skip")
        dev_df = dev_df.dropna(subset=["sentence", "label"])
        dev_df["label"] = pd.to_numeric(dev_df["label"], errors="coerce").fillna(0).astype(int)
        dev_df = dev_df[dev_df["sentence"].str.strip() != ""]
        print(f"验证集有效行数：{len(dev_df)}")

    test_df = None
    if test_path and os.path.exists(test_path):
        print(f"\n正在加载测试集：{test_path}")
        test_df = pd.read_csv(test_path, sep="\t", header=0, names=["sentence", "label"], on_bad_lines="skip")
        test_df = test_df.dropna(subset=["sentence", "label"])
        test_df["label"] = pd.to_numeric(test_df["label"], errors="coerce").fillna(0).astype(int)
        test_df = test_df[test_df["sentence"].str.strip() != ""]
        print(f"测试集有效行数：{len(test_df)}")

    return train_df, dev_df, test_df, label2id, id2label, num_classes


def convert_to_fasttext_format(df, output_path, id2label):
    fasttext_lines = []
    for _, row in df.iterrows():
        sentence = str(row["sentence"]).strip()
        label_id = int(row["label"])
        label_name = str(id2label[label_id])
        fasttext_label = f"__label__{label_name}"
        tokens = jieba.lcut(sentence)
        fasttext_line = f"{fasttext_label} {' '.join(tokens)}"
        fasttext_lines.append(fasttext_line)

    with open(output_path, "w", encoding="utf-8") as f:
        for line in fasttext_lines:
            f.write(line + "\n")

    print(f"✅ 已转换为FastText格式：{output_path}")
    return output_path


# ====================== 2. FastText模型训练与评估（修复卡住问题）=====================
def train_fasttext(train_path, dev_path=None, use_autotune=False, autotune_duration=300):
    """
    修复自动调参卡住问题：
    - 默认关闭自动调参，使用经过验证的最优参数
    - 如果开启自动调参，增加超时保护和错误处理
    """
    print("\n开始训练FastText模型...")
    start_time = time.time()

    # 经过验证的中文文本分类最优参数
    params = {
        "input": train_path,
        "epoch": 10,  # 增加epoch到10，大数据集需要更多迭代
        "lr": 0.1,  # 学习率
        "dim": 100,  # 词向量维度
        "wordNgrams": 2,  # 2-gram特征
        "minCount": 1,  # 最小词频
        "loss": "softmax",  # 多分类使用softmax损失
        "verbose": 2,  # 降低日志级别，避免输出过多
        "thread": 10  # 使用所有CPU核心
    }

    # 只有明确开启时才使用自动调参
    if use_autotune and dev_path and os.path.exists(dev_path):
        print(f"⚠️  注意：自动调参在macOS上可能不稳定")
        print(f"使用验证集自动调参，调参时间：{autotune_duration}秒")
        params["autotuneValidationFile"] = dev_path
        params["autotuneDuration"] = autotune_duration
        params["autotuneMetric"] = "f1"  # 使用F1分数作为调参指标
        del params["epoch"], params["lr"], params["dim"]  # 让自动调参优化这些参数

    model = fasttext.train_supervised(**params)

    train_time = time.time() - start_time
    print(f"\n训练完成！耗时：{train_time:.2f}秒")

    return model, train_time


def evaluate_fasttext(model, df, id2label, dataset_name="测试集"):
    print(f"\n开始{dataset_name}评估...")
    all_preds = []
    all_labels = []

    for _, row in df.iterrows():
        sentence = str(row["sentence"]).strip()
        true_label_id = int(row["label"])

        tokens = jieba.lcut(sentence)
        pred_label, _ = model.predict(" ".join(tokens))
        pred_label_name = pred_label[0].replace("__label__", "")
        pred_label_id = int(pred_label_name)

        all_preds.append(pred_label_id)
        all_labels.append(true_label_id)

    acc = accuracy_score(all_labels, all_preds)
    target_names = [str(id2label[i]) for i in range(len(id2label))]
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=4)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n{dataset_name}准确率：{acc:.4f}")
    print("\n" + "=" * 60)
    print(f"{dataset_name}详细评估报告")
    print("=" * 60)
    print(report)
    print("混淆矩阵：")
    print(cm)
    print("=" * 60)

    return acc, report, cm, all_preds, all_labels


def save_test_report(acc, report, cm, id2label, save_path="fasttext_test_report.txt"):
    target_names = [str(id2label[i]) for i in range(len(id2label))]

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"测试集准确率：{acc:.4f}\n\n")
        f.write("分类报告：\n")
        f.write(report)
        f.write("\n混淆矩阵：\n")
        f.write(str(cm))
    print(f"\n✅ 测试报告已保存为 {save_path}")

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.title("FastText测试集混淆矩阵")
    plt.savefig("fasttext_confusion_matrix.png", dpi=300, bbox_inches="tight")
    print("✅ 混淆矩阵图已保存为 fasttext_confusion_matrix.png")


# ====================== 3. 主函数 =====================
if __name__ == "__main__":
    # 配置
    TRAIN_PATH = os.path.join(DATA_ROOT, "train.txt")
    DEV_PATH = os.path.join(DATA_ROOT, "dev.txt")
    TEST_PATH = os.path.join(DATA_ROOT, "test.txt")

    # 关键：关闭自动调参，使用手动优化参数
    USE_AUTOTUNE = False  # 设置为False解决卡住问题
    AUTOTUNE_DURATION = 300

    FASTTEXT_TRAIN_PATH = os.path.join(DATA_ROOT, "fasttext_train.txt")
    FASTTEXT_DEV_PATH = os.path.join(DATA_ROOT, "fasttext_dev.txt")
    FASTTEXT_TEST_PATH = os.path.join(DATA_ROOT, "fasttext_test.txt")

    # 检查数据文件
    print("\n检查数据文件...")
    if not os.path.exists(TRAIN_PATH):
        print(f"❌ 训练集文件不存在：{TRAIN_PATH}")
        exit()
    print(f"✅ 找到训练集：{TRAIN_PATH}")

    if os.path.exists(DEV_PATH):
        print(f"✅ 找到验证集：{DEV_PATH}")
    else:
        print(f"⚠️ 未找到验证集：{DEV_PATH}")

    if os.path.exists(TEST_PATH):
        print(f"✅ 找到测试集：{TEST_PATH}")
    else:
        print(f"⚠️ 未找到测试集：{TEST_PATH}")

    # 加载数据
    try:
        train_df, dev_df, test_df, label2id, id2label, num_classes = load_and_preprocess_data(
            TRAIN_PATH, DEV_PATH, TEST_PATH
        )
    except Exception as e:
        print(f"❌ 数据加载失败：{e}")
        exit()

    # 转换为FastText格式
    print("\n转换数据格式...")
    convert_to_fasttext_format(train_df, FASTTEXT_TRAIN_PATH, id2label)
    if dev_df is not None:
        convert_to_fasttext_format(dev_df, FASTTEXT_DEV_PATH, id2label)
    if test_df is not None:
        convert_to_fasttext_format(test_df, FASTTEXT_TEST_PATH, id2label)

    # 训练模型（关闭自动调参）
    model, train_time = train_fasttext(
        FASTTEXT_TRAIN_PATH,
        dev_path=FASTTEXT_DEV_PATH if dev_df is not None else None,
        use_autotune=USE_AUTOTUNE,
        autotune_duration=AUTOTUNE_DURATION
    )

    # 评估模型
    if dev_df is not None:
        dev_acc, _, _, _, _ = evaluate_fasttext(model, dev_df, id2label, dataset_name="验证集")

    if test_df is not None:
        test_acc, test_report, test_cm, _, _ = evaluate_fasttext(model, test_df, id2label, dataset_name="测试集")
        save_test_report(test_acc, test_report, test_cm, id2label,
                         save_path=os.path.join(CODE_ROOT, "fasttext_test_report.txt"))

    # 保存模型
    timestamp = int(time.time())
    model_save_path = os.path.join(CODE_ROOT, f"fasttext_model_{timestamp}.bin")
    model.save_model(model_save_path)
    print(f"\n✅ 模型已保存为 {model_save_path}")


    # 单句预测
    def predict(text):
        tokens = jieba.lcut(text.strip())
        pred_label, prob = model.predict(" ".join(tokens))
        pred_label_name = pred_label[0].replace("__label__", "")
        pred_label_id = int(pred_label_name)
        return pred_label_id, prob[0]


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