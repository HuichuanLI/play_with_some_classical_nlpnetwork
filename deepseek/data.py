import re
import pandas as pd
from torch.utils.data import Dataset

SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
</reasoning>


"""


def build_prompt(messages):
    return "\n".join([msg["content"].strip() for msg in messages])


def extract_answer_from_model_output(text):
    """从模型输出中提取answer标签内容，对应文档GRPO部分"""
    parts = text.split("")
    if len(parts) < 2:
        return None
    last_part = parts[-1]
    if "" not in last_part:
        return None
    answer = last_part.split("")[0].strip()
    return None if answer == "..." else answer


def extract_answer_from_dataset(text):
    """从GSM8K数据集中提取答案"""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def extract_single_number(text):
    numbers = re.findall(r'-?\d*\.?\d+', str(text))
    return float(numbers[0]) if len(numbers) == 1 else None


def prepare_dataset(data_path):
    """加载GSM8K数据集并格式化"""
    df = pd.read_parquet(data_path)
    formatted_data = []
    for _, row in df.iterrows():
        prompt_str = build_prompt([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["question"]}
        ])
        answer = extract_answer_from_dataset(row["answer"])
        formatted_data.append({"prompt": prompt_str, "answer": answer})
    return formatted_data


# ========== 奖励函数 ==========
def correctness_reward(prompts, completions, answer):
    """正确性奖励：答案正确得高分，对应文档GRPO奖励"""
    responses = [completion[0]['content'] for completion in completions]
    extracted = [extract_answer_from_model_output(r) for r in responses]
    rewards = []
    for r, a in zip(extracted, answer):
        if r == a:
            rewards.append(2.0)
        else:
            r_num = extract_single_number(r)
            a_num = extract_single_number(a)
            if r_num is not None and a_num is not None and r_num == a_num:
                rewards.append(1.5)
            else:
                rewards.append(0.0)
    return rewards


def format_reward(completions):
    """格式奖励：检查是否符合指定输出格式"""
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    for response in responses:
        score = 0.0
        if "<reasoning>" in response: score += 0.2
        if "</reasoning>" in response: score += 0.2
        if "" in response: score += 0.2
        if "" in response: score += 0.2
        rewards.append(score)
    return rewards


def combined_reward(prompts, completions, answer):
    """合并正确性奖励与格式奖励"""
    correctness_scores = correctness_reward(prompts=prompts, completions=completions, answer=answer)
    format_scores = format_reward(completions=completions)
    return [c + f for c, f in zip(correctness_scores, format_scores)]


class GRPODataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
