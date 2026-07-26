import torch
from config import ModelConfig, GRPOConfig
from model import DeepseekForCausalLM
from data import prepare_dataset_from_hf
from trainer import GRPOTrainer
from transformers import AutoTokenizer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"使用设备: {device}")

    # 1. 加载配置
    model_config = ModelConfig()
    grpo_config = GRPOConfig()

    # 2. 加载分词器与模型
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model_config.vocab_size = tokenizer.vocab_size

    model = DeepseekForCausalLM(model_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params / 1e6:.2f}M")

    # 3. 加载数据集
    from datasets import load_dataset
    import os

    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    train_data = prepare_dataset_from_hf(split="train")
    test_data = prepare_dataset_from_hf(split="test")  # 测试集可留作后续评估

    print(f"训练集大小: {len(train_data)}")

    # 4. 启动训练
    trainer = GRPOTrainer(model, tokenizer, train_data, grpo_config, device)
    trainer.train()


if __name__ == "__main__":
    main()
