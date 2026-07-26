import torch
from config import ModelConfig
from model import DeepseekForCausalLM
from transformers import AutoTokenizer
from data import build_prompt, SYSTEM_PROMPT


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = ModelConfig()

    # 加载模型与分词器
    tokenizer = AutoTokenizer.from_pretrained("./output", padding_side="left")
    model_config.vocab_size = tokenizer.vocab_size
    model = DeepseekForCausalLM(model_config).to(device)
    model.load_state_dict(torch.load("./output/grpo_model.pt", map_location=device))
    model.eval()

    # 测试问题
    test_questions = [
        "1+1等于多少？",
        "小明有5个苹果，吃了2个，又买了3个，现在有几个苹果？"
    ]

    for question in test_questions:
        prompt = build_prompt([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ])
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(input_ids, max_new_tokens=256, temperature=0.7)
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        print("\n" + "=" * 50)
        print(f"问题: {question}")
        print(f"回答:\n{response}")
        print("=" * 50)


if __name__ == "__main__":
    main()
