import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
from data import combined_reward


def get_per_token_logps(model, input_ids, attention_mask):
    """
    计算逐token对数概率，对应文档GRPO第1步核心代码
    """
    logits, _, _ = model(input_ids, attention_mask=attention_mask)
    logits = logits[:, :-1, :]  # 去掉最后一个logit
    input_ids = input_ids[:, 1:]  # 去掉第一个token

    per_token_logps = []
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)


def create_completion_mask(completion_ids, eos_token_id):
    """生成补全部分mask，EOS之后的token不计入损失，对应文档GRPO代码"""
    is_eos = completion_ids == eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
    mask_exists = is_eos.any(dim=1)
    eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]
    sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
    return (sequence_indices <= eos_idx.unsqueeze(1)).int()


def compute_grpo_advantage(rewards, num_generations):
    """组相对优势计算，对应文档GRPO公式3"""
    rewards = rewards.view(-1, num_generations)
    mean_r = rewards.mean(dim=1, keepdim=True)
    std_r = rewards.std(dim=1, keepdim=True)
    advantages = (rewards - mean_r) / (std_r + 1e-4)
    return advantages.flatten()


def compute_kl_divergence(pi_logps, ref_logps):
    """计算逐token KL散度，对应文档GRPO KL公式"""
    return torch.exp(ref_logps - pi_logps) - (ref_logps - pi_logps) - 1


class GRPOTrainer:
    def __init__(self, model, tokenizer, train_data, config, device):
        self.model = model
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.config = config
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    def generate_completions(self, prompts):
        """为每个prompt生成num_generations个补全"""
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").to(self.device)
        prompt_ids = inputs["input_ids"]
        prompt_mask = inputs["attention_mask"]
        prompt_length = prompt_ids.size(1)

        # 重复prompt对应生成次数
        prompt_ids = prompt_ids.repeat_interleave(self.config.num_generations, dim=0)
        prompt_mask = prompt_mask.repeat_interleave(self.config.num_generations, dim=0)

        # 生成补全
        output_ids = self.model.generate(
            prompt_ids,
            max_new_tokens=self.config.max_completion_length,
            temperature=1.0,
            do_sample=True
        )
        completion_ids = output_ids[:, prompt_length:]
        completion_mask = create_completion_mask(completion_ids, self.tokenizer.eos_token_id)

        return prompt_ids, prompt_mask, completion_ids, completion_mask

    def train_step(self, batch_samples):
        """单步GRPO训练，对应文档算法流程"""
        prompts = [sample["prompt"] for sample in batch_samples]
        answers = [sample["answer"] for sample in batch_samples]

        # 1. 生成补全，计算旧策略与参考策略对数概率
        self.model.eval()
        with torch.no_grad():
            prompt_ids, prompt_mask, completion_ids, completion_mask = self.generate_completions(prompts)
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            logits_to_keep = completion_ids.size(1)

            old_log_probs = get_per_token_logps(self.model, input_ids, attention_mask)[:, -logits_to_keep:]
            ref_log_probs = old_log_probs.clone()  # 简化版，正式训练需单独ref_model

            # 计算奖励
            completions_text = [[{'content': self.tokenizer.decode(ids, skip_special_tokens=True)}] for ids in
                                completion_ids]
            repeated_prompts = [p for p in prompts for _ in range(self.config.num_generations)]
            repeated_answers = [a for a in answers for _ in range(self.config.num_generations)]
            rewards = torch.tensor(
                combined_reward(repeated_prompts, completions_text, repeated_answers),
                dtype=torch.float32, device=self.device
            )
            advantages = compute_grpo_advantage(rewards, self.config.num_generations)

        # 2. 策略更新
        self.model.train()
        for _ in range(self.config.mu):
            cur_log_probs = get_per_token_logps(self.model, input_ids, attention_mask)[:, -logits_to_keep:]

            # 重要性采样比率 + 裁剪
            ratio = torch.exp(cur_log_probs - old_log_probs)
            ratio_clip = torch.clamp(ratio, 1 - self.config.epsilon, 1 + self.config.epsilon)
            advantages_expanded = advantages.unsqueeze(1)

            # 策略梯度损失
            surr1 = ratio * advantages_expanded
            surr2 = ratio_clip * advantages_expanded
            surrogate_loss = torch.min(surr1, surr2)

            # KL 惩罚
            kl = compute_kl_divergence(cur_log_probs, ref_log_probs)

            # 最终损失
            per_token_loss = surrogate_loss - self.config.beta * kl
            loss = -((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).mean()

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

        avg_reward = rewards.mean().item()
        return loss.item(), avg_reward

    def train(self):
        """完整训练循环"""
        print("=" * 50)
        print("开始 GRPO 强化学习训练")
        print("=" * 50)

        for step in range(self.config.num_steps):
            batch_samples = random.sample(self.train_data, self.config.batch_size)
            loss, avg_reward = self.train_step(batch_samples)

            if (step + 1) % 10 == 0:
                print(f"Step [{step + 1}/{self.config.num_steps}] | Loss: {loss:.4f} | Avg Reward: {avg_reward:.4f}")

        # 保存模型
        import os
        os.makedirs(self.config.output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), f"{self.config.output_dir}/grpo_model.pt")
        self.tokenizer.save_pretrained(self.config.output_dir)
        print(f"模型已保存至 {self.config.output_dir}")
