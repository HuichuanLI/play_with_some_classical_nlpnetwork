import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from utils import DeepseekRMSNorm
from attention import MLA
from moe import DeepSeekMoE, compute_load_balance_loss


class DeepseekTransformerBlock(nn.Module):
    """DeepSeek Transformer 块：Pre-Norm + MLA + MoE"""

    def __init__(self, config):
        super().__init__()
        self.input_layernorm = DeepseekRMSNorm(config.hidden_size)
        self.self_attn = MLA(config)
        self.post_attention_layernorm = DeepseekRMSNorm(config.hidden_size)
        self.mlp = DeepSeekMoE(config)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False):
        # 注意力子层 + 残差
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, present_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + attn_output

        # MoE 子层 + 残差
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output, gate_probs = self.mlp(hidden_states)
        hidden_states = residual + mlp_output

        return hidden_states, present_key_value, gate_probs


class DeepseekForCausalLM(nn.Module):
    """DeepSeek 风格因果语言模型"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        # 词嵌入
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer 层
        self.layers = nn.ModuleList([
            DeepseekTransformerBlock(config)
            for _ in range(config.num_hidden_layers)
        ])

        # 最终归一化 + 语言模型头
        self.norm = DeepseekRMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=False):
        batch_size, seq_len = input_ids.shape

        # 构造因果掩码
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=input_ids.device)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device)).bool()
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) * causal_mask.unsqueeze(0).unsqueeze(0)

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        # 嵌入层
        hidden_states = self.embed_tokens(input_ids)

        # 逐层前向
        all_gate_probs = []
        presents = []
        past_key_values = past_key_values or [None] * len(self.layers)

        for layer_idx, (layer, past_kv) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present_kv, gate_probs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
                use_cache=use_cache,
            )
            all_gate_probs.append(gate_probs)
            presents.append(present_kv)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits, presents, all_gate_probs

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=128, temperature=0.7, do_sample=True):
        """自回归生成，支持KV Cache加速，对应文档KV Cache逻辑"""
        self.eval()
        batch_size = input_ids.shape[0]
        past_key_values = None

        # Prefill 阶段
        logits, past_key_values, _ = self.forward(input_ids, past_key_values=past_key_values, use_cache=True)
        next_token_logits = logits[:, -1, :] / temperature

        if do_sample:
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        generated = [next_token]

        # Decoding 阶段
        for _ in range(max_new_tokens - 1):
            logits, past_key_values, _ = self.forward(next_token, past_key_values=past_key_values, use_cache=True)
            next_token_logits = logits[:, -1, :] / temperature

            if do_sample:
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated.append(next_token)
            if next_token.item() == 2:  # EOS
                break

        return torch.cat([input_ids, torch.cat(generated, dim=1)], dim=1)
