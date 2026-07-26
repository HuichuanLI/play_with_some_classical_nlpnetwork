import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from utils import DeepseekRMSNorm, DeepseekRotaryEmbedding, apply_rotary_pos_emb
import math


class MLA(nn.Module):
    """
    Multi-Head Latent Attention (MLA)
    DeepSeek 核心注意力架构，低秩压缩KV，大幅降低显存占用
    完整复现文档4 MLA 实现
    """

    def __init__(self, config):
        super().__init__()
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        # MLA 低秩压缩配置
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        # Query 低秩投影 (Down + Up)
        self.q_down_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=config.attention_bias)
        self.q_down_layernorm = DeepseekRMSNorm(self.q_lora_rank)
        self.q_up_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)

        # KV 低秩投影 (Down + Up)，MQA 风格所有头共享压缩后的KV
        self.kv_down_proj = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias
        )
        self.kv_down_layernorm = DeepseekRMSNorm(self.kv_lora_rank)
        self.kv_up_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False
        )

        # 输出投影
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=config.attention_bias)

        # RoPE
        self.rotary_emb = DeepseekRotaryEmbedding(
            self.qk_rope_head_dim,
            self.max_position_embeddings,
            self.rope_theta,
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # ========== 1. Query 分支处理 ==========
        q = self.q_up_proj(self.q_down_layernorm(self.q_down_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # ========== 2. KV 分支低秩压缩处理 ==========
        compressed_kv = self.kv_down_proj(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        # K 的 RoPE 部分所有头共享（MQA思想）
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        # 上投影展开为多头 K_nope 和 V
        kv = self.kv_up_proj(self.kv_down_layernorm(compressed_kv))
        kv = kv.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # ========== 3. 应用 RoPE ==========
        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        # 拼接完整 Q、K
        query_states = torch.empty(bsz, self.num_heads, q_len, self.q_head_dim, device=k_pe.device)
        query_states[:, :, :, :self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim:] = q_pe

        key_states = torch.empty(bsz, self.num_heads, q_len, self.q_head_dim, device=k_pe.device)
        key_states[:, :, :, :self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim:] = k_pe

        # ========== 4. KV Cache 处理 ==========
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=-2)
            value_states = torch.cat([past_key_value[1], value_states], dim=-2)

        past_key_value = (key_states, value_states) if use_cache else None

        # ========== 5. 注意力计算 ==========
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.q_head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value
