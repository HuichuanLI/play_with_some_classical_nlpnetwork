import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """单个专家网络，对应Transformer FFN结构"""

    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.net(x)


class TopKRouter(nn.Module):
    """Top-K 门控路由，输出权重与索引"""

    def __init__(self, input_dim, num_experts, top_k=2):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts, bias=False)
        self.top_k = top_k

    def forward(self, x):
        logits = self.gate(x)
        gate_probs = F.softmax(logits, dim=-1)
        top_k_logits, indices = logits.topk(self.top_k, dim=-1)
        routing_weights = F.softmax(top_k_logits, dim=-1)
        return routing_weights, indices, gate_probs


class DeepSeekMoE(nn.Module):
    """
    DeepSeek 风格 MoE：共享专家(始终激活) + 路由专家(Top-K选中)
    对应文档1 DeepSeek MoE 架构创新点1
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_routed_experts = config.num_routed_experts
        self.num_shared_experts = config.num_shared_experts
        self.top_k = config.top_k

        # 共享专家：所有token都经过，处理通用特征
        self.shared_experts = nn.ModuleList([
            Expert(config.hidden_size, config.moe_hidden_dim, config.moe_dropout)
            for _ in range(self.num_shared_experts)
        ])

        # 路由专家：门控选择Top-K个处理，处理差异化特征
        self.router = TopKRouter(config.hidden_size, self.num_routed_experts, self.top_k)
        self.routed_experts = nn.ModuleList([
            Expert(config.hidden_size, config.moe_hidden_dim, config.moe_dropout)
            for _ in range(self.num_routed_experts)
        ])

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.reshape(-1, hidden_dim)

        # ========== 1. 共享专家计算（全量激活）==========
        shared_out = torch.zeros_like(x_flat)
        for expert in self.shared_experts:
            shared_out = shared_out + expert(x_flat)
        shared_out = shared_out / self.num_shared_experts

        # ========== 2. 路由专家稀疏计算 ==========
        routing_weights, selected_indices, gate_probs = self.router(x_flat)
        routed_out = torch.zeros_like(x_flat, device=x.device)

        for expert_idx in range(self.num_routed_experts):
            batch_idx, pos_idx = torch.where(selected_indices == expert_idx)
            if batch_idx.numel() == 0:
                continue

            expert_input = x_flat[batch_idx]
            expert_output = self.routed_experts[expert_idx](expert_input)
            weights = routing_weights[batch_idx, pos_idx].unsqueeze(-1)
            routed_out.index_add_(0, batch_idx, expert_output * weights)

        # ========== 3. 合并输出 ==========
        output = shared_out + routed_out
        output = output.reshape(batch_size, seq_len, hidden_dim)
        return output, gate_probs


def compute_load_balance_loss(gate_probs, aux_weight=0.01):
    """MoE 负载均衡损失，防止专家塌缩，对应文档GRPO辅助损失思想"""
    expert_avg_probs = gate_probs.mean(dim=0)
    balance_loss = torch.sum(expert_avg_probs ** 2) * gate_probs.shape[-1]
    return aux_weight * balance_loss
