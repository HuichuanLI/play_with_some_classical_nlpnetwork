from dataclasses import dataclass


@dataclass
class ModelConfig:
    # 基础模型配置
    vocab_size: int = 32000  # 词汇表大小
    hidden_size: int = 2048  # 隐藏层维度
    num_hidden_layers: int = 12  # Transformer 层数
    num_attention_heads: int = 16  # 注意力头数
    max_position_embeddings: int = 4096
    rope_theta: float = 128000.0
    attention_dropout: float = 0.0
    attention_bias: bool = False

    # MLA 注意力专属配置（核心低秩压缩）
    q_lora_rank: int = 1536  # Query 低秩压缩维度
    kv_lora_rank: int = 512  # KV 低秩压缩维度（DeepSeek 核心优化）
    qk_rope_head_dim: int = 64  # RoPE 部分头维度
    qk_nope_head_dim: int = 128  # 内容编码部分头维度
    v_head_dim: int = 128  # Value 头维度

    # MoE 专属配置（共享专家 + 路由专家）
    num_routed_experts: int = 8  # 路由专家数量
    num_shared_experts: int = 2  # 共享专家数量（始终激活）
    top_k: int = 2  # 每次激活 Top-K 路由专家
    moe_hidden_dim: int = 5632  # 专家内部隐藏层维度
    moe_dropout: float = 0.1


@dataclass
class GRPOConfig:
    # GRPO 算法超参
    num_generations: int = 4  # 每个问题采样生成数量
    max_completion_length: int = 256  # 生成最大长度
    beta: float = 0.04  # KL 散度惩罚系数
    epsilon: float = 0.1  # PPO 裁剪系数
    learning_rate: float = 5e-6  # 学习率
    mu: int = 1  # 每个 batch 策略更新次数
    max_grad_norm: float = 0.1  # 梯度裁剪阈值

    # 训练超参
    batch_size: int = 2
    num_steps: int = 100
    output_dir: str = "./output"
