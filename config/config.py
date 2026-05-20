from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int   = 50257
    n_embd: int       = 128
    n_head: int       = 4
    n_layer: int      = 4
    block_size: int   = 32
    dropout_rate: float = 0.1

@dataclass
class TrainConfig:
    batch_size: int    = 16
    block_size: int    = 32
    learning_rate: float = 3e-4
    min_lr: float      = 1e-4
    max_iters: int     = 3000
    eval_interval: int = 300
    warmup_iters: int  = 100
    grad_clip: float   = 1.0
    train_split: float = 0.9
    seed: int          = 1337
    optimizer: str = "adamw"