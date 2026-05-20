import flax.linen as nn
from .attention import CausalSelfAttention
from .mlp import MLP

class Block(nn.Module):
    """
    一層 Transformer Block（Pre-Norm 架構）：
    x → LN → Attention → 殘差 → LN → MLP → 殘差
    """
    n_head: int
    n_embd: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        residual = x
        x = nn.LayerNorm()(x)
        x = CausalSelfAttention(
            n_head=self.n_head, n_embd=self.n_embd,
            dropout_rate=self.dropout_rate
        )(x, deterministic=deterministic)
        x = residual + x

        residual = x
        x = nn.LayerNorm()(x)
        x = MLP(
            n_embd=self.n_embd, dropout_rate=self.dropout_rate
        )(x, deterministic=deterministic)
        x = residual + x
        return x