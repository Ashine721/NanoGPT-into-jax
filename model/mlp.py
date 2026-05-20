import flax.linen as nn

class MLP(nn.Module):
    """
    Transformer 的前饋子層：
    Linear(C → 4C) → GELU → Linear(4C → C) → Dropout
    """
    n_embd: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        x = nn.Dense(4 * self.n_embd, name='c_fc')(x)   # 擴展維度 (4x)
        x = nn.gelu(x)                                   # GELU 激活
        x = nn.Dense(self.n_embd,     name='c_proj')(x)  # 投影回原維度
        x = nn.Dropout(self.dropout_rate, deterministic=deterministic)(x)
        return x