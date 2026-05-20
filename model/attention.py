import jax
import jax.numpy as jnp
import flax.linen as nn

class CausalSelfAttention(nn.Module):
    """
    多頭因果自注意力機制（含因果遮罩）。
    輸入/輸出形狀：(B, T, C)
    """
    n_head: int
    n_embd: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        B, T, C = x.shape
        head_size = C // self.n_head

        qkv = nn.Dense(3 * C, name='c_attn')(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = q.reshape(B, T, self.n_head, head_size).transpose((0, 2, 1, 3))
        k = k.reshape(B, T, self.n_head, head_size).transpose((0, 2, 1, 3))
        v = v.reshape(B, T, self.n_head, head_size).transpose((0, 2, 1, 3))

        att = jnp.matmul(q, k.transpose((0, 1, 3, 2))) * (1.0 / jnp.sqrt(head_size))

        mask = jnp.tril(jnp.ones((T, T))).reshape(1, 1, T, T)
        att  = jnp.where(mask == 0, -jnp.inf, att)
        att  = jax.nn.softmax(att, axis=-1)
        att  = nn.Dropout(self.dropout_rate, deterministic=deterministic)(att)

        y = jnp.matmul(att, v)
        y = y.transpose((0, 2, 1, 3)).reshape(B, T, C)
        y = nn.Dense(C, name='c_proj')(y)
        y = nn.Dropout(self.dropout_rate, deterministic=deterministic)(y)
        return y