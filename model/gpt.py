import jax.numpy as jnp
import flax.linen as nn
from .block import Block
from config.config import GPTConfig

class GPT(nn.Module):
    """
    完整 GPT 模型：
    Token Embed + Position Embed → Dropout → N×Block → LayerNorm → LM Head
    """
    cfg: GPTConfig

    @nn.compact
    def __call__(self, idx, deterministic: bool = True):
        B, T = idx.shape
        pos  = jnp.arange(0, T)

        tok_emb = nn.Embed(num_embeddings=self.cfg.vocab_size, features=self.cfg.n_embd, name='wte')(idx)
        pos_emb = nn.Embed(num_embeddings=self.cfg.block_size, features=self.cfg.n_embd, name='wpe')(pos)
        x = tok_emb + pos_emb
        x = nn.Dropout(self.cfg.dropout_rate, deterministic=deterministic)(x)

        for i in range(self.cfg.n_layer):
            x = Block(
                n_head=self.cfg.n_head, n_embd=self.cfg.n_embd,
                dropout_rate=self.cfg.dropout_rate, name=f'h_{i}'
            )(x, deterministic=deterministic)

        x = nn.LayerNorm(name='ln_f')(x)
        logits = nn.Dense(features=self.cfg.vocab_size, name='lm_head')(x)
        return logits