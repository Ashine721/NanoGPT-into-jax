import numpy as np
import jax.numpy as jnp

def get_batch(data_source: np.ndarray, batch_size: int, block_size: int):
    """
    從資料集隨機採樣一個 batch。
    X：輸入序列 (B, T)
    Y：目標序列 (B, T)，即 X 右移一格
    """
    ix = np.random.randint(0, len(data_source) - block_size, batch_size)
    x  = np.stack([data_source[i     : i + block_size    ] for i in ix])
    y  = np.stack([data_source[i + 1 : i + block_size + 1] for i in ix])
    return jnp.array(x, dtype=jnp.int32), jnp.array(y, dtype=jnp.int32)