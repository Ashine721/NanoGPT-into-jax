import jax
import jax.numpy as jnp
import tiktoken

def generate_text(state, model, prompt_text: str,
                  max_new_tokens: int, block_size: int,
                  temperature: float = 1.0):
    enc           = tiktoken.get_encoding("gpt2")
    prompt_tokens = enc.encode(prompt_text)
    idx           = jnp.array([prompt_tokens], dtype=jnp.int32)
    rng           = jax.random.PRNGKey(42)

    @jax.jit
    def get_next_token(params, idx_cond, key):
        logits            = model.apply({'params': params}, idx_cond, deterministic=True)
        next_token_logits = logits[:, -1, :] / temperature
        return jax.random.categorical(key, next_token_logits)

    print(f"--- 開始生成文本 (溫度: {temperature}) ---")
    print(prompt_text, end="")

    for _ in range(max_new_tokens):
        idx_cond        = idx[:, -block_size:]
        rng, sample_key = jax.random.split(rng)
        next_token      = get_next_token(state.params, idx_cond, sample_key)
        new_text        = enc.decode([int(next_token[0])])
        print(new_text, end="")
        idx = jnp.concatenate((idx, jnp.expand_dims(next_token, axis=-1)), axis=1)

    print("\n\n--- 生成結束 ---")