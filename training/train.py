import jax
from config.config import GPTConfig, TrainConfig
from data.dataset import download_data, tokenize, split_data
from data.dataloader import get_batch
from model.gpt import GPT
from training.trainer import create_train_state, train_step

jax.config.update("jax_debug_nans", True)

def main():
    gpt_cfg   = GPTConfig()
    train_cfg = TrainConfig()

    raw_data             = download_data()
    tokens               = tokenize(raw_data)
    train_data, val_data = split_data(tokens, train_cfg.train_split)

    model         = GPT(cfg=gpt_cfg)
    rng           = jax.random.PRNGKey(train_cfg.seed)
    rng, init_rng = jax.random.split(rng)
    state         = create_train_state(init_rng, model, train_cfg)

    print("開始訓練...")
    for step in range(train_cfg.max_iters):
        xb, yb           = get_batch(train_data, train_cfg.batch_size, train_cfg.block_size)
        rng, dropout_key = jax.random.split(rng)
        state, loss      = train_step(state, xb, yb, dropout_key)

        if step % train_cfg.eval_interval == 0 or step == train_cfg.max_iters - 1:
            print(f"Step {step:4d} | Loss: {loss:.4f}")

    print("訓練完成！")
    return state, model

if __name__ == "__main__":
    main()