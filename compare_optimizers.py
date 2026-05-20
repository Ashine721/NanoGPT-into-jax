import jax
import jax.numpy as jnp
from jax import lax
import matplotlib.pyplot as plt

jax.config.update("jax_debug_nans", False)

# ── 設定 ──────────────────────────────────────────────
gpt_cfg   = GPTConfig()
train_cfg = TrainConfig()

OPTIMIZERS = ["adamw", "adam", "sgd", "lion"]
STEPS      = train_cfg.max_iters

rng = jax.random.PRNGKey(train_cfg.seed)

raw_data             = download_data()
tokens               = tokenize(raw_data)
train_data, val_data = split_data(tokens, train_cfg.train_split)

model = GPT(cfg=gpt_cfg)

# ── 為每個 optimizer 建立獨立 state ───────────────────
states = {}
for opt_name in OPTIMIZERS:
    cfg         = TrainConfig()
    cfg.optimizer = opt_name
    rng, init_rng = jax.random.split(rng)
    states[opt_name] = create_train_state(init_rng, model, cfg)

# ── 預先生成所有 step 的 batch ────────────────────────
all_x    = []
all_y    = []
all_keys = []

for _ in range(STEPS):
    xb, yb           = get_batch(train_data, train_cfg.batch_size, train_cfg.block_size)
    rng, dropout_key = jax.random.split(rng)
    all_x.append(xb)
    all_y.append(yb)
    all_keys.append(dropout_key)

all_x    = jnp.stack(all_x)    # (STEPS, B, T)
all_y    = jnp.stack(all_y)    # (STEPS, B, T)
all_keys = jnp.stack(all_keys) # (STEPS, 2)

# ── 用 lax.scan 訓練單一 optimizer ───────────────────
@jax.jit
def run_training(state, all_x, all_y, all_keys):
    def one_step(state, inputs):
        x, y, dropout_key = inputs
        state, loss = train_step(state, x, y, dropout_key)
        return state, loss

    state, losses = lax.scan(one_step, state, (all_x, all_y, all_keys))
    return state, losses

# ── 依序跑每個 optimizer（共用同一份 batch）─────────────
all_losses = {}

for opt_name in OPTIMIZERS:
    print(f"\n⏳ 訓練中：{opt_name} ...")
    final_state, losses = run_training(
        states[opt_name], all_x, all_y, all_keys
    )
    jax.block_until_ready(losses)
    all_losses[opt_name] = losses
    states[opt_name]     = final_state
    print(f"✅ {opt_name} 完成！最終 Loss: {losses[-1]:.4f}")

# ── 印出對比結果 ───────────────────────────────────────
print("\n📊 各 Optimizer 結果對比：")
print(f"{'Step':>6} | " + " | ".join(f"{name:>8}" for name in OPTIMIZERS))
print("-" * (10 + 13 * len(OPTIMIZERS)))
for step in list(range(0, STEPS, train_cfg.eval_interval)) + [STEPS - 1]:
    row = f"{step:>6} | "
    row += " | ".join(f"{all_losses[name][step]:>8.4f}" for name in OPTIMIZERS)
    print(row)

# ── 畫比較圖 ──────────────────────────────────────────
plt.figure(figsize=(10, 5))
for opt_name in OPTIMIZERS:
    plt.plot(all_losses[opt_name], label=opt_name, linewidth=1)

plt.title("Optimizer Comparison - Training Loss (lax.scan)")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{PROJECT_DIR}/optimizer_comparison.png", dpi=150)
plt.show()
print(f"✅ 圖已儲存至 {PROJECT_DIR}/optimizer_comparison.png")
這個檔案該怎麼命名
