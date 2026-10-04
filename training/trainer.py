import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from config.config import TrainConfig

def _build_optimizer(cfg: TrainConfig):
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.learning_rate,
        warmup_steps=cfg.warmup_iters,
        decay_steps=cfg.max_iters,
        end_value=cfg.min_lr,
    )
    match cfg.optimizer:
        case 'adamw':
            opt = optax.adamw(learning_rate=lr_schedule)
        case 'adam':
            opt = optax.adam(learning_rate=lr_schedule)
        case 'sgd':
            opt = optax.sgd(learning_rate=lr_schedule, momentum=0.9)
        case 'lion':
            opt = optax.lion(learning_rate=lr_schedule)
        case 'muon':
            opt = optax.contrib.muon(
                learning_rate=lr_schedule,
                adam_learning_rate=lr_schedule,
                weight_decay=0.0,
                ns_steps=5,
                nesterov=True,
            )
        case _:
            raise ValueError(f'Unknown optimizer: {cfg.optimizer}')
    return optax.chain(optax.clip_by_global_norm(cfg.grad_clip), opt)

def create_train_state(rng, model, cfg: TrainConfig):
    dummy_x   = jnp.ones((1, 1), dtype=jnp.int32)
    variables = model.init(rng, dummy_x, deterministic=True)
    params    = variables['params']
    tx        = _build_optimizer(cfg)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx,
    )

# ↓ 移除 @jax.jit，改由 lax.scan 內部處理
def train_step(state, x, y, dropout_key):
    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params}, x,
            deterministic=False, rngs={'dropout': dropout_key},
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=y
        )
        return loss.mean()

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state       = state.apply_gradients(grads=grads)
    return state, loss
