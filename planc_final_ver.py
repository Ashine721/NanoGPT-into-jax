
import os
import requests
import numpy as np
import matplotlib.pyplot as plt
import tiktoken
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state

#download data
data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
data_path = 'input.txt'

if not os.path.exists(data_path):
    print("Downloading...")
    data = requests.get(data_url).text
    with open(data_path, 'w', encoding='utf-8') as f:
        f.write(data)
else:
    with open(data_path, 'r', encoding='utf-8') as f:
        data = f.read()

print(f"Download completed! Total characters: {len(data)}")

#tokenizer
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(data)
#To save ram, transit into numpy array
token_data = np.array(tokens, dtype=np.uint16)

print(f"Total token: {len(token_data)}")

#Train/Validation Split
n = int(0.9 * len(token_data))
train_data = token_data[:n]
val_data = token_data[n:]

#Batch Loader
def get_batch(split, batch_size, block_size):
  data_source = train_data if split == 'train' else val_data

  # 隨機生成 batch_size 個起始索引(Indices)
  # 最大索引必須預留 block_size + 1 的空間，因為 Y 是 X 往右平移一格
  ix = np.random.randint(0, len(data_source) - block_size, batch_size)

  # 建立輸入矩陣 X 與目標矩陣 Y
  x = np.stack([data_source[i : i + block_size] for i in ix])
  y = np.stack([data_source[i + 1 : i + block_size + 1] for i in ix])

  # 將 Numpy 陣列轉換為 JAX 陣列，準備送入 GPU/TPU
  return jnp.array(x, dtype=jnp.int32), jnp.array(y, dtype=jnp.int32)

#test
B, T = 4, 8 # 設定批次(Batch)為 4，上下文長度(Time)為 8
X_batch, Y_batch = get_batch('train', B, T)

print(f"\n--- 測試 get_batch ---")
print(f"X 形狀(Shape): {X_batch.shape}")
print(f"Y 形狀(Shape): {Y_batch.shape}")
print(f"\n第一筆輸入 X[0]: {X_batch[0]}")
print(f"第一筆目標 Y[0]: {Y_batch[0]}")

#MLP
class MLP(nn.Module):
    n_embd: int       # 嵌入維度 (例如 768)
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        # 1. 擴展維度 (4 倍) -> 對應原版 PyTorch 的 c_fc (Fully Connected)
        x = nn.Dense(4 * self.n_embd, name='c_fc')(x)

        # 2. 激活函數 -> 對應原版的 GELU
        x = nn.gelu(x)

        # 3. 投影回原本維度 -> 對應原版的 c_proj (Projection)
        x = nn.Dense(self.n_embd, name='c_proj')(x)

        # 4. Dropout 丟棄正則化
        # JAX 的 dropout 需要知道現在是不是在「推論模式 (deterministic)」，這跟 PyTorch 的 model.eval() 概念相同
        x = nn.Dropout(self.dropout_rate, deterministic=deterministic)(x)
        return x

#attention
class CausalSelfAttention(nn.Module):
    n_head: int       # 注意力頭數 (例如 12)
    n_embd: int       # 嵌入維度 (例如 768)
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        B, T, C = x.shape
        head_size = C // self.n_head

        # 1. 一次性計算 Q, K, V 的投影 (對應原版的 c_attn)
        # 輸出維度變成 3倍的 C，然後切成三等份
        qkv = nn.Dense(3 * C, name='c_attn')(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        # 2. 形狀變換：(Batch, Time, Channel) -> (Batch, Head, Time, Head_Size)
        q = q.reshape(B, T, self.n_head, head_size).transpose((0, 2, 1, 3))
        k = k.reshape(B, T, self.n_head, head_size).transpose((0, 2, 1, 3))
        v = v.reshape(B, T, self.n_head, head_size).transpose((0, 2, 1, 3))

        # 3. 計算注意力分數 (Q * K^T / sqrt(d))
        # 這裡的 transpose 只轉置最後兩個維度 (Time 和 Head_Size)
        att = jnp.matmul(q, k.transpose((0, 1, 3, 2))) * (1.0 / jnp.sqrt(head_size))

        # 4. 因果遮罩 (Causal Masking) -> 對應原版的 register_buffer("bias", tril)
        # 建立一個 T x T 的下三角矩陣，1 代表可見，0 代表不可見
        mask = jnp.tril(jnp.ones((T, T))).reshape(1, 1, T, T)
        # 把 mask 為 0 的地方替換為 -inf，這樣經過 softmax 後機率就會變成 0
        att = jnp.where(mask == 0, -jnp.inf, att)

        # 5. Softmax 與 Attention Dropout
        att = jax.nn.softmax(att, axis=-1)
        att = nn.Dropout(self.dropout_rate, deterministic=deterministic)(att)

        # 6. 將注意力權重乘上 Value
        y = jnp.matmul(att, v) # (B, Head, T, Head_Size)

        # 7. 復原形狀：把 Head 拼回 Channel -> (Batch, Time, Channel)
        y = y.transpose((0, 2, 1, 3)).reshape(B, T, C)

        # 8. 最終輸出投影 (對應原版的 c_proj)
        y = nn.Dense(C, name='c_proj')(y)
        y = nn.Dropout(self.dropout_rate, deterministic=deterministic)(y)

        return y

#打包
class Block(nn.Module):
    n_head: int
    n_embd: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        # 1. Attention 區塊 (Pre-Norm 架構)
        residual = x
        x = nn.LayerNorm()(x) # 先做 LayerNorm
        x = CausalSelfAttention(
            n_head=self.n_head,
            n_embd=self.n_embd,
            dropout_rate=self.dropout_rate
        )(x, deterministic=deterministic)
        x = residual + x      # 殘差相加

        # 2. MLP 區塊 (Pre-Norm 架構)
        residual = x
        x = nn.LayerNorm()(x) # 先做 LayerNorm
        x = MLP(
            n_embd=self.n_embd,
            dropout_rate=self.dropout_rate
        )(x, deterministic=deterministic)
        x = residual + x      # 殘差相加

        return x

#GPT 完全體 設定GPT的參數
class GPT(nn.Module):
    vocab_size: int   # 詞彙表大小 (例如 50257)
    n_embd: int       # 嵌入維度 (例如 768)
    n_head: int       # 注意力頭數 (例如 12)
    n_layer: int      # 堆疊層數 (例如 12)
    block_size: int   # 上下文長度 (例如 1024)
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, idx, deterministic: bool = True):
        B, T = idx.shape # idx 傳進來的是整數 token，形狀為 (B, T)
        # --- 1. 輸入與嵌入層 ---
        pos = jnp.arange(0, T) # 產生位置 [0, 1, ..., T-1]

        # 將整數轉為向量 (Token Embedding & Position Embedding)
        tok_emb = nn.Embed(num_embeddings=self.vocab_size, features=self.n_embd, name='wte')(idx)
        pos_emb = nn.Embed(num_embeddings=self.block_size, features=self.n_embd, name='wpe')(pos)

        # 相加並過 Dropout
        x = tok_emb + pos_emb
        x = nn.Dropout(self.dropout_rate, deterministic=deterministic)(x)

        # --- 2. 堆疊 Transformer Blocks ---
        # 依序通過 N 層的 Block
        for i in range(self.n_layer):
            x = Block(
                n_head=self.n_head,
                n_embd=self.n_embd,
                dropout_rate=self.dropout_rate,
                name=f'h_{i}' # 給每一層命名，對應原版權重名稱
            )(x, deterministic=deterministic)

        # --- 3. 輸出層 (Final Head) ---
        x = nn.LayerNorm(name='ln_f')(x)
        # 映射回詞彙表大小，得到未歸一化的機率 (Logits)
        logits = nn.Dense(features=self.vocab_size, name='lm_head')(x)

        return logits

# 假設前面的 Block 和 GPT 類別都已經定義好了

# 1. 設定微型的超級參數 (Hyperparameters) 來做測試
vocab_size = 65     # 假設只有 65 個字元 (莎士比亞數據集的等級)
n_embd = 32         # 嵌入維度縮小，方便測試
n_head = 4
n_layer = 2         # 只疊 2 層
block_size = 8
B, T = 2, 8         # Batch Size = 2, Time = 8

# 2. 產生假資料與隨機鑰匙 (PRNG Key)
key = jax.random.PRNGKey(42)
key, init_key = jax.random.split(key)
# 假造一個整數輸入，範圍在 0 ~ vocab_size 之間
dummy_idx = jax.random.randint(key, (B, T), 0, vocab_size)

# 3. 實例化模型
model = GPT(
    vocab_size=vocab_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    block_size=block_size
)

# 4. 初始化參數 (JAX/Flax 需要用 init 來動態推斷維度並生成權重)
# 這裡 deterministic=True 代表我們只是在測試推論，關閉 Dropout
variables = model.init(init_key, dummy_idx, deterministic=True)

# 5. 執行前向傳播 (Forward Pass)
logits = model.apply(variables, dummy_idx, deterministic=True)

print("--- GPT 模型測試結果 ---")
print(f"輸入 idx 形狀: {dummy_idx.shape}  -> (Batch, Time)")
print(f"輸出 logits 形狀: {logits.shape} -> (Batch, Time, Vocab_Size)")
print("點火成功！")

# 初始化模型並建立 TrainState 容器
def create_train_state(rng, model, learning_rate):
    # 1. 隨機生成一個假的輸入，讓 Flax 可以推斷所有矩陣的維度
    dummy_x = jnp.ones((1, 1), dtype=jnp.int32)

    # 2. 初始化模型權重
    variables = model.init(rng, dummy_x, deterministic=True)
    params = variables['params']

    # 3. 選擇優化器 (Andrej Karpathy 在 nanoGPT 使用 AdamW)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate)
      )

    # 4. 建立並回傳狀態容器
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

@jax.jit
def train_step(state, x, y, dropout_key):
    """
    執行一次前向傳播與後向傳播，並更新參數。
    注意：這個函數被 @jax.jit 編譯，執行速度會被極度最佳化 (C++ / CUDA 級別)。
    """

    # 定義一個內部函數來計算 Loss (這是給 value_and_grad 求導用的)
    def loss_fn(params):
        # 1. 前向傳播 (Forward Pass)
        # 訓練時 deterministic 必須為 False，並且要傳入 dropout_key 確保每次隨機丟棄不同神經元
        logits = state.apply_fn(
            {'params': params},
            x,
            deterministic=False,
            rngs={'dropout': dropout_key}
        )

        # 2. 計算交叉熵損失 (Cross Entropy Loss)
        # optax 內建了這個函數，可以直接吃 logits 和正確答案 y
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y)

        # 回傳整個 Batch 的平均 Loss
        return loss.mean()

    # 3. 後向傳播 (Backward Pass)
    # jax.value_and_grad 會自動幫我們對 loss_fn 裡的 params 算微分！
    loss, grads = jax.value_and_grad(loss_fn)(state.params)

    # 4. 梯度下降更新參數 (Optimizer Step)
    # 這一波操作等於 PyTorch 的 optimizer.step()
    state = state.apply_gradients(grads=grads)

    return state, loss

#測試用輕量級train
# 1. 修正後的超級參數 (Hyperparameters)
vocab_size = 50257    # 關鍵修正：必須對齊 gpt2 tokenizer 的字典大小！
n_embd = 64           # 可以稍微放大一點
n_head = 4
n_layer = 4
block_size = 8
B, T = 4, 8
learning_rate = 1e-4  # 學習率：決定每次更新步伐的大小
max_iters = 1000      # 最大迭代次數
eval_interval = 100   # 每隔幾步印出一次資訊

# 2. 初始化狀態與隨機鑰匙 (Initialization)
# JAX 需要嚴格管控隨機性
rng = jax.random.PRNGKey(1337)
rng, init_rng = jax.random.split(rng)

# 建立包含模型權重與優化器的狀態容器 (TrainState)
state = create_train_state(init_rng, model, learning_rate)

print("開始訓練模型...")

# 3. 執行訓練迴圈 (Training Loop)
for step in range(max_iters):
    # 步驟 A：從數據管道抓取一個批次 (Batch) 的輸入與目標答案
    xb, yb = get_batch('train', B, T)

    # 步驟 B：為這一步的 Dropout 產生專屬的隨機鑰匙 (PRNG Key)
    rng, dropout_key = jax.random.split(rng)

    # 步驟 C：執行前向傳播與後向傳播，更新狀態容器與取得損失值 (Loss)
    state, loss = train_step(state, xb, yb, dropout_key)

    # 步驟 D：定期監控訓練狀況
    if step % eval_interval == 0 or step == max_iters - 1:
        print(f"步驟 (Step) {step:4d} | 訓練損失 (Training Loss): {loss:.4f}")

print("訓練完成！")

# 大參數train(要跑很久，T4跑20分鐘還跑不完)
# 0. 啟動 JAX 的 NaN 偵測雷達 (超級大絕招)
jax.config.update("jax_debug_nans", True) # 這行會讓 JAX 在產生 NaN 的那一瞬間立刻報錯並指出是哪一行程式碼，而不是默默算完

# 1. 確保參數完全正確
vocab_size = 50257    # 絕對不能錯，必須對齊 GPT-2 Tokenizer
n_embd = 128
n_head = 4
n_layer = 4
block_size = 32
B, T = 16, 32

learning_rate = 3e-4  # 降低學習率，防止第一步爆炸
max_iters = 3000
eval_interval = 300

# 2. 重新實例化模型
model = GPT(
    vocab_size=vocab_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    block_size=block_size
)

# 3. 定義狀態初始化器 (包含 Optax Chain)
def create_train_state(rng, model, learning_rate):
    dummy_x = jnp.ones((1, 1), dtype=jnp.int32)
    variables = model.init(rng, dummy_x, deterministic=True)
    params = variables['params']

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate)
    )

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

# 4. 初始化亂數與狀態
rng = jax.random.PRNGKey(1337)
rng, init_rng = jax.random.split(rng)
state = create_train_state(init_rng, model, learning_rate)

print("狀態重置完畢，開始全新訓練...")

# 5. 訓練迴圈
for step in range(max_iters):
    xb, yb = get_batch('train', B, T)
    rng, dropout_key = jax.random.split(rng)

    # 執行訓練步
    state, loss = train_step(state, xb, yb, dropout_key)

    if step % eval_interval == 0 or step == max_iters - 1:
        print(f"步驟 {step:4d} | 訓練損失: {loss:.4f}")


# 生成文字測試
# 1. 定義單步預測函數 (JIT 加速)
@jax.jit
def get_next_token(params, idx_cond, key, temperature=1.0):
    """
    給定一段上下文，預測並採樣出「下一個字」。
    """
    # 前向傳播 (Inference 模式，關閉 Dropout)
    logits = model.apply({'params': params}, idx_cond, deterministic=True)

    # 我們只需要最後一個時間點 (最後一個字) 的預測結果
    # logits 形狀從 (Batch, Time, Vocab) 變成 (Batch, Vocab)
    next_token_logits = logits[:, -1, :] / temperature

    # 根據機率分佈進行採樣 (抽籤)
    next_token = jax.random.categorical(key, next_token_logits)

    return next_token

# 2. 定義生成迴圈 (Generate Loop)
def generate_text(state, prompt_text, max_new_tokens, block_size, temperature=1.0):
    # 取得 gpt2 的分詞器
    enc = tiktoken.get_encoding("gpt2")

    # 把我們的 Prompt 轉換成 Token ID 陣列
    prompt_tokens = enc.encode(prompt_text)

    # 轉換成 JAX 矩陣，加上 Batch 維度 (1, T)
    idx = jnp.array([prompt_tokens], dtype=jnp.int32)

    # 準備隨機鑰匙
    rng = jax.random.PRNGKey(42)

    print(f"--- 開始生成文本 (溫度: {temperature}) ---")
    print(prompt_text, end="") # 先印出我們給的開頭

    for _ in range(max_new_tokens):
        # 確保輸入的長度不會超過模型的 block_size (我們設定是 8)
        # 如果超過了，就只取最後面 8 個字
        idx_cond = idx[:, -block_size:]

        # 產生新的亂數鑰匙來抽籤
        rng, sample_key = jax.random.split(rng)

        # 呼叫模型預測下一個字
        next_token = get_next_token(state.params, idx_cond, sample_key, temperature)

        # 把這個新字 (Token ID) 轉回人類看得懂的文字並印出來
        # next_token 是一個形狀為 (1,) 的陣列，我們取 [0] 拿數字出來
        new_text = enc.decode([int(next_token[0])])
        print(new_text, end="")

        # 把新字黏回原來的陣列，作為下一次的輸入
        idx = jnp.concatenate((idx, jnp.expand_dims(next_token, axis=-1)), axis=1)

    print("\n\n--- 🏁 生成結束 ---")

# 3. 測試生成！
# 給它一個簡單的開頭，讓它接續寫 50 個 Token
prompt = "\n"
generate_text(state, prompt, max_new_tokens=50, block_size=block_size, temperature=0.8)


# 預熱訓練
# 1. 設定訓練的總步數與預熱步數
max_iters = 1000          # 訓練總步數
warmup_iters = 100        # 前 10% 的時間用來預熱 (Warmup)
learning_rate = 1e-3      # 學習率的最高峰 (Peak LR)
min_lr = 1e-4             # 訓練結束時的最低學習率 (通常是峰值的 10%)

# 2. 建立動態學習率排程器 (Schedule)
# 在 Optax 中，排程器是一個「吃進 step 數字，吐出目前學習率」的函數
lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,              # 起始學習率 (從 0 開始)
    peak_value=learning_rate,    # 最高點
    warmup_steps=warmup_iters,   # 爬坡要走幾步
    decay_steps=max_iters,       # 衰減總步數 (通常等於總訓練步數)
    end_value=min_lr             # 谷底的學習率
)

# 3. 畫個圖來看看這完美的曲線 (數學的浪漫)
steps = np.arange(max_iters)
lrs = [lr_schedule(step) for step in steps]

plt.plot(steps, lrs)
plt.title("Cosine Decay with Warmup Learning Rate Schedule")
plt.xlabel("Training Steps")
plt.ylabel("Learning Rate")
plt.grid(True)
plt.show()

def create_train_state(rng, model, lr_schedule): # 注意這裡傳入的是 schedule
    dummy_x = jnp.ones((1, 1), dtype=jnp.int32)
    variables = model.init(rng, dummy_x, deterministic=True)
    params = variables['params']

    # 優化器升級！
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_schedule) # 直接把函數塞給它！
    )

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )