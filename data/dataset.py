import os
import requests
import numpy as np
import tiktoken
import jax.numpy as jnp

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(DATA_DIR, 'input.txt')
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def download_data(url: str = DATA_URL, path: str = DATA_PATH) -> str:
    if os.path.exists(path):
        print(f"使用本地檔案：{path}")
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    
    # 找不到才下載
    print("Downloading...")
    data = requests.get(url).text
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(data)
    print(f"Download completed! Total characters: {len(data)}")
    return data

def tokenize(data: str) -> np.ndarray:
    """用 GPT-2 BPE 將字串 tokenize，回傳 uint16 numpy 陣列（節省記憶體）。"""
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(data)
    token_data = np.array(tokens, dtype=np.uint16)
    print(f"Total tokens: {len(token_data)}")
    return token_data

def split_data(token_data: np.ndarray, train_split: float = 0.9):
    """將 token 陣列切成 train / val 兩份。"""
    n = int(train_split * len(token_data))
    return token_data[:n], token_data[n:]