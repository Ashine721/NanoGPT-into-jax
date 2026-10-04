# NanoGPT-into-jax Learing
本專案旨在透過 JAX 函式庫從零復刻 NanoGPT，藉由純函數 (Pure Functions) 的設計思維，深入探討 Transformer 與 GPT 底層架構。
在完成模型建置後，專案進一步探討了不同最佳化演算法（Optimizer）在相同架構與訓練步數下，對模型收斂速度與文本生成品質的差異。

## 實驗
為了觀察不同優化器的收斂動態，本實驗在相同的模型參數、資料集與超參數設定下，測試了五種經典與前沿的優化器：
adamw、adam、sgd、lion、muon
### 參數設置

```
generate_text(
    best_state, model,
    prompt_text    = "To be or not to be",
    max_new_tokens = 100,
    block_size     = gpt_cfg.block_size,
    temperature    = 0.8,
    #seed           = 42,
)
```
### Table of loss of four optimizer per 300 epcohes in 3000 steps 

<div align="center">

各 Optimizer 結果對比

| Step | adamw | adam | sgd | lion | muon |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 11.3024 | 11.4517 | 11.3661 | 11.2937 | 11.4998 |
| **300** | 5.9285 | 5.9389 | 9.9231 | 5.8874 | 8.6707 |
| **600** | 5.6014 | 5.5698 | 9.1242 | 5.3338 | 7.7235 |
| **900** | 5.6479 | 5.6537 | 9.0890 | 5.4682 | 7.3930 |
| **1200** | 5.2728 | 5.2870 | 8.9199 | 4.8958 | 6.9614 |
| **1500** | 4.6504 | 4.6677 | 8.4884 | 4.2444 | 6.0381 |
| **1800** | 4.8439 | 4.8795 | 8.6336 | 4.5223 | 6.2111 |
| **2100** | 4.5559 | 4.5583 | 8.3637 | 4.2581 | 5.8634 |
| **2400** | 4.5812 | 4.5641 | 8.1950 | 4.1139 | 5.8345 |
| **2700** | 4.6950 | 4.7069 | 8.2809 | 4.2566 | 6.0598 |
| **2999** | 4.7967 | 4.8190 | 8.3051 | 4.3358 | 5.9310 |

---
</div>

### loss diagram
<img width="989" height="490" alt="download" src="https://github.com/user-attachments/assets/e2904cec-489e-400c-a37c-a15e9da1c284" />

### 生成文章結果

1. adamw
```
--- 開始生成文本 (溫度: 0.8) ---
To be or not to be
And, as a truth, please thou shalt say to the great lament, that
This I know I pray my stumberine sound.

ISABELLA:
The dost,?

CORIOLANUS:
All:
Myingly:
My good gentle world.

First Citizen:
I will to marry, but I do, I am no daughter,
Even, great barren.
Welcome; but tell me, though sir, my

--- 生成結束 ---
```

2. adam
```
--- 開始生成文本 (溫度: 0.8) ---
To be or not to be
And, that you, for a cause
wows the great lament'd that so his thousand
In the love storst of my device to his,
To mock aomed or a mark.

GLOUCESTER:
Aere more not, Warwick:
My good most world.

First Citizen:
I will for the king's- country, as you'll live,
Even, he bere brother, 'twis tell him.

Second Citizen

--- 生成結束 ---
```

3. sgd
```
--- 開始生成文本 (溫度: 0.8) ---
To be or not to be

,, needles,,


hal,
, Included, that

 I
 I, of disperse





, it. you Contra,,
:
? Ary,
 reflective it,:
:


 dialogue scrambling:
 embed:

. Channel:;99:


 to
 sane

:
?


░, resear

:
,,;


 sites

;


--- 生成結束 ---
```

4. lion 
```
--- 開始生成文本 (溫度: 0.8) ---
To be or not to be broke:
But you, how please it be subitor, sir.

First Senator:
You pray it disperse you, let me be hanged,
To mock aomed or prank'd by health.

KING RICHARD III:
They shall you be: no, good was a little,
With't: every king will to marry, but I do
Stand you no loss of highness,
Which I think 'twere tell it had given him;

--- 生成結束 ---
```

5. muon
```
--- 開始生成文本 (溫度: 0.8) ---
To be or not to be.
PICK:
Not will thou shalthal, and,
I that so his I know as, my cultivated you.


Clown:
LADomed, sir, we?


KING RICHARD III:


My Lord:
My lord was yourIA, I will's:
A will to the king's-,
Now as no of my high,--, and Guerrero, 'tw,
In sites refres, true my

--- 生成結束 ---
```
## 未來展望
由於本次實驗的初衷為建立基準測試，並未針對個別優化器進行超參數調整。未來，我期望能吸取這次實作的經驗，針對不同最佳化器的數學特性進行系統性的超參數調優實驗。深入探討這些演算法背後的理論機制，藉此在未來的專案中，更精準地提升大型語言模型訓練的穩定性與收斂效能。

## 參考資料
1. https://youtu.be/kCc8FmEb1nY?si=tQAh_1TYoZ-TbuUj
2. https://github.com/karpathy/nanoGPT
