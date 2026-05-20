# NanoGPT-into-jax Learing
把NanoGPT專案透過jax library 復刻，並在過程中學習jax與GPT的相關知識

## 實驗
改變不同的優化器(optimizer)以實驗不同的優化器會造成什麼不同的結果。
我挑選以閜四個優化器：
adamw、adam、sgd、lion
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

| Step | adamw | adam | sgd | lion |
| --- | --- | --- | --- | --- |
| 0 | 11.2024 | 11.2784 | 11.3245 | 11.2218 |
| 300 | 6.4992 | 6.4261 | 10.1485 | 6.4037 |
| 600 | 5.5908 | 5.5712 | 9.2428 | 5.4536 |
| 900 | 5.4159 | 5.4316 | 9.0359 | 5.2489 |
| 1200 | 4.7947 | 4.7700 | 8.5666 | 4.4769 |
| 1500 | 5.1697 | 5.1297 | 8.7445 | 4.8309 |
| 1800 | 4.9860 | 4.9564 | 8.5661 | 4.6321 |
| 2100 | 4.4657 | 4.4630 | 8.3499 | 4.0844 |
| 2400 | 4.8719 | 4.8466 | 8.4304 | 4.5087 |
| 2700 | 4.9589 | 4.9807 | 8.6234 | 4.5551 |
| 2999 | 4.5374 | 4.5276 | 8.2307 | 4.06151 |

</div>

### loss diagram
<img width="989" height="490" alt="download" src="https://github.com/user-attachments/assets/5ee3566c-beba-43ad-ae4b-f191b6f681c8" />

### 生成文章結果

1. adamw
```
--- 開始生成文本 (溫度: 0.8) ---
To be or not to be
And, as you give the Tower of the mountingt
Aff'd, that so his one only as he love stert so so.

ISABELLA:
The dost,?

CORIOLANUS:
All more:
I'll no, good most world.

First Citizen:
Acester, sir, but I do, I am no daughter,
Even, he only the queen, sir, tell him.

Second Serving

--- 生成結束 ---
```

2. adam
```
--- 開始生成文本 (溫度: 0.8) ---
To be or not to be
And, as you, for a cause
wows delays in their g own face, I know as
And storst in my worst to the, it hanging you have in or a
Which?

SICINIUS:
Allath not, Warwick:
My good most world.

First Citizen:
Aorder, marry, but I do, as you'll live,
Even, he berear, 'twTill I pray
At true my

--- 生成結束 ---
```

3. sgd
```
--- 開始生成文本 (溫度: 0.8) ---
To be or not to be

,, of,,
:
hal,
, Included, that, the I my I, of disperse





, measures. you Contra in,
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
To be or not to be recompire,
And she comes a cause
wows the midst to thy own face of one of hell,
Or you do so return to thyone;
And he in the doth we have jally
That it, I am an agack: can be:
My good gentle years.

First Citizen:
A ones, marry, but I do not herecester,
And, my lord, only your love,
With tell me false more; and I

--- 生成結束 ---
```

## 參考資料
1. https://youtu.be/kCc8FmEb1nY?si=tQAh_1TYoZ-TbuUj
2. https://github.com/karpathy/nanoGPT
