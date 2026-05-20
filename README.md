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

| Step | adamw | adam | sgd |
| :---: | :---: | :---: | :---: |
| 0 | 11.2058 | 11.3546 | 11.3256 |
| 300 | 6.1922 | 6.1230 | 9.9574 |
| 600 | 5.7129 | 5.7094 | 9.2806 |
| 900 | 5.4815 | 5.4651 | 9.1353 |
| 1200 | 5.3371 | 5.2678 | 8.7729 |
| 1500 | 5.2865 | 5.2964 | 8.7442 |
| 1800 | 5.0539 | 5.0307 | 8.3929 |
| 2100 | 4.8249 | 4.7924 | 8.3947 |
| 2400 | 4.4971 | 4.5671 | 8.2498 |
| 2700 | 4.4143 | 4.4124 | 8.2588 |
| 2999 | 4.6703 | 4.6785 | 8.0719 |

</div>

### loss diagram
<img width="989" height="490" alt="download" src="https://github.com/user-attachments/assets/5ee3566c-beba-43ad-ae4b-f191b6f681c8" />

### 生成文章結果

1. adamw
```
--- 開始生成文本 (溫度: 0.8) ---
To be or not to be.

KING EDWARD IV:
I do not the great lament, that so most I know I pray
Or you do so return to't, it.

SecondWARDESS:
To be thy gentle Warwick, he were other,
Have not caningly: no, good most world.

First Citizen:
I will to marry, but I do, as you to me,
Even, he only the queen, too, tell me.

Second Citizen

--- 生成結束 ---
```

2. adam
```
--- 開始生成文本 (溫度: 0.8) ---
To be or not to be that
Pray by her, please it be subor,
To g own face of one a eye,
Or you do so return to't, it.

Second Murderer:
To be thy gentle Warwick, he hear on an agity.

SecondISABELLA:
The strength; I go,
I thank the king's- country, as you to live,
Even, he berear, 'twis tell him.

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





, measures. you Contra,,
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
To be or not to be recompar,
And she is a cause of the bark, and he, that
This I hadst been shown stadoes.

CLARENCE:
You have not not a mark.

LUCIO:
Pray, what is not aingly:
My lord, like a horse, and I go.

HENRY BOLINGBROKE:
Coment, sir, only thou, 'twar,
But sheUMam,

--- 生成結束 ---
```

## 參考資料
1. https://youtu.be/kCc8FmEb1nY?si=tQAh_1TYoZ-TbuUj
2. https://github.com/karpathy/nanoGPT
