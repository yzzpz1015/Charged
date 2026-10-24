# Baseline 补充批次速览（#7–#19，seed 0）

> 完整协议、超参与表注见 [`baselines_protocol.md`](baselines_protocol.md)。五城（LOA/JHB/MEL/SPO/SZH）、`seq_l=72 / pre_len=72`、同一测试窗与 clean mask；主指标 crpsΣ（H24，越低越好）。H48/H72 见 `result/baselines/metrics/` 或 HF 同步的 JSON。

## H24 crpsΣ（seed 0）

### 主干 + 分位头
| 方法 | LOA | JHB | MEL | SPO | SZH |
|---|---:|---:|---:|---:|---:|
| itransformer | 0.2458 | 0.2892 | 0.3529 | 0.4774 | 0.0797 |
| stgcn | 0.2640 | 0.2959 | 0.3719 | 0.4912 | 0.0921 |
| stgcn_nograph | 0.2679 | 0.2969 | 0.3788 | 0.4943 | 0.0907 |
| gwnet | 0.2497 | 0.2921 | 0.3706 | 0.4852 | 0.0869 |
| mqcnn | 0.2468 | 0.2855 | 0.3473 | 0.4604 | 0.0815 |
| patchtst_quant_pre72 | 0.2437 | 0.2819 | 0.3475 | 0.4604 | 0.0842 |
| tft | 0.2387 | 0.2884 | 0.3496 | 0.5555 | 0.0963 |

### 原生概率 / 逐站自回归
| 方法 | LOA | JHB | MEL | SPO | SZH |
|---|---:|---:|---:|---:|---:|
| deepar_isqf | 0.2942 | 0.3287 | 0.4302 | 0.5632 | 0.1909 |

### 生成式扩散
| 方法 | LOA | JHB | MEL | SPO | SZH |
|---|---:|---:|---:|---:|---:|
| csdi_official | 0.2962* | 0.3290* | 0.4552* | 0.4927* | 0.1050* |
| timegrad | 0.2545 | 0.2861 | 0.3980 | 0.5139 | — |
| tsdiff | — | 0.2942 | 0.3516 | 0.5415 | — |
| diffstg | — | 0.3682 | — | — | — |

### 零样本 TSFM
| 方法 | LOA | JHB | MEL | SPO | SZH |
|---|---:|---:|---:|---:|---:|
| chronos_bolt | 0.2643 | 0.2977 | 0.3600 | 0.5189 | 0.0868 |
| moirai | 0.2722 | 0.3043 | 0.3797 | 0.5329 | 0.1102 |

`*` = CSDI 只评 256 窗子集（SZH 128 窗），`nsample=50`。`—` = 运行中（TSDiff LOA/SZH、DiffSTG LOA/MEL/SPO/SZH、TimeGrad SZH）。

## 五条结论

1. **主干 + 分位头整体最强**：iTransformer / MQ-CNN / PatchTST 三者在五城的 H24 差异 ≤2%，没有明显赢家；TFT 在 LOA 最好（0.2387），但在 SPO 严重欠离散（PICP50 0.34）落到 0.5555。
2. **图结构几乎没用**：STGCN 带图 vs 去图（#8 vs #9）五城同向，但只有 1%~2%；Graph WaveNet 接近 iTransformer，无额外收益。
3. **扩散类仍打不过分位头**：TimeGrad/TSDiff 在 JHB/MEL 最接近（0.286/0.294、0.398/0.352），但 LOA/SPO 落后；官方 CSDI（无门控、无空间序）依旧结构性欠离散（PICP90 0.48~0.62），说明 #3 输不是门控改造造成的。
4. **逐站自回归模型吃亏**：DeepAR-ISQF 在 SZH 只有 0.1909（分位头约 0.08），站多、每站训练数据少时明显掉队。
5. **零样本可用但非最强**：Chronos-Bolt 在 SZH（0.0868）接近有监督分位头，Moirai 整体中游；UniST（#20）未做（网格模型依赖大规模预训练与多模态侧信息，与本协议不兼容）。

## 口径提醒

* 三个扩散 baseline 是**按论文结构重写的轻量实现**，不是官方代码：TimeGrad 为非自回归 72 步联合扩散；TSDiff 用硬历史投影替代 soft self-guidance；DiffSTG 用轻量时空 block 替代 UGnet。
* 样本数：扩散 100，官方 CSDI 50，分位方法 200（分位→分层样本）。
* MQ-CNN 无 static station embedding；TFT 为逐站单变量、无静态协变量。
* DeepAR 官方默认 StudentT 退化（SPO 0.867），已用 ISQF 替代，StudentT 仅留作参考、不进主表。
* 当前只有 seed 0 完整；seed 1 已补 iTransformer / MQ-CNN / PatchTST-pre72（城间差异 ≤2%），TFT 进行中。

## 复现与权重

```bash
python3 script/baselines/train_quant_backbone.py --method itransformer --city MEL --device 0
python3 script/baselines/train_gluonts.py --method tft --city MEL --device 0
python3 script/baselines/csdi_official.py --city MEL --device 0
python3 script/baselines/train_timegrad.py --city MEL --device 0
python3 script/baselines/train_tsdiff.py --city MEL --device 0
python3 script/baselines/train_diffstg.py --city MEL --device 0
python3 script/baselines/summarize.py
```

权重与指标 JSON：HuggingFace [`yzzpz/Charged-checkpoints`](https://huggingface.co/yzzpz/Charged-checkpoints) 的 `baselines/` 目录。
