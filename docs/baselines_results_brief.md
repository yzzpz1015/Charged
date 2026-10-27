# Baseline 补充批次速览（#7–#21）

> 完整协议、超参与表注见 [`baselines_protocol.md`](baselines_protocol.md)。五城（LOA/JHB/MEL/SPO/SZH）、`seq_l=72 / pre_len=72`、同一测试窗与 clean mask；主指标 crpsΣ（H24，越低越好）。H48/H72、五城汇总与 WIS/校准见 §11.4；点预测见 §11.3。本文件只是早期批次的速览，数值以 protocol 为准。

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
| timegrad | 0.2545 | 0.2853±0.0011 | 0.3980 | 0.5066±0.0103 | 0.1028 |
| tsdiff | 0.2701 | 0.2942 | 0.3516 | 0.5415 | 0.0889* |
| diffstg | 0.3931‡ | 0.3201 | 0.4251±0.0060 | 0.6117±0.0712 | 0.2363* |

### 零样本 TSFM
| 方法 | LOA | JHB | MEL | SPO | SZH |
|---|---:|---:|---:|---:|---:|
| chronos_bolt | 0.2643 | 0.2977 | 0.3600 | 0.5189 | 0.0868 |
| moirai | 0.2722 | 0.3043 | 0.3797 | 0.5329 | 0.1102 |

`*` = 只评固定子集（CSDI 全城 256 窗、SZH 128 窗；TSDiff/DiffSTG 的 SZH 256 窗），`nsample=50`（CSDI）或 100（其余扩散）。`‡` = DiffSTG-LOA 对 `sample_chunk` 敏感：本表用主评测 `sample_chunk=16`（0.3931），WIS 重评用 `chunk=8` 得 0.9913，仅 RNG 流不同；该格不与 §11.4 的 WIS 并列比较。

## 五条结论

1. **主干 + 分位头整体最强**：MQ-CNN / PatchTST / iTransformer 的五城 H24 均值 0.2842 / 0.2848 / 0.2891，差 ≤0.005，没有明显赢家；TFT 五城均值 0.2987，正值格最准（WIS_nz 全场最低），但 SPO 零格贡献过高（0.2080 vs iTransformer 0.1299）把边际 crpsΣ 拉到 0.5247。
2. **图结构几乎没用**：STGCN 带图 vs 去图（#8 vs #9）五城同向，但只有 1%~2%；Graph WaveNet 接近 iTransformer，无额外收益。
3. **扩散类整体仍打不过分位头，但城间差异大**：TimeGrad 在 LOA/JHB（0.2545/0.2853）已逼近最好的分位头，TSDiff 在 LOA/JHB/MEL（0.2701/0.2942/0.3516）稳定在中游；两者都在 SPO 掉队（0.5066/0.5415），DiffSTG 尤甚。DiffSTG v3（逐 block 历史条件 FiLM）在 JHB 为 0.3201、MEL 0.4251±0.0060，但 SPO（0.6117±0.0712）与 LOA 全量（0.3931，且对采样分块不稳）明显欠离散；官方 CSDI 依旧结构性欠离散（PICP90 0.48~0.62），说明 #3 的失败不是门控改造单独造成的。
4. **逐站自回归模型吃亏**：DeepAR-ISQF 在 SZH 只有 0.1909（分位头约 0.08），站多、每站训练数据少时明显掉队。
5. **零样本可用但非最强**：Chronos-Bolt 在 SZH（0.0868）接近有监督分位头，Moirai 整体中游。补充批次还接入了 UniTS-FT（#20，五城 H24 crpsΣ 均值 0.3018）与 OpenCity-Plus（#21，0.2920，3 seeds），均未超过第一梯队；完整数值与子集/口径说明见 §11.4。

## 口径提醒

* 三个扩散 baseline 是**按论文结构重写的轻量实现**，不是官方代码：TimeGrad 为非自回归 72 步联合扩散；TSDiff 用硬历史投影替代 soft self-guidance；DiffSTG 用轻量时空 block 替代 UGnet。
* DiffSTG 历史条件经过 v1（时间均值）→ v2（GRU 末状态，仅输入层）→ v3（逐 block FiLM）三轮；主表只用 v3，v1/v2 已归档、不进表。SPO 上 v2 与 v3 都塌缩，属该类方法在本城的真实负结果。
* 样本数：扩散 100，官方 CSDI 50，分位方法 200（分位→分层样本）。
* MQ-CNN 无 static station embedding；TFT 为逐站单变量、无静态协变量。
* DeepAR 官方默认 StudentT 退化（SPO 0.867），已用 ISQF 替代，StudentT 仅留作参考、不进主表。
* 逐方法 seed 覆盖以 `result/baselines/metrics/{method}/{city}/H*_seed*.json` 为准；主表按实际可得 seed 报 mean±std，单 seed 直接报 seed 0。

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

权重：HuggingFace [`yzzpz/Charged-checkpoints`](https://huggingface.co/yzzpz/Charged-checkpoints) 的 `baselines/ckpt/{method}/{city}/seed{s}.pth`。**160 个 baseline checkpoint 已全部同步**（16 组方法 × 各城各 seed，含 TimeGrad-SZH、DiffSTG-LOA/SZH、OpenCity seed0–2），本地与远端逐文件字节数一致；零样本 Chronos-Bolt / Moirai 无本地训练权重，走官方 HF 权重。

指标 JSON 已同步到同一 HF 仓库的 `baselines/` 目录：

- `metrics/`：510 个（含 OpenCity seed0–2、UniTS 五城）；
- `metrics_point/`：496 个（含 TimeGrad/CSDI 的 SZH、DiffSTG 的 LOA）；
- `metrics_wis/`：495 个（WIS / WIS_nz / 零格与正格贡献）。

`unist_pool_smoke/` 冒烟产物已从远端删除。本地三套目录与远端逐文件一致，可用 `summarize.py` / `summarize_point.py` 重生成表格。
