# Charged — 多站点 EV 充电需求概率预测

小时级、站级电动汽车充电量（`volume`，kWh/h）的概率预测。当前重心不再是“再加一路外部模态”，而是：

> **在大量真实零值 + 重尾 + 站间弱相关的数据上，概率分布应该怎么给？**

主比较对象是三类做法：① 点预测 + 残差池、③ 逐站分位头、④/⑤ 整块扩散；主干为冻结的 `Kairos_23m` TSFM，PatchTST 作为主干消融。

**当前结论：逐站分位头（③）稳定小胜；两种扩散（④ 整块 CSDI、⑤ 吸收态 zidiff）都输在结构而不是调参；零不是噪声，而是需要单独表示的“状态”。**

零值形态、宏观/单站曲线、各方法预测分布的直观对比见：
[`docs/zero_inflation_report.html`](docs/zero_inflation_report.html)（自包含 HTML，下载后用浏览器打开）。
完整推导与全部负结果见 [`docs/概率预测阶段总结_2026-10-20.md`](docs/概率预测阶段总结_2026-10-20.md)。

---

## 1. 主结果（H=24，clean 口径，crpsΣ 越低越好）

crpsΣ = 先按站算 CRPS，再对站取平均、按训练期站均值归一化；它是当前主判指标。

| 城市 | ① 点+残差池 | ③ 分位头 | ④ 整块扩散 | ⑤ 吸收态扩散 | 参照：PatchTST+池 |
|---|---:|---:|---:|---:|---:|
| LOA | 0.2540 | **0.2504** | 0.3301 | 0.2981 | 0.2460 |
| JHB | 0.3136 | **0.2876** | 0.4326 | 0.3943 | 0.2919 |
| MEL | 0.3842 | **0.3542** | 0.4947 | 0.4336 | 0.3664 |
| SPO | 0.4882 | **0.4649** | 0.6386 | 0.5457 | 0.4867 |
| SZH | 0.0866 | **0.0861** | 0.1016 | — | 0.0786 |
| **五城平均** | **0.3053** | **0.2886（−5.5%）** | 0.3995（+31%） | 四城 +16% | 0.2939（−1.8%） |

- ③ 相对 ①：H=24 −5.5%，H=48 −2.5%，H=72 −3.1%；4/5 城小胜。
- ③ 相对 PatchTST+池只赢 1.8%，且在 LOA/SZH 输：**收益来自“逐站直接优化 pinball loss”，不是“Kairos 预训练”本身**。
- ④/⑤ 在所有城垫底，差距 12%~38%；④ 的 PICP@90 只有 0.45~0.50（名义 0.90），是结构性欠离散，不是位置偏。
- `np.sort(Q)` 修分位交叉，保留为标准后处理；但它**不是收益来源**：在 16 窗 × 32 站的分层诊断子集上实测仅 −0.01%~−0.08%（四个 99 分位方法 × 五城），此前“−1.0%~−2.4%”的表述无法复现，已撤回。
- AMS 测试期约 95.9% 的站数值冻结，**单列，不参与跨城平均**。

补充 baseline（iTransformer / STGCN / Graph WaveNet / TFT / DeepAR / MQ-CNN / 官方 CSDI / TimeGrad / TSDiff / DiffSTG / Chronos-Bolt / Moirai 等）的统一协议、三 horizon 主表与校准诊断见 [`docs/baselines_protocol.md`](docs/baselines_protocol.md)。

### 主干 × 头消融（H=24，crpsΣ）

| 城市 | PatchTST+池 | PatchTST+分位 | Kairos+分位（排序后） | Kairos+池 |
|---|---:|---:|---:|---:|
| MEL | 0.3664 | 0.3475 | 0.3542 | 0.3842 |
| SPO | 0.4867 | 0.4604 | 0.4649 | 0.4882 |

> 勘误：原表 MEL 的 0.3479 实为 PatchTST+分位在 `pre_len=24`（1925 窗）上的某个子结果，被误标成 Kairos+分位；SPO 的 0.4539 无对应产物。现统一为 `pre_len=72` 口径：PatchTST+分位 0.3475/0.4604，Kairos+分位 0.3542/0.4649（与主表 ③ 一致）。旧 `pre_len=24` 下 PatchTST+分位为 0.3483。

结论：分位头的增益与主干基本无关（PatchTST+池 → +分位：−5.2%；Kairos+池 → +分位：−7.8%）；在当前窗口（`seq_l=72`）下换主干不是主要矛盾，Kairos 预训练也没有带来额外收益（Kairos+分位比 PatchTST+分位差 1.9%）。

### 补充 baseline：TimeDiff（SPO，H=24）

| 指标 | TimeDiff（no-mixup 适配） | ③ 分位头 | ⑤ 吸收态扩散 |
|---|---:|---:|---:|
| crpsΣ | 0.6522 | 0.4539 | 0.5457 |
| PICP50 / PICP90 | 0.080 / 0.240 | 0.482 / — | — |
| MPIW50 | 2.07 | 4.76 | — |

官方源码不可得，使用第三方实现并按逐站单变量适配。训练时的 future mixup 会让去噪器“抄真值”，采样必崩（旧版 crpsΣ 0.8153、MPIW50 0.021）；关掉 mixup 后分布能张开，但仍明显欠离散（PICP50 仅 8%）。**该结果只作为“可复现适配版”的参考，不据此断言 TimeDiff 方法本身的上限。**

---

## 2. 零膨胀：零是状态，不是噪声

- **零是真实空闲，不是缺失**：`volume=0` 时 `duration` 也几乎总为 0（MEL/LOA/AMS 100%，JHB 93.4%）；非零功率落在 7–75 kW 的合理区间。
- **零只在站级出现，城市级看不到**：六城的全城合计都没有任何一小时为 0；单个站可以连续几十小时为 0。这正是宏观曲线平滑、单站曲线“零-峰”并存的原因。
- **站间异质性极大**：站级零值率中位从 0%（SZH）到 30.7%（MEL）；JHB/LOA/SPO 的站级 p90 达 39%~45%；样例站窗口内零值率 0%~73%。
- **零有强持续性**：现在零 → 下一小时仍零 84.6%~90.4%；现在正 → 零仅 0.4%~6.4%。
- **零侧 oracle 很大，但现有门拿不到**：若每格知道真零/非零，crpsΣ 可降 MEL −26.9%、LOA −14.8%、JHB −33.7%、SPO −32.1%、SZH −4.3%；现有门控的 AUC 0.76~0.83，但真非零格上仍留 18%~22% 零质量，单边硬规则收益 ≤0.7%。
- **峰值不可由历史稳定预测**：真正值残差线性 R² 仅 MEL .254 / LOA .302 / JHB .316 / SPO .123 / SZH .010；hurdle 的正值半没有优于主干点预测，crpsΣ 反而输给 ③。

零值专项分析：[`docs/数据零值分析.md`](docs/数据零值分析.md)；
零侧/峰侧待做实验：[`零侧峰侧实验_简版.md`](零侧峰侧实验_简版.md)；
图表汇总：[`docs/zero_inflation_report.html`](docs/zero_inflation_report.html)。

---

## 3. 数据与权重

**代码在 GitHub，大文件在 HuggingFace。**

| 内容 | 位置 | 说明 |
|---|---|---|
| 源代码 | [`github.com/yzzpz1015/Charged`](https://github.com/yzzpz1015/Charged) | 不含 `data/`、`result/`、权重 |
| 数据集 | HF Dataset [`yzzpz/Charged-data`](https://huggingface.co/datasets/yzzpz/Charged-data) | 原始 6 城 + `_remove_zero` 版本 + `tiles_dataset/` |
| 训练权重 | HF Model [`yzzpz/Charged-checkpoints`](https://huggingface.co/yzzpz/Charged-checkpoints) | ①③④⑤ 的 checkpoint（按 `result/` 子目录组织） |
| 主干权重 | `mldi-lab/Kairos_23m` | 训练时经 `--checkpoint` 自动下载 |
| 历史多模态权重 | GPT-2 / SigLIP v2 | 仅旧多模态线需要，概率主线不需要 |

### 下载数据

HF 仓库根目录是城市文件夹（`AMS/`、`JHB/`… 与 `tiles_dataset/`），项目期望的城市数据放在 `data/` 下：

```bash
# 直连慢时加：export HF_ENDPOINT=https://hf-mirror.com
hf download yzzpz/Charged-data --repo-type dataset --local-dir _hf_data
mkdir -p data
mv _hf_data/AMS* _hf_data/JHB* _hf_data/LOA* _hf_data/MEL* _hf_data/SPO* _hf_data/SZH* data/
mv _hf_data/tiles_dataset .
```

### 生成训练默认版本（`_remove_const`）

`_remove_zero` 仍可能残留“整段恒值”的站（恒 0 闲置残漏、恒非 0 的年度平铺伪影）。训练默认使用再剔一层恒值站的版本：

```bash
python3 script/filter_constant_sites.py
# 产出 data/<CITY>_remove_const/ 与排除清单 data/constant_sites.csv
```

### 训练权重

仓库**不包含** `result/` 下的训练产物（`.gitignore` 已排除 2.4 GB 的 Kairos L1 权重）。复现当前主结果需要的权重：

| 产物 | 路径 | 大小 | 说明 |
|---|---|---:|---|
| Kairos L1 `plain` / `quant` | `result/kairos_l1_clean/<CITY>/*_pre72_fold6.pth` | 12 × ~100 MB | ①/③ 主结果 |
| PatchTST+分位 | `result/patchtst_quant_clean/{MEL,SPO}/` | 7.4 MB | 主干消融 |
| CSDI / zidiff | `result/csdi_clean/` | 16 MB | ④/⑤ baseline |
| TimeDiff（no-mixup） | `result/timediff_nomix_clean/` | 30 MB | SPO/MEL 参考 |

> 注意：`result/timediff_clean/` 是 future-mixup 版，采样崩塌（MPIW50 0.02），**不要用于复现或对比**。

从 HF 取回权重（仓库内的目录名与 `result/` 下一致）：

```bash
# 直连慢时加：export HF_ENDPOINT=https://hf-mirror.com
hf download yzzpz/Charged-checkpoints --repo-type model --local-dir result
```

### 环境

测试环境：Python 3.10+、PyTorch 2.5.1 + CUDA 12.1、`transformers` 4.57.6，另需 `numpy pandas scipy matplotlib huggingface_hub`。

---

## 4. 评测口径（改模型前先对齐）

- **停摆行剔除**：某小时 >80% 的站同时为 0 → 整行丢弃（城市停摆）。
- **划分**：剩余可用行按时间先后**计数**切 train 2 / valid 1 / test 3（不是按日历天）。
- **窗口**：`seq_l=72`；目标块 `pre_len=24/48/72`；窗口要求历史与目标全部可用，历史允许跨界。
- **clean mask**：按“站 × 窗”剔除含注入式合成直线段的窗口（JHB 52%、SPO 42% 的窗口受影响）。
- **筛站**：只用**训练期**统计剔除死站/常数平台站（训练期均值 < 活跃站中位数的 1%，或 std/mean < 1%）；用 valid/test 筛站即泄漏。
- **AMS**：测试期 95.9% 站冻结，单列，不参与跨城平均。

### 指标

| 指标 | 含义 | 用法 |
|---|---|---|
| **crpsΣ** | 逐站边际 CRPS，站平均后按训练期站均值归一化 | **主判** |
| crpsΣsum | 先把同一样本的所有站求和，再算 CRPS | 联合结构参考；当前边际误差主导，分辨力有限 |
| crps_nz | 仅真值 >0 的格子 | 扩散类必报，防止“真零格白送分” |
| PICP / MPIW | 区间覆盖率 / 宽度 | 校准与欠离散诊断 |

对扩散类方法，**只报边际 crpsΣ 不够**：零格的“免费满分”会掩盖正值格的问题，必须同时报 `crps_nz` 和 PICP。

---

## 5. 复现命令

```bash
# 1) 残差池基线 + CRPS 脚手架（输出 result/prob_baseline/）
python3 script/prob_baseline.py

# 2) Kairos L1：点预测头 / 分位头（pre_len=72，评测时取 H=24/48/72）
python3 script/train_kairos_l1.py --city MEL --variant plain \
    --seq_l 72 --pre_len 72 --out result/kairos_l1_clean --device 0
python3 script/train_kairos_l1.py --city MEL --variant quant \
    --seq_l 72 --pre_len 72 --out result/kairos_l1_clean --device 0

# 3) 统一概率评测（①③ + 可选 ④⑤）
python3 script/kairos_prob_eval.py --cities LOA JHB MEL SPO SZH --clean \
    --out_root result/kairos_l1_clean --csdi_root result/csdi_clean \
    --skip_csdi --device 0

# 4) 扩散 baseline（先训练，再去掉上面的 --skip_csdi 评测）
python3 script/train_csdi.py  --cities LOA JHB MEL SPO SZH \
    --ckpt_root result/kairos_l1_clean --out result/csdi_clean --device 0
python3 script/train_zidiff.py --cities LOA JHB MEL SPO SZH \
    --ckpt_root result/kairos_l1_clean --out result/csdi_clean --device 0

# 5) PatchTST + 分位头消融（MEL / SPO）
python3 script/train_kairos_l1.py --city MEL --backbone patchtst --variant quant \
    --tag_prefix patch_quant --seq_l 72 --pre_len 24 \
    --out result/patchtst_quant_clean --device 0
python3 script/eval_patch_quant.py --city MEL --device 0

# 6) TimeDiff 参考（仅 SPO/MEL；no-mixup 已是默认）
python3 script/train_timediff.py --city SPO --out result/timediff_nomix_clean --device 0

# 7) 图与零膨胀报告
python3 script/viz_city_macro.py --city MEL
KAIROS_CKPT_ROOT=result/kairos_l1_clean \
    python3 script/prob_viz_multi.py MEL,SPO 168 200 0
python3 script/build_zero_inflation_report.py   # → docs/zero_inflation_report.html
```

> `--data_path` 默认 `data/`，运行时自动 append `--data_suffix`（默认 `_remove_const`）。
> 首跑 Kairos 会从 HF 下载 `mldi-lab/Kairos_23m`；可用 `--kairos_repo` 指定本地缓存目录。

---

## 6. 目录结构

```
api/
  model/        # kairos_l1 / csdi / zidiff / patchtst_quant / frets / ev_vlm …
  dataset/      # 数据加载、划分（停摆剔除 + 计数 2/1/3）、collate
  trainer/      # train/valid/test 循环
  parsing/      # CLI 参数
script/
  prob_baseline.py        # 残差池基线 + CRPS 脚手架
  train_kairos_l1.py      # L1 点预测 / 分位头 / PatchTST 消融
  kairos_prob_eval.py     # 统一概率评测（①③④⑤）
  train_csdi.py / train_zidiff.py / train_timediff.py
  prob_viz*.py / viz_city_macro.py / build_zero_inflation_report.py
docs/
  概率预测阶段总结_2026-10-20.md
  zero_inflation_report.html
  数据零值分析.md
data/                  # 从 HF 下载；_remove_const 由脚本生成
tiles_dataset/         # 卫星瓦片与 DINOv2 嵌入（仅历史多模态线）
result/                # 实验输出与权重（不入库）
```

---

## 7. 历史方向：外部模态与检索（负结果存档）

早期主线是“在微调之上，卫星图 / POI / 文本 / 邻居 / 检索能否再带来增量”。结论是**都不稳健**：

- 卫星图在 station-week 划分下，“随机替身码”不弱于真卫星图；leave-station-out 全负 → 图像更像站身份记忆。
- 视野 380 m 有微弱信号（+0.146），2.5 km 为 0；端到端微调 gate 失败；被 `charger_num`（0.761）碾压。
- POI / 土地利用只解释图像，不解释需求残差。
- 邻居流、温度/湿度/电价、TS-Memory kNN 检索均≈0 或负增量；L2 三路逐层注入在 pre24 跌破 L1。

相关材料：[`docs/satellite_transfer_verdict.html`](docs/satellite_transfer_verdict.html)、
[`docs/satellite_verdict_onepager.html`](docs/satellite_verdict_onepager.html)、
[`docs/Agent方向.md`](docs/Agent方向.md)（Agent 探索方向，未进入当前主线）。
