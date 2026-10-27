# 概率预测 Baseline 统一协议（#7–#21）

> 目的：在现有 #1–#6 之外补齐 15 个 baseline，全部使用**同一批测试窗、同一 clean mask、同一指标、同一 horizon 口径**，形成 21 方法横向对比表，并作为零膨胀方法的试验底座。
> 代码：`script/baselines/`；结果：`result/baselines/{metrics,diag,ckpt}/`。

## 1. 数据与划分

* 城市：LOA / JHB / MEL / SPO / SZH（AMS 单列，测试期 95.9% 站冻结，不并入主表）。
* 数据：`data/{city}_remove_zero/{city}_volume.csv`（`common.load_city` 自动追加 `_remove_zero`）。
* 划分：`EVDataset.split_cross_validation(fold=6, total_fold=6)` —— 先丢弃停摆小时（>80% 站为 0），可用行按计数 **2/1/3** 切 train/valid/test（不是 80/10/10）。
* 站过滤：只用**训练期**统计量剔除退化站（全零/常量/覆盖率不足），无泄漏。
  * 窗/站数自检：LOA 218、JHB 29、MEL 56、SPO 33、SZH 843。
  * 测试窗：LOA/JHB/SPO/SZH 2125，MEL 1781。
* 五城时间戳均为完整小时网格（T=4392，2023-04-01 00:00 – 2023-09-30 23:00，无缺失小时）；但 MEL 有 **19 行 >80% 站为 0**（三段：2023-07-08 23:00–07-09 10:00、07-11 13:00–16:00、07-29 19:00–21:00），按停摆规则丢弃后把可用时间轴切成 **4 段连续区间**。窗口只在区间内取，不跨断点；MEL 的测试窗因此是 1781 而不是 2125。
* 站集、窗集、clean mask 由 `common.load_city` 统一生成；任何方法窗数不符即判无效。

## 2. 窗口与 horizon 口径

* 统一 `seq_l = 72`、`pre_len = 72`，窗口起点 = `ds.window_starts(split, 72, 72)`。
* H = 24 / 48 / 72 都**不是**重新按 H 开窗，而是取同一 72 步输出的第 `j ∈ {23, 47, 71}` 步。
  这样 #1/#2 无需重训，三个 horizon 共用同一预测轨迹，避免三套窗口不可比。
* 一个模型一次前向出 72 步，测试集边推理边累计三个 horizon 的指标。

## 3. clean mask

`straight_window_keep(raw, starts, 72, 72)` → `[M, N]` bool，标记“窗口内无非物理直线段（合成线性斜坡）”的 站×窗。

* 训练/验证：作为**样本权重**（按 站×窗 置零），不是整条丢弃（JHB/SPO 的 clean 率仅约 35%/58%）。
* 测试：只在指标里屏蔽，不改变输入。

## 4. 指标

| 字段 | 含义 |
|---|---|
| `crps_norm_sum`（主指标 crpsΣ） | mean-of-means：先对每站求平均 CRPS 与平均 smean，再在站间取均值后相除 |
| `crps_sum_norm`（crpsΣsum） | Σ CRPS / (n_scored × Σ smean) |
| `crps_nz` | 只在真实值 >0 的点上算 CRPS，诊断零分布处理 |
| `picp50/80/90`, `mpiw50/80/90` | 区间覆盖率 / 平均宽度 |
| `wis` | 加权区间分数，口径与 crpsΣ 同为 mean-of-means（先逐站平均 WIS 与 smean，再站间平均后相除）；见下方公式 |
| `wis_nz` / `wis_zero` | 只在真实值 >0 / =0 的格上算 WIS；逐窗先对格取平均，再对窗取平均后除以 `mean(smean)` |
| `crps_zero_contrib` / `crps_pos_contrib` | crpsΣ 的**精确**零格/正格分解；两项相加恒等于 `crps_norm_sum`（逐站求和后同一 mean-of-means 归一化） |
| `crps_zero_mean` / `crps_pos_mean` | 零格/正格上 pooled 的 CRPS 均值（不做站间平均，仅供形状诊断） |
| `zero_cell_frac`, `n_zero_cells`, `n_pos_cells` | 参与评测的 clean cell 中真实值为 0 的比例与计数（方法无关；子集方法按子集窗计） |
| `mae`, `rmse`, `r2` | **点预测指标**：该 horizon 的预测点（分位方法取 τ=0.5，样本方法取样本中位数）在 clean 站×窗上 pool 后计算 |
| `n_point_cells` | 点指标实际参与的有效 站×窗 数（= clean 站×窗，每个 horizon 相同） |
| `n_scored_windows` | 实际参与指标的有效窗数（排除全不 clean 的窗） |

* CRPS 用 fair CRPS（`crps_fair`）。
* 分位方法统一转成 200 条样本再算 CRPS（`M_SAMPLE=200`，与 #1–#6 一致；`strat_u` 分层均匀采样 + `quant_to_samples_ind`，转前先 `np.sort(quantiles)`）。
* 样本方法（DeepAR / Moirai / CSDI）使用模型自身样本：DeepAR 200、Moirai 200、官方 CSDI 50（受采样成本限制，见 §7）。
* WIS 用 5 个中心区间：α ∈ {0.02, 0.05, 0.10, 0.20, 0.50}，分位水平 {0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99}，权重 w₀=0.5、w_k=α_k/2，分母 5.5：
  `WIS = [0.5|m−y| + Σ_k (α_k/2)((u_k−l_k) + (2/α_k)(l_k−y)₊ + (2/α_k)(y−u_k)₊)] / 5.5`。
  分位模型先用同一套 `quant_to_samples_ind` 展开成 200 条样本（与 CRPS 完全相同的样本），再由样本经验分位取水平；样本方法直接用模型自身样本。**注意** TimeGrad/TSDiff/DiffSTG 为 100 样本、CSDI 为 50 样本，0.01/0.99 尾部由最小/最大次序统计量估计，方差大于 200 样本；这会使扩散类的 WIS 尾部项略偏保守，跨方法比较时只作方向性参考。
* WIS 与零/正分解均由 `StreamingMetrics` 边推理边累计，**不从 `diag/*.npz` 反算**：诊断子集只有 16 窗 × 32 分层站，与全量评测不在同一批 cell 上。WIS 为重评产物，写 `metrics_wis/`（`BASELINE_TAG=wis`），不覆盖原 `metrics/`。
* 点指标口径：`mae = mean|ŷ−y|`；`rmse = sqrt(mean(ŷ−y)²)`；`r2 = 1 − SSE / (Σy² − (Σy)²/n)`（pool 后总体方差，非逐站 R²）。SZH 的 R² 被站间水平差主导，读表以 MAE/nMAE 为主。
* 统一汇总口径 `nMAE = MAE / mean(y)`（同一批 clean 站×窗），五城取简单平均，用于消除城市量纲差异。
* 点预测重评不重训：`BASELINE_TAG=point` 把指标写到 `metrics_point/`，ckpt 与原指标目录保持不变；`summarize_point.py` 只读该目录。

## 5. 归一化

各方法内部对每个窗口做 **CausalRevIN**：只用该窗口 72h 历史算 per-station mean/std，再反归一化回 raw 空间算 loss 与指标。禁止用未来窗统计量。

* 分位头方法：raw 空间 pinball（2×pinball 即分位形式 CRPS）。
* CSDI：只用训练期逐站 mean/std 做全局归一化（官方实现口径），std 下限 clamp。

## 6. 方法与实现

| # | 方法 | 脚本 | 关键实现 |
|---|---|---|---|
| 7 | iTransformer | `train_quant_backbone.py --method itransformer` | 变元 token Transformer，`d_model=64, nhead=4, layers=2, d_ff=128, dropout=0.1` |
| 8 | STGCN | 同上 `--method stgcn` | ST-Conv block + 距离 kNN 邻接 |
| 9 | STGCN-nograph | 同上 `--method stgcn_nograph` | 同结构、`A=I`，参数量与 #8 完全相同（图消融） |
| 10 | Graph WaveNet | 同上 `--method gwnet` | 复用现有 GWNet，构造时传邻接（事后赋值无效） |
| 11 | TFT | `train_gluonts.py --method tft` | GluonTS 0.14.4，逐站单变量，直接训练 99 分位；`hidden_dim=64, num_heads=4`（GluonTS TFT **没有** `num_layers` 参数） |
| 12 | DeepAR | `train_gluonts.py --method deepar` | GluonTS StudentT 似然，`hidden=64, layers=2`，`num_parallel_samples=200`，非负样本 |
| 13 | MQ-CNN | `train_quant_backbone.py --method mqcnn` | GluonTS torch 无官方实现，按原结构重写：因果空洞 conv（dilation 1–16）+ 共用分位头；**无 static station embedding**（与其它 baseline 同输入） |
| 14 | 官方 CSDI | `csdi_official.py` | `ermongroup/CSDI` 的 `CSDI_Forecasting`，模型/损失/扩散过程不改，仅替换数据适配层 |
| 15 | TimeGrad | `train_timegrad.py --method timegrad` | GRU 历史编码 + FiLM 条件 DDPM；**非自回归**（72 步联合扩散，原版逐步自回归），DDIM 50 步、η=1 |
| 16 | TSDiff | `train_tsdiff.py` | 整段 [72 历史+72 未来] 联合扩散，无 RNN；采样时把历史段硬投影回真值（hard self-guidance，替代原版软梯度引导）；DDIM 50 步、η=1 |
| 17 | DiffSTG | `train_diffstg.py` | 时空图扩散：[N,72] 联合扩散，每 block「时间膨胀 conv + kNN 稀疏图聚合 + 时间 FiLM + **历史 FiLM**」；历史条件用 GRU 末状态，并在每个 block 再做一次 FiLM（v3）。v1 的 `cond_enc(...).mean(dim=3)` 在 RevIN 后≈0；v2 只在输入层加一次条件，ε 学不动。v1/v2 产物已归档且不进主表。轻量版 UGnet；DDIM 50 步、η=1 |
| 18 | Chronos-Bolt | `run_zeroshot.py --method chronos_bolt` | 零样本，`chronos-bolt-small`，原生 9 分位 |
| 19 | Moirai | `run_zeroshot.py --method moirai` | 零样本，`moirai-1.0-R-small`，200 条样本→99 经验分位 |
| 20 | UniTS-FT + pool | `train_unist.py` | 公开权重 `units_x64_supervised_checkpoint.pth`（d_model=64）作主干；每城新建 dataset prompt、其余权重迁移，用 pre72 窗 + clean mask 少样本微调（30 epoch，patience 6，lr 1e-4，80 步/epoch）；概率输出走与 #1/#5 相同的残差池（roll28）。概率侧五城 seed 0 均已补齐，点预测五城也都跑；仍作附录行，不是官方 UniST（后者是 grid 模型） |
| 21 | OpenCity-Plus | `train_opencity.py` | 公开权重 `hkuds/OpenCity-Plus`（`OpenCity-plus.pth`）；72→72 只替换末层为 99 分位头，沿用官方 eval 口径**冻结主干、只训头**（30 epoch，patience 5，lr 1e-3，全训练集，head 2.62M / 总 25.06M）；kNN k=8 图 + Laplacian PE 8；`module.` 前缀剥离后 138 个张量加载成功 |

### 6.1 图邻接

`distance.csv` 的 index 为 int、columns 为 str 且顺序与 `volume.csv` 不同：必须先 `.astype(str)` 再 `reindex` 到 `ds.feat_columns`。
邻接 = 距离 kNN（k=8，权重 `exp(-d/median)`），对称化后加自环。

### 6.2 训练预算（#7–#10 共用）

* epochs 60，patience 10，每 epoch 固定 `steps_per_epoch=80` 个窗口批（大城小城同预算，避免 SZH 把 epoch 拉成几小时）。
* AdamW，lr 1e-3，weight decay 1e-4，cosine，grad clip 1.0。
* 早停看验证集 masked pinball；只保存 best。
* 四方法共用同一个分位头（`api/model/quant_head.py`）与同一 RevIN。

## 7. 已知差异与注意事项（表注必须写）

1. **官方 CSDI 只评测试窗子集**：采样成本随站数近线性（K=56 → 210 ms/样本/窗；K=843 → 3.06 s/样本/窗）。LOA/JHB/MEL/SPO 评 256 窗，SZH 评 128 窗，`nsample=50`。JSON 里 `n_test_windows / n_test_windows_total` 记录子集与全量。
2. **Chronos-Bolt 尾部被夹断**：原生只训练了 0.1–0.9 分位，请求 0.01–0.99 时管线把越界分位设成边界值，PICP90 偏低（约 0.78）属于方法限制，主看 PICP50/80；CRPS 仍按 99 分位积分。
3. **样本数不齐**：DeepAR/Moirai 200，CSDI 50，TimeGrad/TSDiff/DiffSTG 100。样本方法用自身样本；CSDI 与 SZH 扩散受采样成本限制，未对齐到 200。指标均为 fair CRPS，表注需写实际样本数。
4. **MEL 的停摆缺口**：MEL 有 **19 行 >80% 站为 0**（三段：2023-07-08 23:00–07-09 10:00、07-11 13:00–16:00、07-29 19:00–21:00）。时间戳网格本身完整（无缺失小时），但这 19 行观测不可用，按停摆规则丢弃后把可用时间轴切成 **4 段连续区间**。`split_window_starts` 要求整段 `[i, i+144)` 可用，跨缺口的窗口全部丢弃，所以 MEL 测试窗 1781、其余四城 2125。
   * **不能写成「不是缺数据」**：时间戳连续不等于观测可用。
   * GluonTS（#11/#12）直接取连续 `raw[row0:row1]`，这 19 行仍以 0 进入训练/上下文输入（未插值）；窗口方法不使用跨缺口窗口。该差异只影响 MEL，表注需写。
5. **seed 现状**：#7–#10、#13、#6 pre72、#11 TFT、#21 OpenCity-Plus（五城）为 3 seeds；#15 TimeGrad 仅 JHB/SPO 有 2 seeds；#17 DiffSTG 的 MEL 2 seeds、SPO 3 seeds；#20 UniTS-FT 概率侧单 seed；#14 CSDI、#12 DeepAR-ISQF 与零样本（#18/#19）为单次。主表对多 seed 方法统一报 mean±std，单 seed 直接报 seed 0。
6. **#6 pre72**：现有 `patchtst_quant_clean` 是 `pre_len=24`（1925 窗），不能与 pre72 口径直接比；本目录的 `patchtst_quant_pre72` 是同结构在 pre72 窗上重训的结果。
7. **训练/验证 split 不重叠**：GluonTS 训练序列截到 `starts["tr"][-1]+144`，验证序列只覆盖验证期（起点前移 72h 作 context）。
8. **扩散类（TimeGrad / TSDiff / DiffSTG）的采样口径**：三者都是模型自身样本（不是分位→样本），统一 `n_sample=100`（JSON 的 `n_sample` 记录实际值；与 #1–#6 的 200 不同，表注必须写）；DDIM 50 步、η=1（`eta=1` 补齐跳步随机性；η=0 会系统性欠离散；JHB/SPO 重评显示 η=1 后 CRPS 仅小幅改善，欠离散主要来自模型/数据而非采样设置）。训练用 ε-参数化 + 线性 beta（T=1000）。
   * **SZH 的 TSDiff / DiffSTG 只评等间隔 256 个测试窗**（N=843，全量采样约 20 小时）；与官方 CSDI 的 128 窗子集同为成本约束，JSON 的 `n_test_windows` 记录实际窗数。TimeGrad SZH 仍为全量 2125 窗。
9. **三个扩散 baseline 均为「按原论文结构重写、采样加速」**，不是官方代码：
   * TimeGrad：原版逐步自回归（每步一个扩散），这里是 GRU 历史编码 + 72 步联合扩散（非自回归）；
   * TSDiff：原版 soft self-guidance（观测一致性梯度），这里是 DDIM 每步把历史段硬投影回真值；不做 unconditional 预训练；
   * DiffSTG：原版 UGnet，这里是「GRU 历史编码 + 时间膨胀 conv + 稀疏 kNN 图聚合 + 时间/历史双 FiLM」轻量堆叠。历史条件经过两轮修正：
     - v1 用「历史特征的时间均值」作条件。RevIN 后均值≈0，模型拿不到当前周期位置，退化为预测窗口均值（JHB E[q]/target=0.67、PICP90=0.45）。
     - v2 改为 GRU 末状态，但只在输入层加一次；去噪 block 内没有历史条件，ε 仍学不动（JHB 测试窗 t=100 的 ε-MSE 0.89，TimeGrad/TSDiff≈0.23）。
     - v3 在**每个 block 用历史条件做 FiLM**（时间 FiLM + 历史 FiLM 相加），JHB 恢复校准（crpsΣ 0.3201、PICP90 0.922、E[q]/target 0.88）。主表只用 v3。
     - v1/v2 结果已移到 `result/baselines/archive_meancond_v1/` 与 `archive_diffstg_v2/`，不进主表。
   表注必须写明这三条，不能当作官方实现对比。
10. **MQ-CNN 无 static station embedding**：原版用站点类别 embedding，这里与 #7–#10 同输入（只有 72h 历史）；结构与损失为 MQ-CNN 的因果空洞 CNN + 分位头。
11. **#20 UniTS-FT 不是官方 UniST**：官方 UniST 是 grid（图像张量）模型，依赖跨场景预训练与多模态侧信息，无法直接接入站级 72h 窗口。这里改为「UniTS 公开权重主干 + 每城 dataset prompt 少样本微调 + 残差池」，只作附录行；概率侧五城 seed 0 均已补齐，点预测五城也都跑。
12. **#21 OpenCity-Plus 只训分位头**：官方 eval 口径是冻结时空主干、只训预测头；这里保留该口径，把 288→288 点预测头换成 72×99 分位头（head 2.62M / 总 25.06M），未做全参数微调。

## 8. 产物

```
result/baselines/
  metrics/{method}/{city}/H{24,48,72}_seed{s}.json        # 概率指标 + 超参 + 窗口数
  metrics_point/{method}/{city}/H{24,48,72}_seed{s}.json  # 同结构；另含 mae/rmse/r2/n_point_cells
  metrics_point/_y_mean.json                              # 每 (city,H) clean cells 的 mean(y)，nMAE 缓存
  diag/{method}/{city}/seed0.npz                          # 16 窗 × 32 分层站 × 72 × 99 分位，float16
  diag_point/{method}/{city}/seed0.npz                    # 同上；点重评通常加 --no_diag，当前为空目录
  ckpt/{method}/{city}/seed{s}.pth                        # 训练产物；BASELINE_TAG 不改此目录
  logs/*.log
```

* `BASELINE_TAG=point` **只切评测目录**：`metrics → metrics_point`、`diag → diag_point`，`ckpt` 不动。这样 `--eval_only` 仍能读到原 checkpoint，也不会覆盖旧 JSON；默认空 tag 写回 `metrics/`。
* 点指标不另存预测：`*_point.json` 的 `mae/rmse/r2/n_point_cells` 就是该 horizon 在 clean 站×窗上 pool 后的结果（分位方法取 τ=0.5，样本方法取样本中位数），与 `metrics/` 的 crpsΣ 共用同一批窗口和 mask。
* `metrics_point/_y_mean.json` 由 `summarize_point.py` 首次运行时生成并复用，只统计测试期 clean cells，不参与训练。
* `diag_point/` 与 `diag/` 同构；点重评为省时统一加了 `--no_diag`，所以当前 `diag_point/` 只有空目录（概率诊断 `diag/` 仍在，约 778 MB）。
* `unist_pool_smoke/` 是冒烟产物，不进主表、不上 HF；`archive_*` 是 DiffSTG v1/v2 归档，不进主表。

## 9. 复现

默认 seed 0；多 seed 加 `--seed 1` / `--seed 2`。补点指标时在命令前加 `BASELINE_TAG=point`，评测目录自动切到 `metrics_point/`，checkpoint 仍从 `ckpt/` 读取。

### 9.1 训练 + 概率评测

```bash
# #7–#10, #13（五城；seed 1/2 改 --seed）
for c in LOA JHB MEL SPO SZH; do
  for m in itransformer stgcn stgcn_nograph gwnet mqcnn; do
    python3 script/baselines/train_quant_backbone.py --method $m --city $c --device 0
  done
done

# #6 PatchTST+quant pre72
python3 script/baselines/train_patch_quant_pre72.py --city MEL --device 0

# #11 TFT / #12 DeepAR（主表用 ISQF）
python3 script/baselines/train_gluonts.py --method tft    --city MEL --device 0
python3 script/baselines/train_gluonts.py --method deepar --distr isqf --city MEL --device 0

# #14 官方 CSDI（默认均匀抽 256 窗；SZH 用 128 窗）
python3 script/baselines/csdi_official.py --city MEL --device 0 --n_eval_windows 256
python3 script/baselines/csdi_official.py --city SZH --device 0 --n_eval_windows 128

# #15 TimeGrad / #16 TSDiff / #17 DiffSTG
python3 script/baselines/train_timegrad.py --city MEL --device 0
python3 script/baselines/train_tsdiff.py  --city MEL --device 0
python3 script/baselines/train_diffstg.py --city MEL --device 0

# #18/#19 零样本（确定性，单次）
python3 script/baselines/run_zeroshot.py --method chronos_bolt --city MEL --device 0
python3 script/baselines/run_zeroshot.py --method moirai --city MEL --device 0

# #20 UniTS-FT + pool（公开主干 + 每城 prompt；附录行）
# 权重：external/weights/units_x64_supervised_checkpoint.pth
python3 script/baselines/train_unist.py --city MEL --device 0

# #21 OpenCity-Plus（冻结主干、只训 99 分位头）
# 权重：external/OpenCity/model_weights/OpenCity/OpenCity-plus.pth
python3 script/baselines/train_opencity.py --city MEL --seed 0 --device 0
```

### 9.2 点预测补评（`--eval_only`，不重训）

```bash
# 主干类
BASELINE_TAG=point python3 script/baselines/train_quant_backbone.py \
  --method itransformer --city LOA --seed 0 --device 0 --eval_only --no_diag
BASELINE_TAG=point python3 script/baselines/train_patch_quant_pre72.py \
  --city LOA --seed 0 --device 0 --eval_only --no_diag

# TFT：fast path，batch 只影响速度/显存，不影响结果
BASELINE_TAG=point python3 script/baselines/train_gluonts.py \
  --method tft --city LOA --seed 0 --device 0 --eval_only \
  --predict_batch_size 4096 --no_diag

# DeepAR / ISQF：4096 在 LOA/SZH 会 OOM，用 512；SZH 另加 --limit_test 256
BASELINE_TAG=point python3 script/baselines/train_gluonts.py \
  --method deepar --distr isqf --city LOA --seed 0 --device 0 --eval_only \
  --predict_batch_size 512 --no_diag

# 扩散类
BASELINE_TAG=point python3 script/baselines/csdi_official.py \
  --city LOA --device 0 --eval_only --n_eval_windows 256 --no_diag
BASELINE_TAG=point python3 script/baselines/train_timegrad.py \
  --city LOA --seed 0 --device 0 --eval_only --n_sample 100 --no_diag
BASELINE_TAG=point python3 script/baselines/train_tsdiff.py \
  --city LOA --seed 0 --device 0 --eval_only --n_sample 100 --no_diag
BASELINE_TAG=point python3 script/baselines/train_diffstg.py \
  --city LOA --seed 0 --device 0 --eval_only --n_sample 100 \
  --sample_windows 8 --sample_chunk 8 --no_diag

# 零样本
BASELINE_TAG=point python3 script/baselines/run_zeroshot.py \
  --method moirai --city SZH --device 0 --no_diag \
  --batch_windows 8 --batch_series 512

# OpenCity / UniTS
BASELINE_TAG=point python3 script/baselines/train_opencity.py \
  --city LOA --seed 0 --device 0 --eval_only --no_diag
BASELINE_TAG=point python3 script/baselines/train_unist.py \
  --city LOA --seed 0 --device 0 --eval_only
```

### 9.3 汇总

```bash
# 概率主表（crpsΣ 等）
python3 script/baselines/summarize.py
python3 script/baselines/summarize.py --markdown

# 点预测表（MAE/RMSE/R²；默认读 metrics_point/）
python3 script/baselines/summarize_point.py --seeds 0,1,2 --markdown
python3 script/baselines/summarize_point.py --seeds 0,1,2 --metric rmse
python3 script/baselines/summarize_point.py --seeds 0,1,2 --metric r2

# 若要用 summarize.py 读 metrics_point/，加 --tag point（不是独立脚本）
python3 script/baselines/summarize.py --tag point
```

### 9.4 显存敏感参数（只影响能否跑完，不改变模型结果）

| 方法 | 默认/过大时的症状 | 实际使用 |
|---|---|---|
| TFT | `--predict_batch_size 4096` 可跑；输出只有 99 分位，不展开 200 样本 | `--predict_batch_size 4096`，只用于提速 |
| DeepAR / ISQF | 4096 在 LOA/SZH `OutOfMemoryError`（输出 200 样本 × 72 步） | `--predict_batch_size 512`；SZH 再加 `--limit_test 256` |
| Moirai | 默认 32 窗 × 4096 序列在 LOA/MEL/SPO/SZH OOM | `--batch_windows 8 --batch_series 512` |
| DiffSTG | SZH 默认采样请求 18.6 GB，OOM | SZH `--sample_windows 2 --sample_chunk 8`；LOA `--sample_windows 8 --sample_chunk 8` |
| CSDI | SZH 843 站全站并行采样 OOM | `--n_eval_windows 128 --eval_batch 2 --sample_chunk 2`（小城市自动 8/32） |

`--sample_windows` / `--eval_batch` / `--predict_batch_size` 都只切分推理批次；结果应与不分块一致（TFT fast path 已用 32 窗逐条对账，byte-identical）。若怀疑 fast path，可加 `--slow_eval` 走 GluonTS 官方 `predict`，速度慢很多。

## 10. 当前状态（2026-10-26 15:30）

* #7 / #13 / #6（iTransformer / MQ-CNN / PatchTST+quant-pre72）与 #8–#10（STGCN / STGCN-nograph / GWNet）：五城 seed 0/1/2 完成。
* #11 TFT：五城 seed 0/1/2 完成。LOA/JHB/MEL 的 seed 差异 <2%；SZH 约 6%；**SPO H24 seed0/1/2 = 0.5555 / 0.4655 / 0.5532（H72 达 0.7941 / 0.5311 / 0.6429）**，SPO 训练集仅 33 站，报告用 mean±std，不把单点当稳定排名。
* #12 DeepAR：主表用 ISQF 版本（`--distr isqf`），五城 seed 0 完成；StudentT 默认版本在多数城市退化（预测塌到 0，SPO crpsΣ 0.87），仅作参考、不进主表。
* #13 MQ-CNN：五城 seed 0/1/2 完成。
* #14 官方 CSDI：五城完成（LOA/JHB/MEL/SPO 256 窗，SZH 128 窗，`nsample=50`）。
* #15 TimeGrad：五城全量完成。H24 crpsΣ LOA/JHB/MEL/SPO/SZH = 0.2545 / 0.2861 / 0.3980 / 0.5139 / 0.1028；SZH PICP90 0.796，偏欠离散。
* #16 TSDiff：五城完成（SZH 为 256 窗子集），H24 crpsΣ 0.2701 / 0.2942 / 0.3516 / 0.5415 / 0.0889*；LOA/SZH PICP90 分别为 0.922 / 0.876。
* #17 DiffSTG v3（每 block 历史 FiLM）：
  * JHB 0.3201（PICP50/90 0.515/0.922，已校准）；
  * MEL seed0/1 0.4294/0.4209（PICP90 0.603/0.633，仍偏欠离散，但 seed 稳定）；
  * SPO 三 seed 0.6927/0.5828/0.5594，PICP50 0.056/0.393/0.648 —— **seed0 是极端欠离散个案，不是方法均值**，主表用 0.612±0.058 并注明；
  * SZH（256 窗子集）H24 0.2363、PICP50/90 0.282/0.696：明显欠离散，且比 TimeGrad（0.1028）/ TSDiff（0.0889*）差一个档；843 站图扩散在本数据上不稳，表注需写。
  * LOA 全量 H24 0.3931（PICP50/90 0.151/0.677，MPIW50 1.27）：同样严重欠离散。综合看，DiffSTG 只有在 29 站的 JHB 校准良好，站数一放大就塌；v1/v2 已归档，不进主表。
* #18/#19 Chronos-Bolt / Moirai：五城完成（确定性，单次）。
* #20 UniTS-FT + pool：概率与点预测均五城 seed 0 完成（概率侧原先只有 MEL/SPO，LOA/JHB/SZH 已从 `--eval_only` 产物补齐）。仍作附录行，不是官方 UniST，见 §7.11。
* #21 OpenCity-Plus：概率侧与点预测侧五城 seed 0/1/2 全部完成（冻结主干、只训分位头）；概率 seed 1/2 于 2026-10-27 用 `--eval_only` 补评，`metrics/` 与 `metrics_wis/` 均已补齐，概率表改报 mean±std。
* **T4 phase2 WIS 重评完成（2026-10-26 15:01）**：16 个方法 × 5 城 × 3 H 共 465 个 `metrics_wis` JSON 全部落盘（每个 horizon 155 个）。WIS 由 `StreamingMetrics` 在评测时累计，没有用 `diag/*.npz` 反算；结果见 §11.4。
* **T1 点预测补齐完成**：CSDI-SZH、DiffSTG-LOA、TimeGrad-SZH 已用 `BASELINE_TAG=point --eval_only` 补评，点预测主表覆盖 496 个 JSON，见 §11.3。
* **T5 零/正格分解完成**：基于同一批 clean cells 的精确分解（closure 最大误差 3.38e-7），见 §11.5。
* **DiffSTG-LOA 采样不稳**：同一 ckpt 在 `sample_chunk=16`（主评测）与 `=8`（WIS 重评）下 crpsΣ 0.3931 vs 0.9913，仅 RNG 流不同。§11.4 的 crpsΣ/PICP/MPIW 用主评测、WIS 用重评，该行标 `‡`，不作跨指标比较。
* **指标 JSON 已同步 HF（2026-10-27）**：`baselines/metrics` 510 个、`metrics_point` 496 个、`metrics_wis` 495 个；补齐 OpenCity seed1/2、UniTS 的 LOA/JHB/SZH、TimeGrad/CSDI 的 SZH 点指标、DiffSTG 的 LOA 点指标。`unist_pool_smoke` 已从远端删除。
* 时空 FM 调研（2026-10-25）：FactoST 官方仓库返回 404，代码与权重均不可得；UrbanDiT 代码公开（MIT）但 **无预训练权重、无 release**，只有从零训练脚本，不能当 FM 基线；UniTS 权重公开（x32/x64/x128，GitHub release `ckpt`），但它是多变量通用模型、无图结构、原生输出点预测而非分布。可落地的公开权重时空 FM 只有 **OpenCity-Plus**（#21，已接入）与 **UniTS**（#20，附录行）。

## 11. 主结果表（crpsΣ，越低越好）

**口径**：同一 clean mask、`seq_l=pre_len=72`；H=24/48/72 取自同一 72 步输出的第 23/47/71 步。`seeds` 列是该行实际 seed 数的范围；多 seed 报 mean±std，单 seed 直接报 seed 0。平均为先逐城对 seed 取均值、再对可用城市做简单平均。

**为什么拆成两张表**：概率结果的测试窗覆盖并不一致，**窗数相同 ≠ 窗集相同**。实测：

* CSDI 用 `np.linspace(0, len(starts)-1, n)` 等间隔抽窗；其 256 窗与「前 256 窗」只重叠 LOA/JHB/SPO **31/256**、MEL **37/256**；SZH 的 128 窗与 TSDiff/DiffSTG 的 256 窗只重叠 **18/128**。
* TSDiff / DiffSTG 的 SZH 256 窗同样用 `np.linspace` 等间隔抽样（两个脚本与 CSDI 同一公式），二者窗集完全相同；但都不是「前 256 窗」。
* 因此 CSDI 只能单独看；TSDiff/DiffSTG 的 SZH 可互比，但不能与全量表的 SZH 比较。

表注：`—` 未跑；`†` 该城只在子集窗上评测，已移入 11.B；`（4 城）` 表示 SZH 在子集表，行平均只对四个全量城市取平均，**不可与五城平均直接比较**（SZH 的 crpsΣ 绝对水平显著低于其余城市，五城平均会被它拉低）。

### 11.A 全量测试窗

#### H=24

| # | 方法 | seeds | LOA | JHB | MEL | SPO | SZH | 五城平均 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 13 | MQ-CNN | 3 | 0.2463±0.0004 | 0.2848±0.0009 | 0.3474±0.0004 | 0.4612±0.0030 | 0.0814±0.0001 | **0.2842** |
| 6 | PatchTST+quant-pre72 | 3 | 0.2442±0.0004 | 0.2835±0.0028 | 0.3496±0.0016 | 0.4621±0.0025 | 0.0845±0.0002 | 0.2848 |
| 7 | iTransformer | 3 | 0.2458±0.0015 | 0.2893±0.0016 | 0.3534±0.0009 | 0.4771±0.0019 | 0.0797±0.0002 | 0.2891 |
| 21 | OpenCity-Plus | 3 | 0.2466±0.0019 | 0.2927±0.0025 | 0.3592±0.0012 | 0.4786±0.0040 | 0.0829±0.0005 | 0.2920 |
| 10 | Graph WaveNet | 3 | 0.2505±0.0007 | 0.2920±0.0021 | 0.3708±0.0023 | 0.4876±0.0052 | 0.0870±0.0002 | 0.2976 |
| 11 | TFT | 3 | 0.2381±0.0009 | 0.2892±0.0047 | 0.3484±0.0009 | 0.5247±0.0419 | 0.0931±0.0024 | 0.2987 |
| 20 | UniTS-FT+pool | 1 | 0.2470 | 0.3168 | 0.3685 | 0.4959 | 0.0810 | 0.3018 |
| 8 | STGCN | 3 | 0.2639±0.0001 | 0.2947±0.0014 | 0.3734±0.0013 | 0.4895±0.0020 | 0.0920±0.0001 | 0.3027 |
| 18 | Chronos-Bolt | 1 | 0.2643 | 0.2977 | 0.3600 | 0.5189 | 0.0868 | 0.3055 |
| 9 | STGCN-nograph | 3 | 0.2677±0.0002 | 0.2958±0.0012 | 0.3799±0.0011 | 0.4937±0.0008 | 0.0909±0.0002 | 0.3056 |
| 15 | TimeGrad | 1–2 | 0.2545 | 0.2853±0.0008 | 0.3980 | 0.5066±0.0073 | 0.1028 | 0.3095 |
| 19 | Moirai | 1 | 0.2722 | 0.3043 | 0.3797 | 0.5329 | 0.1102 | 0.3199 |
| 12 | DeepAR-ISQF | 1 | 0.2942 | 0.3287 | 0.4302 | 0.5632 | 0.1909 | 0.3614 |
| 16 | TSDiff | 1 | 0.2701 | 0.2942 | 0.3516 | 0.5415 | —† | 0.3644（4 城） |
| 17 | DiffSTG | 1–3 | 0.3931 | 0.3201 | 0.4251±0.0042 | 0.6117±0.0581 | —† | 0.4375（4 城） |

#### H=48

| # | 方法 | seeds | LOA | JHB | MEL | SPO | SZH | 五城平均 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 13 | MQ-CNN | 3 | 0.2651±0.0002 | 0.3356±0.0021 | 0.3683±0.0007 | 0.5088±0.0030 | 0.0859±0.0001 | **0.3128** |
| 6 | PatchTST+quant-pre72 | 3 | 0.2635±0.0004 | 0.3285±0.0026 | 0.3695±0.0015 | 0.5142±0.0051 | 0.0885±0.0002 | 0.3128 |
| 7 | iTransformer | 3 | 0.2616±0.0007 | 0.3448±0.0048 | 0.3734±0.0008 | 0.5201±0.0015 | 0.0839±0.0001 | 0.3167 |
| 21 | OpenCity-Plus | 3 | 0.2653±0.0019 | 0.3410±0.0021 | 0.3780±0.0041 | 0.5248±0.0036 | 0.0864±0.0003 | 0.3191 |
| 20 | UniTS-FT+pool | 1 | 0.2624 | 0.3543 | 0.3847 | 0.5224 | 0.0850 | 0.3218 |
| 10 | Graph WaveNet | 3 | 0.2726±0.0017 | 0.3456±0.0016 | 0.3984±0.0039 | 0.5480±0.0054 | 0.0920±0.0002 | 0.3313 |
| 11 | TFT | 3 | 0.2540±0.0014 | 0.3468±0.0103 | 0.3651±0.0010 | 0.6091±0.0684 | 0.0963±0.0024 | 0.3342 |
| 8 | STGCN | 3 | 0.2927±0.0004 | 0.3426±0.0032 | 0.3984±0.0011 | 0.5453±0.0016 | 0.0985±0.0002 | 0.3355 |
| 9 | STGCN-nograph | 3 | 0.2962±0.0002 | 0.3420±0.0016 | 0.4020±0.0006 | 0.5495±0.0008 | 0.0973±0.0002 | 0.3374 |
| 15 | TimeGrad | 1–2 | 0.3060 | 0.3340±0.0007 | 0.4271 | 0.5513±0.0091 | 0.1286 | 0.3494 |
| 18 | Chronos-Bolt | 1 | 0.2944 | 0.3704 | 0.3835 | 0.6156 | 0.0917 | 0.3511 |
| 19 | Moirai | 1 | 0.3105 | 0.3849 | 0.4176 | 0.6446 | 0.1281 | 0.3771 |
| 12 | DeepAR-ISQF | 1 | 0.3214 | 0.3723 | 0.4631 | 0.6029 | 0.2508 | 0.4021 |
| 16 | TSDiff | 1 | 0.2946 | 0.3419 | 0.3735 | 0.6038 | —† | 0.4034（4 城） |
| 17 | DiffSTG | 1–3 | 0.4028 | 0.3467 | 0.4414±0.0050 | 0.6419±0.0615 | —† | 0.4582（4 城） |

#### H=72

| # | 方法 | seeds | LOA | JHB | MEL | SPO | SZH | 五城平均 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 6 | PatchTST+quant-pre72 | 3 | 0.2701±0.0008 | 0.3474±0.0016 | 0.3879±0.0018 | 0.5242±0.0035 | 0.0895±0.0002 | **0.3238** |
| 13 | MQ-CNN | 3 | 0.2724±0.0004 | 0.3562±0.0009 | 0.3870±0.0008 | 0.5245±0.0030 | 0.0874±0.0001 | 0.3255 |
| 21 | OpenCity-Plus | 3 | 0.2705±0.0018 | 0.3533±0.0016 | 0.3980±0.0042 | 0.5344±0.0017 | 0.0878±0.0007 | 0.3288 |
| 7 | iTransformer | 3 | 0.2673±0.0012 | 0.3666±0.0025 | 0.3931±0.0004 | 0.5381±0.0017 | 0.0851±0.0001 | 0.3301 |
| 20 | UniTS-FT+pool | 1 | 0.2655 | 0.3752 | 0.4028 | 0.5371 | 0.0862 | 0.3334 |
| 10 | Graph WaveNet | 3 | 0.2818±0.0019 | 0.3707±0.0012 | 0.4152±0.0039 | 0.5593±0.0013 | 0.0940±0.0001 | 0.3442 |
| 8 | STGCN | 3 | 0.3008±0.0004 | 0.3702±0.0027 | 0.4152±0.0014 | 0.5687±0.0016 | 0.1008±0.0001 | 0.3511 |
| 11 | TFT | 3 | 0.2560±0.0010 | 0.3677±0.0115 | 0.3810±0.0010 | 0.6561±0.1078 | 0.0975±0.0029 | 0.3517 |
| 9 | STGCN-nograph | 3 | 0.3037±0.0001 | 0.3693±0.0008 | 0.4193±0.0005 | 0.5700±0.0008 | 0.0998±0.0002 | 0.3524 |
| 15 | TimeGrad | 1–2 | 0.3188 | 0.3593±0.0013 | 0.4411 | 0.5664±0.0068 | 0.1376 | 0.3646 |
| 18 | Chronos-Bolt | 1 | 0.2973 | 0.4134 | 0.4096 | 0.6726 | 0.0937 | 0.3773 |
| 19 | Moirai | 1 | 0.3177 | 0.4126 | 0.4389 | 0.6522 | 0.1323 | 0.3907 |
| 12 | DeepAR-ISQF | 1 | 0.3283 | 0.3834 | 0.4832 | 0.6021 | 0.2549 | 0.4104 |
| 16 | TSDiff | 1 | 0.3009 | 0.3726 | 0.4089 | 0.6122 | —† | 0.4237（4 城） |
| 17 | DiffSTG | 1–3 | 0.3693 | 0.3551 | 0.4553±0.0017 | 0.6337±0.0493 | —† | 0.4534（4 城） |

### 11.B 子集测试窗（不报跨方法平均/排名）

#### H=24

| # | 方法 | seeds | 窗数 LOA/JHB/MEL/SPO/SZH | LOA | JHB | MEL | SPO | SZH |
|---|---|---:|---|---:|---:|---:|---:|---:|
| 14 | CSDI（官方） | 1 | 256/256/256/256/128 | 0.2962 | 0.3290 | 0.4552 | 0.4927 | 0.1050 |
| 16 | TSDiff | 1 | —/—/—/—/256 | — | — | — | — | 0.0889 |
| 17 | DiffSTG | 1 | —/—/—/—/256 | — | — | — | — | 0.2363 |

#### H=48

| # | 方法 | seeds | 窗数 LOA/JHB/MEL/SPO/SZH | LOA | JHB | MEL | SPO | SZH |
|---|---|---:|---|---:|---:|---:|---:|---:|
| 14 | CSDI（官方） | 1 | 256/256/256/256/128 | 0.3126 | 0.3757 | 0.4621 | 0.5515 | 0.1147 |
| 16 | TSDiff | 1 | —/—/—/—/256 | — | — | — | — | 0.0959 |
| 17 | DiffSTG | 1 | —/—/—/—/256 | — | — | — | — | 0.2367 |

#### H=72

| # | 方法 | seeds | 窗数 LOA/JHB/MEL/SPO/SZH | LOA | JHB | MEL | SPO | SZH |
|---|---|---:|---|---:|---:|---:|---:|---:|
| 14 | CSDI（官方） | 1 | 256/256/256/256/128 | 0.3044 | 0.3928 | 0.4561 | 0.5392 | 0.1177 |
| 16 | TSDiff | 1 | —/—/—/—/256 | — | — | — | — | 0.1066 |
| 17 | DiffSTG | 1 | —/—/—/—/256 | — | — | — | — | 0.1919 |

**子集表解读边界**：

* CSDI 的五城数字都在等间隔子集上，**不能与 11.A 的任何一行比较**；它的 128 窗 SZH 与 TSDiff/DiffSTG 的 256 窗只重叠 18/128，也不能与二者比较。
* TSDiff 与 DiffSTG 的 SZH 共用同一批 256 个等间隔窗，二者可以互比（H24 0.0889 vs 0.2363），但仍**不能**与全量 SZH 比较。
* CSDI `nsample=50`，TSDiff/DiffSTG `nsample=100`；这也是方法间的差异，不能只归因于窗口子集。

### 11.1 读表要点

1. **逐站分位头类仍占据前三**：MQ-CNN 0.2842、PatchTST+quant-pre72 0.2848、iTransformer 0.2891（H24 五城平均）。三者最大差 0.0049，与 PatchTST/iTransformer 的 seed std（最大 ±0.0028/±0.0019）同量级；**前三内部不排名**。
2. **两个公开权重时空 FM 落在中游**：OpenCity-Plus 0.2920（H24，第 4，3 seeds，逐城 std ≤0.004）受益于冻结主干 + 分位头，明显好于 UniTS-FT+pool 0.3018；UniTS 的 LOA/SZH 很强（0.2470/0.0810），但 JHB 0.3168 拖累平均。UniTS 概率侧仍只有 seed 0，不能当稳定排名；OpenCity 现已 3 seeds。
3. **图结构收益很小且不稳定**：STGCN vs STGCN-nograph 的差异在 LOA/JHB/MEL/SPO 为 0.5–1.4%，SZH 反向；GWNet 在 SZH 与 iTransformer 持平，其余城市落后 2–5%。站间弱相关在数据层面已确认，图卷积拿不到额外信息。
4. **零样本 Chronos-Bolt 很强**（H24 五城平均 0.3055，全量五城），与 TimeGrad 0.3095 接近；Moirai 0.3199 弱一档。预训练 TSFM 先验有用，但不及逐站微调的分位头。
5. **扩散类整体不进入上半区**：TimeGrad 0.3095 是全量五城；TSDiff 只有四个全量城（0.3644，4 城），DiffSTG 同（0.4375，4 城），CSDI 只有子集。**TSDiff 与 TimeGrad 的覆盖不同（TSDiff 的 SZH 是子集窗），且都缺少足够的 seed 证据，不作排序**。若把 TSDiff 的 SZH 子集值硬拼成「五城」会得到 0.3093，与 TimeGrad 的 0.3095 只差 0.0002；这个差值远小于扩散类在 SPO 上的 seed 波动，但拼接本身跨了口径，只能说明「不可排序」，不能当作结果。DiffSTG 在 LOA/SZH 严重欠离散，CSDI 在子集上同样欠离散。
6. **DeepAR-ISQF 明显落后**（0.3614），SZH 尤其差（0.1909）；GluonTS 逐站单变量 + ISQF 在大规模站点集上效率低。
7. **TFT 的 SPO 方差极大**：H24 0.5247±0.0419（per-seed 0.4655–0.5555），H72 0.6561±0.1078（per-seed 0.5311–0.7941）；SPO 训练集仅 33 站，seed 选择会显著改变结论。现在五城均为 3 seeds，表中统一 mean±std，**不再用 (min/max)，也不把 TFT 的 SPO 单点当稳定排名**。
8. **DiffSTG 的失败是结构性的**：JHB（29 站）校准良好（PICP90 0.922），但 MEL（56 站）0.618、LOA（218 站）0.677、SZH（843 站）0.696，且 LOA 的 MPIW50 只有 1.27；SPO 三 seed 在 0.5594–0.6927 间摆动。当前轻量图扩散在站数放大时学不动历史条件。

### 11.2 校准诊断（H=24；PICP 多 seed 先逐城平均再对可用城市平均，MPIW50 逐城报）

名义 PICP50=0.50、PICP90=0.90。crpsΣ 会被「真零格白送分」污染，扩散/深度生成方法必须同时看这张表。表中「口径」列标明覆盖：**全量**=五城全量；**4 城**=SZH† 不进平均；**子集**=CSDI 的等间隔子集。不同口径的 crps_nz 只作形状参考，不作横向排名。

MPIW50 的单位是原始 kWh/h，各城需求水平相差约一个数量级（MEL/SZH 的 40+ 对 LOA/JHB 的 4–7），跨城平均会被大城主导甚至翻转符号（LOA 上 TFT 4.4 < iTransformer 4.7，但五城简单平均为 21.5 > 20.0）。因此 **MPIW50 只在同一城的列内比较，不做跨城平均**。表中数值为多 seed 逐城平均；† 表示该城为子集口径（CSDI 全部，TSDiff/DiffSTG 的 SZH）。

| # | 方法 | 口径 | PICP50 | PICP90 | MPIW50 LOA | MPIW50 JHB | MPIW50 MEL | MPIW50 SPO | MPIW50 SZH | crps_nz |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | PatchTST+quant-pre72 | 全量 | 0.512 | 0.885 | 4.4 | 7.0 | 41.0 | 5.6 | 42.3 | 0.382 |
| 7 | iTransformer | 全量 | 0.549 | 0.883 | 4.7 | 7.5 | 39.9 | 5.6 | 42.4 | 0.377 |
| 8 | STGCN | 全量 | 0.546 | 0.849 | 4.4 | 7.3 | 45.0 | 5.4 | 43.5 | 0.391 |
| 9 | STGCN-nograph | 全量 | 0.546 | 0.850 | 4.5 | 7.3 | 46.4 | 5.2 | 41.7 | 0.394 |
| 10 | Graph WaveNet | 全量 | 0.568 | 0.880 | 4.4 | 7.2 | 45.1 | 5.3 | 49.2 | 0.394 |
| 11 | TFT | 全量 | 0.504 | 0.868 | 4.4 | 6.7 | 42.3 | 5.4 | 48.6 | 0.362 |
| 12 | DeepAR-ISQF | 全量 | 0.368 | 0.809 | 5.5 | 9.1 | 24.2 | 5.6 | 95.1 | 0.496 |
| 13 | MQ-CNN | 全量 | 0.540 | 0.893 | 4.6 | 6.9 | 42.5 | 6.3 | 43.9 | 0.379 |
| 14 | CSDI（官方） | 子集 | 0.280 | 0.590 | 3.4† | 4.1† | 42.7† | 8.1† | 42.2† | 0.439 |
| 15 | TimeGrad | 全量 | 0.375 | 0.780 | 4.4 | 6.6 | 52.4 | 4.7 | 58.7 | 0.418 |
| 16 | TSDiff | 4 城（不含 SZH†） | 0.429 | 0.852 | 5.7 | 7.5 | 46.0 | 6.1 | 60.5† | 0.437 |
| 17 | DiffSTG | 4 城（不含 SZH†） | 0.351 | 0.737 | 1.3 | 7.5 | 51.1 | 3.8 | 68.4† | 0.567 |
| 18 | Chronos-Bolt | 全量 | 0.540 | 0.776 | 4.1 | 4.9 | 37.0 | 5.3 | 50.6 | 0.393 |
| 19 | Moirai | 全量 | 0.347 | 0.852 | 4.1 | 5.0 | 38.9 | 5.1 | 76.8 | 0.403 |
| 20 | UniTS-FT+pool | 全量 | 0.449 | 0.801 | 4.3 | 6.9 | 35.4 | 6.1 | 39.3 | 0.384 |
| 21 | OpenCity-Plus | 全量 | 0.453 | 0.882 | 4.3 | 6.5 | 39.1 | 5.4 | 43.1 | 0.389 |

要点：

* 分位头类（#6/#7/#13）与图主干类（#8–#10）PICP90 在 0.85–0.89，接近名义 0.90；逐城看它们的 MPIW50 与扩散类同量级，crpsΣ 优势不是靠把区间开大。
* 官方 CSDI 是最严重的欠离散（PICP50 0.28、PICP90 0.59，且为子集口径）——与主实验 #3 的结论一致：整块扩散在本数据上结构性欠离散。
* TimeGrad / DiffSTG 也明显欠离散（PICP90 0.78 / 0.74）；TSDiff 在四个全量城上相对好（0.852），但仍是单 seed，且 SZH 子集未计入。
* **TFT 的 `crps_nz` 0.362 略优于 iTransformer 0.377，但边际 crpsΣ 更差（0.2987 vs 0.2891）**。区间宽度的方向逐城不一致：TFT 在 LOA/JHB/SPO 更窄（4.4/6.7/5.4 vs 4.7/7.5/5.6），在 MEL/SZH 更宽（42.3/48.6 vs 39.9/42.4），所以不能用五城平均的 21.5 vs 20.0 下结论。WIS 与零/正格分解（§11.4/§11.5）确认 TFT 在正值格确实更准（WIS_nz 三个 horizon 全场最低），边际落后的来源是 SPO 零格（TFT 0.2080 vs iTransformer 0.1299），而不是正值格区间更宽。
* DiffSTG 的 `crps_nz` 在扩散类中最高（0.567，4 城口径）：正值格误差比边际 crpsΣ 显示的更差，符合「零格白送分掩盖正值格问题」的判断；PICP50 仅 0.351 也说明问题在分布宽度而非位置。
* Chronos-Bolt 的 PICP90 0.776 受原生 9 分位夹断影响（见 §7.2），不应据此判定其分布形状差；其 PICP50 0.540 是可信的。

### 11.3 点预测补充（MAE / RMSE / R²）

点预测口径：分位模型取 τ=0.5（线性插值），采样模型取样本中位数；在全部 clean station×window cell 上 pooled 计算。分位模型与采样模型共用同一测试窗与 clean mask，与 §11 的概率结果一一对应。

覆盖：CSDI 的全部城市为子集口径（LOA/JHB/MEL/SPO 256 窗、SZH 128 窗）；TSDiff、DiffSTG、DeepAR-ISQF/StudentT 仅 SZH 为子集（256 窗，DeepAR 因 `--predict_batch_size 512` 下 SZH 全量会 OOM）。其余方法为五城全量。表中 `口径` 列标出；raw MAE 的逐城三 horizon 表可由 `summarize_point.py` 直接复现（§9.3）。注意这里说的是**点预测覆盖**：概率主表中 DeepAR-ISQF 的 SZH 是全量 2125 窗（WIS 重评也是全量），所以 §11.4 不给它打子集标记。

跨城直接平均 raw MAE 会被 MEL/SZH 的大量级主导，故下表用 nMAE = MAE / mean(y)（同一 clean cells）与 R² 的五城平均做紧凑汇总；逐城比较请用原始 MAE。SZH 的 R²（≈0.997）由站间水平差主导，不能当作预测能力指标。

| # | 方法 | 口径 | nMAE H24 | nMAE H48 | nMAE H72 | R² H24 | R² H48 | R² H72 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 7 | iTransformer | 全量 | 0.4558 | 0.4934 | 0.5016 | 0.7077 | 0.6700 | 0.6625 |
| 8 | STGCN | 全量 | 0.4641 | 0.5129 | 0.5287 | 0.6891 | 0.6412 | 0.6261 |
| 9 | STGCN-nograph | 全量 | 0.4697 | 0.5158 | 0.5314 | 0.6793 | 0.6328 | 0.6176 |
| 10 | Graph WaveNet | 全量 | 0.4617 | 0.5094 | 0.5186 | 0.6785 | 0.6358 | 0.6298 |
| 11 | TFT | 全量 | 0.4712 | 0.5232 | 0.5437 | 0.6773 | 0.6315 | 0.6102 |
| 12 | DeepAR-ISQF | 4 城 + SZH* | 0.5370* | 0.5785* | 0.5819* | 0.6406* | 0.5893* | 0.5902* |
| 12b | DeepAR-StudentT | 4 城 + SZH* | 0.7813* | 0.9345* | 0.9166* | 0.1861* | -0.1864* | -0.0886* |
| 13 | MQ-CNN | 全量 | 0.4499 | 0.4865 | 0.4963 | 0.6942 | 0.6619 | 0.6537 |
| 14 | CSDI（官方） | 子集* | 0.5269* | 0.5579* | 0.5568* | 0.6092* | 0.5977* | 0.5836* |
| 15 | TimeGrad | 全量 | 0.4900 | 0.5489 | 0.5638 | 0.6527 | 0.5981 | 0.6012 |
| 16 | TSDiff | 4 城 + SZH* | 0.4916* | 0.5403* | 0.5584* | 0.6647* | 0.6145* | 0.6049* |
| 17 | DiffSTG | 4 城 + SZH* | 0.7903* | 0.8049* | 0.7024* | 0.3489* | 0.3248* | 0.4846* |
| 18 | Chronos-Bolt | 全量 | 0.4607 | 0.5228 | 0.5312 | 0.6498 | 0.5858 | 0.5903 |
| 19 | Moirai | 全量 | 0.4898 | 0.5718 | 0.5875 | 0.6170 | 0.4981 | 0.5045 |
| 6 | PatchTST+quant-pre72 | 全量 | 0.4515 | 0.4900 | 0.4988 | 0.6921 | 0.6601 | 0.6495 |
| 20 | UniTS-FT+pool | 全量 | 0.4958 | 0.5265 | 0.5279 | 0.6938 | 0.6655 | 0.6616 |
| 21 | OpenCity-Plus | 全量 | 0.4627 | 0.4967 | 0.4997 | 0.6817 | 0.6506 | 0.6552 |

`*` = 该行的五城平均包含子集城（CSDI 全城；DeepAR-ISQF/StudentT、TSDiff、DiffSTG 的 SZH 为 256 窗），不与全量行直接比较。表中另含点预测参考行 12b（DeepAR-StudentT），它不进概率主表。

**raw MAE 不能跨城平均（覆盖必须一致）**：TimeGrad 只算 LOA/JHB/MEL/SPO 时 raw MAE 均值 9.53，补上 SZH（34.67）后为 14.56；CSDI 四城 10.60 → 五城 15.29；DiffSTG 四城 12.56 → 五城 26.18；MQ-CNN 五城 12.11。若按四城平均，TimeGrad/CSDI 会凭空优于 MQ-CNN，纯属缺 SZH 的假象。本表因此只报 nMAE（按同一城 `mean(y)` 归一化）与 R²，不报 raw MAE 的五城均值。

#### 逐城明细（MAE / RMSE / R² / nMAE）

每格 = `MAE / RMSE / R² / nMAE`；MAE/RMSE 为原始 kWh/h，R² 与 nMAE 为同一城 clean cells 上的值。末两列为五城简单平均。`*` = 该城为子集窗；raw MAE/RMSE 不跨城平均。

##### H24

| # | 方法 | 口径 | LOA | JHB | MEL | SPO | SZH | nMAE 五城 | R² 五城 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 13 | MQ-CNN | 全量 | 2.88 / 5.09 / 0.877 / 0.396 | 4.29 / 8.88 / 0.400 / 0.589 | 22.94 / 88.20 / 0.701 / 0.530 | 3.85 / 7.47 / 0.496 / 0.620 | 26.58 / 149.51 / 0.997 / 0.113 | 0.4499 | 0.6942 |
| 6 | PatchTST+quant-pre72 | 全量 | 2.87 / 5.09 / 0.877 / 0.394 | 4.28 / 8.87 / 0.401 / 0.588 | 23.03 / 88.52 / 0.699 / 0.532 | 3.88 / 7.54 / 0.487 / 0.625 | 27.73 / 156.55 / 0.997 / 0.118 | 0.4515 | 0.6921 |
| 7 | iTransformer | 全量 | 2.85 / 4.95 / 0.884 / 0.391 | 4.44 / 8.63 / 0.433 / 0.609 | 23.06 / 86.53 / 0.712 / 0.533 | 3.94 / 7.35 / 0.512 / 0.635 | 26.01 / 146.24 / 0.997 / 0.111 | 0.4558 | 0.7077 |
| 18 | Chronos-Bolt | 全量 | 3.03 / 5.52 / 0.856 / 0.416 | 4.29 / 9.58 / 0.301 / 0.588 | 23.39 / 91.91 / 0.675 / 0.541 | 3.97 / 8.01 / 0.420 / 0.639 | 28.01 / 164.08 / 0.997 / 0.119 | 0.4607 | 0.6498 |
| 10 | Graph WaveNet | 全量 | 2.90 / 5.18 / 0.873 / 0.398 | 4.28 / 8.78 / 0.413 / 0.588 | 24.12 / 93.54 / 0.663 / 0.557 | 4.00 / 7.83 / 0.446 / 0.645 | 28.25 / 161.70 / 0.997 / 0.121 | 0.4617 | 0.6785 |
| 21 | OpenCity-Plus | 全量 | 2.88 / 5.02 / 0.881 / 0.396 | 4.49 / 9.16 / 0.361 / 0.616 | 23.53 / 90.22 / 0.687 / 0.544 | 3.98 / 7.57 / 0.483 / 0.641 | 27.31 / 156.02 / 0.997 / 0.117 | 0.4627 | 0.6817 |
| 8 | STGCN | 全量 | 3.02 / 5.44 / 0.860 / 0.414 | 4.29 / 8.71 / 0.423 / 0.589 | 24.10 / 92.36 / 0.672 / 0.557 | 3.94 / 7.48 / 0.494 / 0.635 | 29.36 / 169.75 / 0.997 / 0.125 | 0.4641 | 0.6891 |
| 9 | STGCN-nograph | 全量 | 3.06 / 5.57 / 0.853 / 0.420 | 4.28 / 8.70 / 0.424 / 0.588 | 24.88 / 96.55 / 0.642 / 0.575 | 3.98 / 7.58 / 0.481 / 0.642 | 28.86 / 166.71 / 0.997 / 0.123 | 0.4697 | 0.6793 |
| 11 | TFT | 全量 | 2.79 / 4.87 / 0.888 / 0.383 | 4.46 / 9.04 / 0.377 / 0.612 | 22.77 / 85.39 / 0.720 / 0.526 | 4.36 / 8.11 / 0.406 / 0.703 | 30.84 / 181.97 / 0.996 / 0.132 | 0.4712 | 0.6773 |
| 19 | Moirai | 全量 | 3.15 / 5.71 / 0.846 / 0.432 | 4.48 / 10.18 / 0.210 / 0.614 | 24.69 / 94.43 / 0.657 / 0.571 | 4.20 / 8.31 / 0.377 / 0.678 | 36.15 / 209.57 / 0.995 / 0.154 | 0.4898 | 0.6170 |
| 15 | TimeGrad | 全量 | 3.00 / 5.48 / 0.858 / 0.413 | 4.32 / 8.66 / 0.429 / 0.592 | 26.56 / 103.68 / 0.587 / 0.614 | 4.24 / 8.18 / 0.395 / 0.683 | 34.67 / 204.23 / 0.995 / 0.148 | 0.4900 | 0.6527 |
| 20 | UniTS-FT+pool | 全量 | 3.12 / 5.44 / 0.860 / 0.429 | 4.75 / 8.73 / 0.420 / 0.652 | 23.66 / 87.41 / 0.706 / 0.547 | 4.28 / 7.52 / 0.489 / 0.691 | 37.62 / 223.20 / 0.994 / 0.160 | 0.4958 | 0.6938 |
| 16 | TSDiff | 4 城 + SZH* | 3.19 / 5.50 / 0.856 / 0.439 | 4.44 / 9.00 / 0.383 / 0.610 | 23.47 / 91.55 / 0.678 / 0.543 | 4.61 / 8.09 / 0.409 / 0.744 | 28.92 / 165.16 / 0.997 / 0.123* | 0.4916* | 0.6647* |
| 14 | CSDI（官方） | 子集* | 3.40 / 5.62 / 0.851 / 0.467* | 4.84 / 10.38 / 0.203 / 0.664* | 30.05 / 116.63 / 0.514 / 0.695* | 4.12 / 7.50 / 0.483 / 0.664* | 34.03 / 196.33 / 0.995 / 0.145* | 0.5269* | 0.6092* |
| 12 | DeepAR-ISQF | 4 城 + SZH* | 3.37 / 5.89 / 0.835 / 0.463 | 5.08 / 9.64 / 0.293 / 0.697 | 26.95 / 95.92 / 0.646 / 0.623 | 4.59 / 7.93 / 0.433 / 0.740 | 38.12 / 189.87 / 0.996 / 0.163* | 0.5370* | 0.6406* |
| 17 | DiffSTG | 4 城 + SZH* | 10.87 / 16.61 / -0.308 / 1.493 | 5.05 / 10.58 / 0.147 / 0.693 | 29.78 / 112.62 / 0.512 / 0.688 | 4.55 / 7.98 / 0.424 / 0.733 | 80.64 / 520.77 / 0.968 / 0.344* | 0.7903* | 0.3489* |

##### H48

| # | 方法 | 口径 | LOA | JHB | MEL | SPO | SZH | nMAE 五城 | R² 五城 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 13 | MQ-CNN | 全量 | 3.11 / 5.44 / 0.860 / 0.425 | 4.92 / 9.69 / 0.301 / 0.660 | 23.96 / 89.69 / 0.687 / 0.552 | 4.16 / 7.68 / 0.465 / 0.675 | 28.08 / 157.43 / 0.997 / 0.120 | 0.4865 | 0.6619 |
| 6 | PatchTST+quant-pre72 | 全量 | 3.10 / 5.44 / 0.860 / 0.425 | 4.90 / 9.50 / 0.327 / 0.657 | 24.02 / 89.93 / 0.685 / 0.554 | 4.25 / 7.92 / 0.431 / 0.690 | 29.01 / 163.53 / 0.997 / 0.124 | 0.4900 | 0.6601 |
| 7 | iTransformer | 全量 | 3.03 / 5.23 / 0.871 / 0.414 | 5.20 / 9.58 / 0.317 / 0.698 | 24.00 / 88.08 / 0.698 / 0.553 | 4.22 / 7.66 / 0.467 / 0.685 | 27.39 / 153.94 / 0.997 / 0.117 | 0.4934 | 0.6700 |
| 18 | Chronos-Bolt | 全量 | 3.37 / 6.13 / 0.823 / 0.462 | 5.29 / 10.89 / 0.117 / 0.711 | 24.43 / 91.60 / 0.673 / 0.563 | 4.63 / 8.66 / 0.319 / 0.752 | 29.56 / 168.09 / 0.997 / 0.126 | 0.5228 | 0.5858 |
| 10 | Graph WaveNet | 全量 | 3.16 / 5.57 / 0.854 / 0.432 | 4.96 / 9.56 / 0.320 / 0.666 | 25.86 / 99.09 / 0.617 / 0.596 | 4.46 / 8.19 / 0.392 / 0.725 | 29.94 / 170.28 / 0.997 / 0.128 | 0.5094 | 0.6358 |
| 21 | OpenCity-Plus | 全量 | 3.10 / 5.36 / 0.864 / 0.424 | 5.07 / 9.84 / 0.280 / 0.680 | 24.49 / 91.04 / 0.677 / 0.564 | 4.27 / 7.89 / 0.435 / 0.693 | 28.40 / 162.67 / 0.997 / 0.121 | 0.4967 | 0.6506 |
| 8 | STGCN | 全量 | 3.38 / 6.06 / 0.826 / 0.462 | 4.95 / 9.54 / 0.322 / 0.664 | 25.60 / 96.73 / 0.636 / 0.590 | 4.39 / 7.95 / 0.426 / 0.713 | 31.50 / 179.87 / 0.996 / 0.134 | 0.5129 | 0.6412 |
| 9 | STGCN-nograph | 全量 | 3.43 / 6.17 / 0.820 / 0.469 | 4.90 / 9.56 / 0.319 / 0.658 | 26.08 / 99.20 / 0.617 / 0.601 | 4.43 / 8.05 / 0.412 / 0.719 | 31.00 / 176.22 / 0.996 / 0.132 | 0.5158 | 0.6328 |
| 11 | TFT | 全量 | 2.97 / 5.13 / 0.875 / 0.407 | 5.27 / 9.81 / 0.283 / 0.707 | 23.56 / 85.65 / 0.714 / 0.543 | 5.07 / 8.82 / 0.289 / 0.823 | 31.89 / 188.46 / 0.996 / 0.136 | 0.5232 | 0.6315 |
| 19 | Moirai | 全量 | 3.63 / 6.63 / 0.792 / 0.496 | 5.49 / 12.17 / -0.104 / 0.737 | 27.06 / 101.87 / 0.596 / 0.624 | 5.06 / 9.31 / 0.214 / 0.821 | 42.29 / 264.17 / 0.992 / 0.181 | 0.5718 | 0.4981 |
| 15 | TimeGrad | 全量 | 3.65 / 6.59 / 0.795 / 0.499 | 4.91 / 9.44 / 0.337 / 0.659 | 29.01 / 111.18 / 0.519 / 0.669 | 4.48 / 8.47 / 0.349 / 0.728 | 44.61 / 270.06 / 0.991 / 0.190 | 0.5489 | 0.5981 |
| 20 | UniTS-FT+pool | 全量 | 3.32 / 5.83 / 0.839 / 0.454 | 5.19 / 9.26 / 0.362 / 0.696 | 25.50 / 92.55 / 0.666 / 0.588 | 4.49 / 7.67 / 0.466 / 0.729 | 38.57 / 228.17 / 0.994 / 0.165 | 0.5265 | 0.6655 |
| 16 | TSDiff | 4 城 + SZH* | 3.52 / 6.01 / 0.830 / 0.481 | 5.13 / 9.95 / 0.263 / 0.689 | 24.56 / 93.03 / 0.663 / 0.566 | 5.12 / 8.66 / 0.320 / 0.831 | 31.35 / 178.50 / 0.996 / 0.134* | 0.5403* | 0.6145* |
| 14 | CSDI（官方） | 子集* | 3.65 / 5.96 / 0.832 / 0.499* | 5.17 / 10.02 / 0.205 / 0.694* | 30.05 / 101.88 / 0.611 / 0.693* | 4.59 / 8.56 / 0.345 / 0.745* | 37.19 / 213.33 / 0.995 / 0.159* | 0.5579* | 0.5977* |
| 12 | DeepAR-ISQF | 4 城 + SZH* | 3.66 / 6.32 / 0.811 / 0.501 | 5.65 / 10.61 / 0.162 / 0.759 | 28.49 / 99.98 / 0.611 / 0.657 | 4.84 / 8.33 / 0.370 / 0.786 | 44.48 / 246.06 / 0.993 / 0.190* | 0.5785* | 0.5893* |
| 17 | DiffSTG | 4 城 + SZH* | 10.90 / 16.73 / -0.323 / 1.492 | 5.34 / 11.08 / 0.086 / 0.717 | 30.53 / 113.87 / 0.495 / 0.704 | 4.72 / 8.15 / 0.397 / 0.767 | 80.86 / 521.31 / 0.968 / 0.345* | 0.8049* | 0.3248* |

##### H72

| # | 方法 | 口径 | LOA | JHB | MEL | SPO | SZH | nMAE 五城 | R² 五城 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 13 | MQ-CNN | 全量 | 3.19 / 5.60 / 0.852 / 0.436 | 5.00 / 9.87 / 0.292 / 0.667 | 24.88 / 92.15 / 0.672 / 0.571 | 4.21 / 7.74 / 0.456 / 0.685 | 28.58 / 160.85 / 0.997 / 0.122 | 0.4963 | 0.6537 |
| 6 | PatchTST+quant-pre72 | 全量 | 3.18 / 5.60 / 0.852 / 0.435 | 5.00 / 9.83 / 0.298 / 0.667 | 24.76 / 91.49 / 0.676 / 0.569 | 4.30 / 7.97 / 0.424 / 0.699 | 29.42 / 165.76 / 0.997 / 0.126 | 0.4988 | 0.6495 |
| 7 | iTransformer | 全量 | 3.10 / 5.36 / 0.865 / 0.423 | 5.21 / 9.68 / 0.319 / 0.695 | 24.90 / 90.52 / 0.683 / 0.572 | 4.30 / 7.80 / 0.448 / 0.699 | 27.78 / 156.64 / 0.997 / 0.119 | 0.5016 | 0.6625 |
| 18 | Chronos-Bolt | 全量 | 3.32 / 5.95 / 0.833 / 0.454 | 5.44 / 10.79 / 0.155 / 0.726 | 25.26 / 93.15 / 0.664 / 0.580 | 4.71 / 8.77 / 0.303 / 0.766 | 30.35 / 173.15 / 0.996 / 0.130 | 0.5312 | 0.5903 |
| 10 | Graph WaveNet | 全量 | 3.27 / 5.76 / 0.843 / 0.447 | 5.11 / 9.84 / 0.297 / 0.682 | 26.41 / 99.72 / 0.615 / 0.607 | 4.47 / 8.15 / 0.397 / 0.727 | 30.57 / 174.63 / 0.996 / 0.131 | 0.5186 | 0.6298 |
| 21 | OpenCity-Plus | 全量 | 3.15 / 5.45 / 0.860 / 0.431 | 5.03 / 9.77 / 0.307 / 0.672 | 25.29 / 92.88 / 0.666 / 0.581 | 4.26 / 7.81 / 0.446 / 0.692 | 28.88 / 164.42 / 0.997 / 0.123 | 0.4997 | 0.6552 |
| 8 | STGCN | 全量 | 3.49 / 6.25 / 0.816 / 0.477 | 5.13 / 9.88 / 0.291 / 0.685 | 26.35 / 98.40 / 0.626 / 0.605 | 4.54 / 8.12 / 0.402 / 0.739 | 32.34 / 184.81 / 0.996 / 0.138 | 0.5287 | 0.6261 |
| 9 | STGCN-nograph | 全量 | 3.53 / 6.32 / 0.812 / 0.482 | 5.11 / 9.96 / 0.280 / 0.682 | 26.82 / 101.16 / 0.604 / 0.616 | 4.55 / 8.16 / 0.396 / 0.741 | 31.93 / 182.51 / 0.996 / 0.136 | 0.5314 | 0.6176 |
| 11 | TFT | 全量 | 2.99 / 5.15 / 0.875 / 0.408 | 5.44 / 9.97 / 0.277 / 0.726 | 24.20 / 86.49 / 0.711 / 0.556 | 5.47 / 9.34 / 0.192 / 0.890 | 32.32 / 190.43 / 0.996 / 0.138 | 0.5437 | 0.6102 |
| 19 | Moirai | 全量 | 3.71 / 6.81 / 0.781 / 0.507 | 5.77 / 11.95 / -0.037 / 0.769 | 28.19 / 104.16 / 0.581 / 0.647 | 5.09 / 9.35 / 0.207 / 0.828 | 43.52 / 271.37 / 0.991 / 0.186 | 0.5875 | 0.5045 |
| 15 | TimeGrad | 全量 | 3.78 / 6.74 / 0.786 / 0.516 | 5.08 / 9.61 / 0.329 / 0.678 | 29.47 / 110.14 / 0.531 / 0.677 | 4.56 / 8.33 / 0.370 / 0.741 | 48.37 / 290.50 / 0.990 / 0.207 | 0.5638 | 0.6012 |
| 20 | UniTS-FT+pool | 全量 | 3.34 / 5.81 / 0.841 / 0.457 | 5.20 / 9.54 / 0.339 / 0.694 | 25.44 / 90.99 / 0.680 / 0.584 | 4.55 / 7.75 / 0.455 / 0.740 | 38.56 / 227.64 / 0.994 / 0.165 | 0.5279 | 0.6616 |
| 16 | TSDiff | 4 城 + SZH* | 3.59 / 6.21 / 0.818 / 0.490 | 5.41 / 10.27 / 0.233 / 0.722 | 27.08 / 100.14 / 0.612 / 0.622 | 4.95 / 8.36 / 0.366 / 0.804 | 36.08 / 210.78 / 0.995 / 0.154* | 0.5584* | 0.6049* |
| 14 | CSDI（官方） | 子集* | 3.60 / 5.99 / 0.831 / 0.491* | 5.44 / 11.10 / 0.088 / 0.725* | 29.19 / 101.84 / 0.615 / 0.670* | 4.51 / 8.27 / 0.390 / 0.733* | 38.32 / 222.30 / 0.994 / 0.164* | 0.5568* | 0.5836* |
| 12 | DeepAR-ISQF | 4 城 + SZH* | 3.72 / 6.39 / 0.807 / 0.508 | 5.71 / 10.68 / 0.171 / 0.762 | 29.33 / 102.07 / 0.597 / 0.674 | 4.78 / 8.23 / 0.385 / 0.778 | 43.97 / 283.12 / 0.991 / 0.188* | 0.5819* | 0.5902* |
| 17 | DiffSTG | 4 城 + SZH* | 7.89 / 12.46 / 0.267 / 1.079 | 5.20 / 10.24 / 0.238 / 0.694 | 29.86 / 111.99 / 0.515 / 0.686 | 4.77 / 7.96 / 0.425 / 0.775 | 65.09 / 432.85 / 0.978 / 0.278* | 0.7024* | 0.4846* |

要点：

* 点预测第一梯队（H24 nMAE 0.450–0.462）为 MQ-CNN 0.4499、PatchTST 0.4515、iTransformer 0.4558、Chronos-Bolt 0.4607、GWNet 0.4617、OpenCity 0.4627、STGCN 0.4641，差距小于其在概率主表上的差距；概率指标的区分度主要来自分布形状，而不是中位数位置。
* **TFT 的 `crps_nz` 优势不是点预测带来的**：其 H24/H72 nMAE 0.4712/0.5437 均为第一梯队之外，随 horizon 退化也最快（+15.4%）；结合 §11.4 的 WIS_nz，TFT 的正值格收益更可能来自区间/尾部建模而非中位数。
* 扩散类的样本中位数普遍偏弱：TimeGrad 0.4900、TSDiff 0.4916、CSDI 0.5269（子集）；DiffSTG 因 LOA 中位数坍缩（R² −0.31，nMAE 被拉高）不可作为有效点预测，H72 反而「好转」只是坍缩程度变化，不是预测能力。
* DeepAR-ISQF 的点预测显著弱于其概率表现（0.5370），StudentT 版本更差（0.7813，JHB/SPO 的 R² 为负）；GluonTS 的逐站单变量路线在本数据上不适合直接取中位数。
* 随 horizon 退化：第一梯队 H24→H72 的 nMAE 增加约 9–11%（如 iTransformer +10.0%、MQ-CNN +10.3%），TFT +15.4%、Chronos-Bolt +15.3%、Moirai +19.9%；点预测的 horizon 稳定性与概率指标不完全同序。

### 11.4 概率主表汇总（crpsΣ / crps_nz / WIS / 校准）

**口径**：与 §11.A/11.B 同一 clean mask、`seq_l=pre_len=72`；H24/48/72 取自同一 72 步输出。五城平均 = 先逐城对 seed 取均值、再对可用城市简单平均；完整逐城 mean±std 见 §11.A/11.B，本节不重复。

**WIS**：`WIS = [0.5|m−y| + Σ_k (α_k/2)((u_k−l_k) + (2/α_k)(l_k−y)_+ + (2/α_k)(y−u_k)_+)] / 5.5`，`α_k ∈ {0.02, 0.05, 0.10, 0.20, 0.50}`，由 `StreamingMetrics` 在评测时直接累计，**不是**从 `diag/*.npz` 反算。WIS 已按 `mean(smean)` 归一化，与 crpsΣ 同量纲（无量纲），因此可以跨城平均；**MPIW 仍是原始 kWh/h，只在同城列内比较**，不做五城平均。样本数：分位类 200（99 分位展开为分层样本），扩散 100，官方 CSDI 50；WIS 含 1%/99% 尾部，CSDI 的 50 样本尾部与 Chronos-Bolt 的 9 个原生分位（见 §7.2）噪声更大。

**子集标记**：`*` = 该行/该城为子集口径。CSDI 全城为等间隔子集（LOA/JHB/MEL/SPO 256 窗、SZH 128 窗）；TSDiff、DiffSTG 的 SZH 为 256 窗子集。**DeepAR-ISQF 的概率评测 SZH 是全量 2125 窗，故本节不打 `*`；它的 `*` 只出现在 §11.3 的点预测表。** 表注：**子集评测，256 窗；843 站全量采样约 4 天**（CSDI-SZH 为 128 窗，是例外）。

**‡ 采样不稳定**：DiffSTG-LOA 的同一 ckpt 在两次独立 100 样本采样间差异极大——主评测（`sample_chunk=16`）crpsΣ 0.3931 / PICP90 0.677 / MPIW90 6.4，WIS 重评（`sample_chunk=8`）crpsΣ 0.9913 / PICP90 0.274 / MPIW90 25.2，WIS 0.6448。二者仅 RNG 流不同，不是权重变化；说明 LOA 的 100 样本分布极不稳定。表中 crpsΣ/crps_nz/PICP/MPIW 用主评测，WIS 用重评，**该行任何跨指标比较都不成立**；DiffSTG 的 WIS 五城平均 0.3292‡ 含 LOA，不含 LOA 的四城平均为 0.2503。DiffSTG 行不参与排序解释。

#### 11.4a 精度：crpsΣ / crps_nz / WIS（五城平均）

| # | 方法 | 口径 | crpsΣ H24 | crpsΣ H48 | crpsΣ H72 | crps_nz H24 | crps_nz H48 | crps_nz H72 | WIS H24 | WIS H48 | WIS H72 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 13 | MQ-CNN | 全量 | 0.2842 | 0.3128 | 0.3255 | 0.3787 | 0.4162 | 0.4304 | 0.1714 | 0.1902 | 0.2001 |
| 6 | PatchTST+quant-pre72 | 全量 | 0.2848 | 0.3128 | 0.3238 | 0.3822 | 0.4187 | 0.4324 | 0.1704 | 0.1886 | 0.1966 |
| 7 | iTransformer | 全量 | 0.2891 | 0.3167 | 0.3301 | 0.3770 | 0.4092 | 0.4248 | 0.1773 | 0.1963 | 0.2065 |
| 21 | OpenCity-Plus | 全量 | 0.2920 | 0.3191 | 0.3288 | 0.3890 | 0.4255 | 0.4350 | 0.1763 | 0.1948 | 0.2026 |
| 10 | Graph WaveNet | 全量 | 0.2976 | 0.3313 | 0.3442 | 0.3944 | 0.4333 | 0.4444 | 0.1828 | 0.2039 | 0.2142 |
| 11 | TFT | 全量 | 0.2987 | 0.3342 | 0.3517 | 0.3619 | 0.3988 | 0.4146 | 0.1792 | 0.2005 | 0.2115 |
| 20 | UniTS-FT+pool | 全量 | 0.3018 | 0.3218 | 0.3334 | 0.3838 | 0.4075 | 0.4167 | 0.1924 | 0.2037 | 0.2113 |
| 8 | STGCN | 全量 | 0.3027 | 0.3355 | 0.3511 | 0.3911 | 0.4262 | 0.4431 | 0.1881 | 0.2087 | 0.2200 |
| 18 | Chronos-Bolt | 全量 | 0.3055 | 0.3511 | 0.3773 | 0.3925 | 0.4433 | 0.4763 | 0.2068 | 0.2360 | 0.2653 |
| 9 | STGCN-nograph | 全量 | 0.3056 | 0.3374 | 0.3524 | 0.3941 | 0.4307 | 0.4470 | 0.1892 | 0.2092 | 0.2197 |
| 16 | TSDiff | 4 城 + SZH* | 0.3093* | 0.3419* | 0.3603* | 0.3702* | 0.4062* | 0.4301* | 0.1848* | 0.2058* | 0.2204* |
| 15 | TimeGrad | 全量 | 0.3095 | 0.3494 | 0.3646 | 0.4181 | 0.4619 | 0.4677 | 0.1856 | 0.2115 | 0.2237 |
| 19 | Moirai | 全量 | 0.3199 | 0.3771 | 0.3907 | 0.4032 | 0.4770 | 0.4931 | 0.2085 | 0.2480 | 0.2496 |
| 14 | CSDI（官方） | 子集* | 0.3356* | 0.3633* | 0.3620* | 0.4388* | 0.4789* | 0.4801* | 0.2069* | 0.2225* | 0.2200* |
| 12 | DeepAR-ISQF | 全量 | 0.3614 | 0.4021 | 0.4104 | 0.4960 | 0.5552 | 0.5616 | 0.2181 | 0.2474 | 0.2551 |
| 17 | DiffSTG | 4 城 + SZH* | 0.3973* | 0.4139* | 0.4011* | 0.5107* | 0.5311* | 0.4891* | 0.3292*‡ | 0.3423*‡ | 0.2941*‡ |

#### 11.4b 校准：PICP90（五城平均）与 MPIW90（逐城，H=24）

MPIW90 单位原始 kWh/h；`*` 为子集城。PICP90 为五个可用城市的简单平均，子集行同样标 `*`，不与全量行直接比较。

| # | 方法 | 口径 | PICP90 | MPIW90 LOA | MPIW90 JHB | MPIW90 MEL | MPIW90 SPO | MPIW90 SZH |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 13 | MQ-CNN | 全量 | 0.893 | 11.0 | 18.0 | 92.3 | 16.2 | 117.6 |
| 6 | PatchTST+quant-pre72 | 全量 | 0.885 | 10.7 | 17.9 | 87.7 | 15.4 | 112.8 |
| 7 | iTransformer | 全量 | 0.883 | 11.7 | 20.1 | 92.3 | 16.0 | 113.3 |
| 21 | OpenCity-Plus | 全量 | 0.882 | 11.1 | 18.3 | 89.4 | 15.4 | 108.6 |
| 10 | Graph WaveNet | 全量 | 0.880 | 11.0 | 21.6 | 98.8 | 16.6 | 132.7 |
| 11 | TFT | 全量 | 0.868 | 10.7 | 18.9 | 88.9 | 14.9 | 132.4 |
| 20 | UniTS-FT+pool | 全量 | 0.801 | 9.8 | 18.3 | 87.6 | 14.6 | 102.2 |
| 8 | STGCN | 全量 | 0.849 | 11.8 | 21.2 | 101.6 | 15.9 | 130.6 |
| 18 | Chronos-Bolt | 全量 | 0.776 | 8.2 | 10.7 | 71.8 | 11.1 | 101.2 |
| 9 | STGCN-nograph | 全量 | 0.850 | 12.3 | 20.1 | 103.1 | 16.0 | 126.1 |
| 16 | TSDiff | 4 城 + SZH* | 0.857* | 12.3 | 16.7 | 89.4 | 14.5 | 154.8* |
| 15 | TimeGrad | 全量 | 0.780 | 10.0 | 17.1 | 90.1 | 12.5 | 144.3 |
| 19 | Moirai | 全量 | 0.852 | 12.4 | 18.2 | 116.0 | 17.1 | 212.3 |
| 14 | CSDI（官方） | 子集* | 0.590* | 8.1* | 11.8* | 78.5* | 16.4* | 99.5* |
| 12 | DeepAR-ISQF | 全量 | 0.809 | 14.0 | 19.2 | 65.9 | 13.4 | 275.6 |
| 17 | DiffSTG | 4 城 + SZH* | 0.729* | 6.4 | 19.6 | 92.7 | 18.0 | 156.2* |

#### 11.4c 扩散四方法：WIS / WIS_nz / MPIW50 逐城

每格 = `WIS / WIS_nz / MPIW50`；WIS、WIS_nz 已归一化，MPIW50 为原始 kWh/h。`*` = 子集；`‡` = DiffSTG-LOA 的 WIS 来自 chunk8 重评（与主评测 crpsΣ 非同一次采样，见上）。

| 方法 | H | LOA | JHB | MEL | SPO | SZH |
|---|---|---|---|---|---|---|
| CSDI（官方） | H24 | 0.1877 / 0.1900 / 3.42* | 0.2185 / 0.3279 / 4.06* | 0.2798 / 0.3342 / 42.67* | 0.2818 / 0.3935 / 8.15* | 0.0666 / 0.0759 / 42.18* |
| CSDI（官方） | H48 | 0.1925 / 0.1968 / 4.13* | 0.2540 / 0.3467 / 4.84* | 0.2812 / 0.2981 / 46.02* | 0.3120 / 0.4977 / 6.52* | 0.0727 / 0.0834 / 43.71* |
| CSDI（官方） | H72 | 0.1826 / 0.1917 / 4.56* | 0.2599 / 0.4050 / 3.47* | 0.2793 / 0.2975 / 43.31* | 0.3048 / 0.4451 / 8.92* | 0.0736 / 0.0825 / 49.92* |
| TimeGrad | H24 | 0.1489 / 0.1500 / 4.43 | 0.1834 / 0.2259 / 6.62 | 0.2300 / 0.2652 / 52.42 | 0.3052 / 0.5048 / 4.69 | 0.0604 / 0.0702 / 58.64 |
| TimeGrad | H48 | 0.1788 / 0.1753 / 5.13 | 0.2164 / 0.2558 / 6.93 | 0.2501 / 0.2839 / 53.26 | 0.3387 / 0.5495 / 4.77 | 0.0734 / 0.0861 / 73.62 |
| TimeGrad | H72 | 0.1874 / 0.1818 / 4.77 | 0.2394 / 0.2820 / 6.59 | 0.2643 / 0.2888 / 46.61 | 0.3497 / 0.5410 / 4.93 | 0.0777 / 0.0913 / 73.79 |
| TSDiff | H24 | 0.1567 / 0.1530 / 5.70 | 0.1888 / 0.2353 / 7.42 | 0.2052 / 0.2498 / 45.81 | 0.3195 / 0.3960 / 6.07 | 0.0538 / 0.0619 / 60.61* |
| TSDiff | H48 | 0.1702 / 0.1649 / 6.27 | 0.2204 / 0.2653 / 8.02 | 0.2204 / 0.2617 / 48.61 | 0.3602 / 0.4397 / 7.30 | 0.0579 / 0.0671 / 68.33* |
| TSDiff | H72 | 0.1743 / 0.1684 / 6.15 | 0.2437 / 0.2855 / 7.40 | 0.2413 / 0.2723 / 48.28 | 0.3801 / 0.4730 / 7.16 | 0.0629 / 0.0732 / 70.28* |
| DiffSTG | H24 | 0.6448 / 0.5643 / 8.83‡ | 0.1937 / 0.2803 / 7.46 | 0.2469 / 0.2653 / 51.12 | 0.4230 / 0.6024 / 3.76 | 0.1377 / 0.1664 / 68.50* |
| DiffSTG | H48 | 0.6508 / 0.5710 / 8.83‡ | 0.2165 / 0.3083 / 7.40 | 0.2592 / 0.2760 / 51.11 | 0.4468 / 0.6294 / 3.75 | 0.1384 / 0.1671 / 68.30* |
| DiffSTG | H72 | 0.4352 / 0.3751 / 7.85‡ | 0.2304 / 0.2968 / 9.10 | 0.2783 / 0.2862 / 42.24 | 0.4192 / 0.5429 / 3.56 | 0.1074 / 0.1290 / 86.91* |

**要点**：

* **crpsΣ 前三不变**：MQ-CNN 0.2842、PatchTST 0.2848、iTransformer 0.2891；最大差 0.0049，仍与 seed std 同量级。H48/H72 的排序与 H24 基本一致，PatchTST 在 H72 反超 MQ-CNN（0.3238 vs 0.3255）。
* **WIS 把 TFT 推回第一梯队，但没有稳定赢家**：H24 五城 WIS PatchTST 0.1704、MQ-CNN 0.1714、OpenCity 0.1763、iTransformer 0.1773、TFT 0.1792；前四差 ≤0.007；TFT 与 PatchTST 差 0.0088，小于 TFT 自身 SPO 的逐城 seed std（0.027），也小于 MQ-CNN 的 SPO seed std（0.0043）。PatchTST 与 MQ-CNN 只差 0.0010，与二者逐城 seed std（0.0002–0.002）同量级。WIS 排名与 crpsΣ 排名不完全同序，但前排差异都在 seed 噪声量级，不能据此重排。
* **TFT 的正值格优势是真的**：WIS_nz 五城平均 TFT 0.2131 / 0.2343 / 0.2431（H24/48/72），在三个 horizon 都是全场最低；PatchTST 0.2200 / 0.2405 / 0.2480，iTransformer 0.2231 / 0.2434 / 0.2533，MQ-CNN 0.2213 / 0.2419 / 0.2511。TFT 的边际 WIS 仍落后，是因为零格（§11.5：SPO 零格贡献 0.2080 vs iTransformer 0.1299），不是因为正值格区间更宽。
* **扩散类 WIS 没有翻盘**：全量五城的 TimeGrad H24 WIS 0.1856，已接近第一梯队；TSDiff 0.1848* 但 SZH 是子集；CSDI 0.2069*，且 PICP90 只有 0.590*；DiffSTG 0.3292*‡（不含 LOA 0.2503），仍最差。WIS 与 crpsΣ 对扩散类的判断一致。
* **校准**：分位头/图主干 PICP90 为 0.80–0.89（TFT 0.868、UniTS 0.801）；零样本 Moirai 0.852、Chronos-Bolt 0.776（后者受 9 分位夹断影响，不作形状结论）；TimeGrad 0.780、DiffSTG 0.729*、CSDI 0.590* 仍欠离散。MPIW90 逐城看，分位头类在 MEL/SZH 更宽、LOA/JHB/SPO 更窄，TFT 与 iTransformer 的方向仍逐城不一致，不能五城平均。
* **DeepAR-ISQF 的概率 SZH 是全量**（crpsΣ 0.3614 五城、SZH 0.1909），其 WIS 也按全量；点预测表的 SZH `*` 是另一回事。

### 11.5 T5 零格/正格分解（H=24）

分解是**测量的精确分解**，不是 oracle 投影：`crps_zero_contrib + crps_pos_contrib = crps_norm_sum`（同一批 clean cells；closure 校验最大误差 3.4e-7）。两项均按 `mean(smean)` 归一化。表中每格 = `零格贡献 / 正格贡献 (crpsΣ)`，括号内为本次评测的 crpsΣ（应等于两项之和）。

| # | 方法 | LOA | JHB | MEL | SPO | SZH |
|---|---|---:|---:|---:|---:|---:|
| 6 | PatchTST+quant-pre72 | 0.0309 / 0.2133 (0.2442) | 0.0716 / 0.2119 (0.2835) | 0.0741 / 0.2754 (0.3496) | 0.1134 / 0.3488 (0.4621) | 0.0032 / 0.0813 (0.0845) |
| 7 | iTransformer | 0.0337 / 0.2120 (0.2458) | 0.0922 / 0.1971 (0.2893) | 0.0824 / 0.2709 (0.3534) | 0.1299 / 0.3472 (0.4771) | 0.0031 / 0.0767 (0.0797) |
| 8 | STGCN | 0.0364 / 0.2275 (0.2639) | 0.0858 / 0.2089 (0.2947) | 0.0903 / 0.2831 (0.3734) | 0.1347 / 0.3548 (0.4895) | 0.0031 / 0.0889 (0.0920) |
| 9 | STGCN-nograph | 0.0364 / 0.2313 (0.2677) | 0.0867 / 0.2090 (0.2958) | 0.0922 / 0.2877 (0.3799) | 0.1363 / 0.3574 (0.4937) | 0.0031 / 0.0878 (0.0909) |
| 10 | Graph WaveNet | 0.0339 / 0.2167 (0.2505) | 0.0841 / 0.2079 (0.2920) | 0.0799 / 0.2909 (0.3708) | 0.1282 / 0.3595 (0.4876) | 0.0030 / 0.0840 (0.0870) |
| 11 | TFT | 0.0334 / 0.2047 (0.2381) | 0.0940 / 0.1952 (0.2892) | 0.0796 / 0.2689 (0.3484) | 0.2080 / 0.3168 (0.5247) | 0.0033 / 0.0899 (0.0931) |
| 12 | DeepAR-ISQF | 0.0349 / 0.2593 (0.2942) | 0.0445 / 0.2841 (0.3286) | 0.1283 / 0.3019 (0.4302) | 0.0894 / 0.4741 (0.5635) | 0.0047 / 0.1863 (0.1910) |
| 13 | MQ-CNN | 0.0325 / 0.2139 (0.2463) | 0.0735 / 0.2113 (0.2848) | 0.0737 / 0.2737 (0.3474) | 0.1194 / 0.3418 (0.4612) | 0.0031 / 0.0783 (0.0814) |
| 14 | CSDI（官方） | 0.0495 / 0.2467 (0.2962)* | 0.0678 / 0.2612 (0.3290)* | 0.0873 / 0.3679 (0.4552)* | 0.1424 / 0.3503 (0.4927)* | 0.0035 / 0.1014 (0.1050)* |
| 15 | TimeGrad | 0.0307 / 0.2237 (0.2544) | 0.0818 / 0.2035 (0.2853) | 0.0914 / 0.3066 (0.3980) | 0.0966 / 0.4103 (0.5069) | 0.0032 / 0.0995 (0.1027) |
| 16 | TSDiff | 0.0428 / 0.2271 (0.2700) | 0.0813 / 0.2124 (0.2938) | 0.0685 / 0.2837 (0.3522) | 0.2266 / 0.3142 (0.5408) | 0.0033 / 0.0856 (0.0889)* |
| 17 | DiffSTG | 0.2476 / 0.7437 (0.9913)‡ | 0.0524 / 0.2674 (0.3198) | 0.1209 / 0.3045 (0.4254) | 0.1909 / 0.4208 (0.6117) | 0.0008 / 0.2350 (0.2359)* |
| 18 | Chronos-Bolt | 0.0325 / 0.2318 (0.2643) | 0.0944 / 0.2033 (0.2977) | 0.0793 / 0.2807 (0.3600) | 0.1562 / 0.3627 (0.5189) | 0.0029 / 0.0838 (0.0868) |
| 19 | Moirai | 0.0378 / 0.2344 (0.2722) | 0.1010 / 0.2036 (0.3045) | 0.0942 / 0.2853 (0.3795) | 0.1651 / 0.3681 (0.5332) | 0.0028 / 0.1076 (0.1104) |
| 20 | UniTS-FT+pool | 0.0328 / 0.2142 (0.2470) | 0.0944 / 0.2224 (0.3168) | 0.0861 / 0.2824 (0.3685) | 0.1428 / 0.3531 (0.4959) | 0.0030 / 0.0780 (0.0810) |
| 21 | OpenCity-Plus | 0.0351 / 0.2115 (0.2466) | 0.0835 / 0.2092 (0.2927) | 0.0761 / 0.2831 (0.3592) | 0.1177 / 0.3609 (0.4786) | 0.0032 / 0.0797 (0.0829) |

**零格占比**（方法无关的测试集属性，全量口径）：LOA 21.34%、JHB 24.02%、MEL 30.87%、SPO 29.11%、SZH 2.97%。子集口径：TSDiff/DiffSTG 的 SZH 256 窗为 3.033%，CSDI 的 SZH 128 窗为 2.969%（与全量几乎相同）。

**要点**：

* **SZH 的 crpsΣ 几乎全是正格**（零格仅 ~3%），所以 SZH 上的差异不能再用「零格白送分」解释；LOA/JHB/MEL/SPO 的零格占 21–31%，方法在零格上的行为会显著改变边际 crpsΣ。
* **TFT vs iTransformer 的两面性被量化**：TFT 在 LOA/JHB/MEL 的正格贡献更低（0.2047/0.1952/0.2689 vs 0.2120/0.1971/0.2709），在 SZH 反而更高（0.0899 vs 0.0767）；边际 crpsΣ 落后主要来自 SPO 零格（0.2080 vs 0.1299），而 SPO 正格 TFT 更好（0.3168 vs 0.3472）。这与 WIS_nz 的结论一致。
* **TSDiff 的 SPO 塌缩在零格**：零格贡献 0.2266，是 SPO 上所有方法中最高；正格 0.3142 仍属中游。TSDiff 的边际 crpsΣ 0.5408 不能解读为「正值格预测差」。
* **DiffSTG 是正格失败，不是零格问题**：SZH 零格仅 0.0008、正格 0.2350；LOA 在 chunk8 重评下零格 0.2476、正格 0.7437（‡，与主评测非同一次采样）。即使只看正格，它也与第一梯队差一个档。
* **DeepAR-ISQF 的正格贡献在五城都偏高**（SZH 0.1863、SPO 0.4741），与它逐站单变量、大站集训练不足的结论一致；CSDI 的正格贡献也普遍高（MEL 0.3679、JHB 0.2612），说明官方 CSDI 的问题不只是零格。
* **closure 与重评差异**：括号内 crpsΣ 是本表所用评测的 crpsΣ。分位/图/零样本方法与 §11.A 主表逐位一致；扩散类因 WIS 重评而重采样，差异 ≤0.001（DiffSTG-LOA 除外），所以分解只对括号内的 crpsΣ 严格闭合，不保证与 §11.A 的独立采样逐位相同。
