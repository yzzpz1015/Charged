# 概率预测 Baseline 统一协议（#7–#21）

> 目的：在现有 #1–#6 之外补齐 15 个 baseline，全部使用**同一批测试窗、同一 clean mask、同一指标、同一 horizon 口径**，形成 21 方法横向对比表，并作为零膨胀方法的试验底座。
> 代码：`script/baselines/`；结果：`result/baselines/{metrics,diag,ckpt}/`。

## 1. 数据与划分

* 城市：LOA / JHB / MEL / SPO / SZH（AMS 单列，测试期 95.9% 站冻结，不并入主表）。
* 数据：`data/{city}_remove_zero/{city}_volume.csv`（`common.load_city` 自动追加 `_remove_zero`）。
* 划分：`EVDataset.split_cross_validation(fold=6, total_fold=6)` —— 先丢弃停摆小时（>80% 站为 0），可用行按计数 **2/1/3** 切 train/valid/test（不是 80/10/10）。
* 站过滤：只用**训练期**统计量剔除退化站（全零/常量/覆盖率不足），无泄漏。
  * 窗/站数自检：LOA 218、JHB 29、MEL 56、SPO 33、SZH 843。
  * 测试窗：LOA/JHB/SPO/SZH 2125，MEL 1781（MEL 训练期少 7 行）。
* 五城均为完整小时网格（T=4392，2023-04-01 00:00 – 2023-09-30 23:00，无缺小时）。
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
| `mae`, `rmse`, `r2` | **点预测指标**：该 horizon 的预测点（分位方法取 τ=0.5，样本方法取样本中位数）在 clean 站×窗上 pool 后计算 |
| `n_point_cells` | 点指标实际参与的有效 站×窗 数（= clean 站×窗，每个 horizon 相同） |
| `n_scored_windows` | 实际参与指标的有效窗数（排除全不 clean 的窗） |

* CRPS 用 fair CRPS（`crps_fair`）。
* 分位方法统一转成 200 条样本再算 CRPS（`M_SAMPLE=200`，与 #1–#6 一致；`strat_u` 分层均匀采样 + `quant_to_samples_ind`，转前先 `np.sort(quantiles)`）。
* 样本方法（DeepAR / Moirai / CSDI）使用模型自身样本：DeepAR 200、Moirai 200、官方 CSDI 50（受采样成本限制，见 §7）。
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
| 11 | TFT | `train_gluonts.py --method tft` | GluonTS 0.14.4，逐站单变量，直接训练 99 分位；`hidden=64, heads=4, layers=2` |
| 12 | DeepAR | `train_gluonts.py --method deepar` | GluonTS StudentT 似然，`hidden=64, layers=2`，`num_parallel_samples=200`，非负样本 |
| 13 | MQ-CNN | `train_quant_backbone.py --method mqcnn` | GluonTS torch 无官方实现，按原结构重写：因果空洞 conv（dilation 1–16）+ 共用分位头；**无 static station embedding**（与其它 baseline 同输入） |
| 14 | 官方 CSDI | `csdi_official.py` | `ermongroup/CSDI` 的 `CSDI_Forecasting`，模型/损失/扩散过程不改，仅替换数据适配层 |
| 15 | TimeGrad | `train_timegrad.py --method timegrad` | GRU 历史编码 + FiLM 条件 DDPM；**非自回归**（72 步联合扩散，原版逐步自回归），DDIM 50 步、η=1 |
| 16 | TSDiff | `train_tsdiff.py` | 整段 [72 历史+72 未来] 联合扩散，无 RNN；采样时把历史段硬投影回真值（hard self-guidance，替代原版软梯度引导）；DDIM 50 步、η=1 |
| 17 | DiffSTG | `train_diffstg.py` | 时空图扩散：[N,72] 联合扩散，每 block「时间膨胀 conv + kNN 稀疏图聚合 + 时间 FiLM + **历史 FiLM**」；历史条件用 GRU 末状态，并在每个 block 再做一次 FiLM（v3）。v1 的 `cond_enc(...).mean(dim=3)` 在 RevIN 后≈0；v2 只在输入层加一次条件，ε 学不动。v1/v2 产物已归档且不进主表。轻量版 UGnet；DDIM 50 步、η=1 |
| 18 | Chronos-Bolt | `run_zeroshot.py --method chronos_bolt` | 零样本，`chronos-bolt-small`，原生 9 分位 |
| 19 | Moirai | `run_zeroshot.py --method moirai` | 零样本，`moirai-1.0-R-small`，200 条样本→99 经验分位 |
| 20 | UniTS-FT + pool | `train_unist.py` | 公开权重 `units_x64_supervised_checkpoint.pth`（d_model=64）作主干；每城新建 dataset prompt、其余权重迁移，用 pre72 窗 + clean mask 少样本微调（30 epoch，patience 6，lr 1e-4，80 步/epoch）；概率输出走与 #1/#5 相同的残差池（roll28）。概率结果**只跑 MEL/SPO 两城，作附录行**；点预测五城都跑；不是官方 UniST（后者是 grid 模型） |
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
4. **MEL 测试窗少**：MEL 有效测试窗 1781，其余四城 2125。经复核 `load_city` 后五城均为 4392 行连续小时网格（`diff > 1h` 的断点数为 0，任意窗口时间跨度恰为 143h），因此 GluonTS 无需插值，也不存在跨断点的时间错位。MEL 的差异来自划分边界（train/valid 边界比其余城市早 7/11 行），不是缺数据。
5. **seed 现状**：#7–#10、#13、#6 pre72 为 3 seeds；#11 TFT 为 3 seeds；#15 TimeGrad 仅 JHB/SPO 有 2 seeds；#17 DiffSTG 的 MEL 2 seeds、SPO 3 seeds；#14 CSDI、#12 DeepAR-ISQF 与零样本（#18/#19）为单次。主表对多 seed 方法报 mean±std 或 (min/max)，其余报 seed 0。
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
11. **#20 UniTS-FT 不是官方 UniST**：官方 UniST 是 grid（图像张量）模型，依赖跨场景预训练与多模态侧信息，无法直接接入站级 72h 窗口。这里改为「UniTS 公开权重主干 + 每城 dataset prompt 少样本微调 + 残差池」，只作附录行；概率结果仅 MEL/SPO 两城，点预测五城都跑。
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

## 10. 当前状态（2026-10-25 20:30）

* #7 / #13 / #6（iTransformer / MQ-CNN / PatchTST+quant-pre72）与 #8–#10（STGCN / STGCN-nograph / GWNet）：五城 seed 0/1/2 完成。
* #11 TFT：五城 seed 0/1/2 完成。LOA/JHB/MEL 前两 seed 差异 <2%；SZH −6%；**SPO seed0 0.5555 vs seed1 0.4655（−16%）**，SPO 训练集仅 33 站，报告时用多 seed 并列或 mean±std。
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
* #20 UniTS-FT + pool：点预测五城 seed 0 完成；crpsΣ 仅 MEL/SPO 两城（残差池），作附录行。不是官方 UniST，见 §7.11。
* #21 OpenCity-Plus：五城 seed 0/1/2 完成（冻结主干、只训分位头）。
* 时空 FM 调研（2026-10-25）：FactoST 官方仓库返回 404，代码与权重均不可得；UrbanDiT 代码公开（MIT）但 **无预训练权重、无 release**，只有从零训练脚本，不能当 FM 基线；UniTS 权重公开（x32/x64/x128，GitHub release `ckpt`），但它是多变量通用模型、无图结构、原生输出点预测而非分布。可落地的公开权重时空 FM 只有 **OpenCity-Plus**（#21，已接入）与 **UniTS**（#20，附录行）。

## 11. 主结果表（crpsΣ，越低越好）

**口径**：同一批测试窗、同一 clean mask、`seq_l=pre_len=72`；H=24/48/72 取自同一 72 步输出的第 23/47/71 步。多 seed 方法报 mean±std（3 seeds）或 (min/max)（2 seeds）；单 seed 方法直接报 seed 0。五城平均为可用城市的简单平均，缺城不计入。

> `*`：测试窗为该城子集（官方 CSDI 全部城市 256 窗、SZH 128 窗；TSDiff/DiffSTG 的 SZH 为 256 窗）。
> DeepAR 的 StudentT 默认版已排除（预测塌到 0）；表中为 ISQF 版。
> #1–#5 为主实验已有方法，不在此表；#6 的 pre72 版本列入作参照。

#### H=24（主判）

| # | 方法 | LOA | JHB | MEL | SPO | SZH | 五城平均 |
|---|---|---:|---:|---:|---:|---:|---:|
| 7 | iTransformer | 0.2458±0.0015 | 0.2893±0.0016 | 0.3534±0.0009 | 0.4771±0.0019 | 0.0797±0.0002 | 0.2891 |
| 13 | MQ-CNN | 0.2463±0.0004 | 0.2848±0.0009 | 0.3474±0.0004 | 0.4612±0.0030 | 0.0814±0.0001 | **0.2842**（最低） |
| 6 | PatchTST+quant-pre72 | 0.2442±0.0004 | 0.2835±0.0028 | 0.3496±0.0016 | 0.4621±0.0025 | 0.0845±0.0002 | 0.2848 |
| 10 | Graph WaveNet | 0.2505±0.0007 | 0.2920±0.0021 | 0.3708±0.0023 | 0.4876±0.0052 | 0.0870±0.0002 | 0.2976 |
| 11 | TFT | 0.2388 (0.2387/0.2388) | 0.2861 (0.2839/0.2884) | 0.3489 (0.3482/0.3496) | 0.5105 (0.4655/0.5555) | 0.0935 (0.0906/0.0963) | 0.2956 |
| 8 | STGCN | 0.2639±0.0001 | 0.2947±0.0014 | 0.3734±0.0013 | 0.4895±0.0020 | 0.0920±0.0001 | 0.3027 |
| 9 | STGCN-nograph | 0.2677±0.0002 | 0.2958±0.0012 | 0.3799±0.0011 | 0.4937±0.0008 | 0.0909±0.0002 | 0.3056 |
| 18 | Chronos-Bolt | 0.2643 | 0.2977 | 0.3600 | 0.5189 | 0.0868 | 0.3055 |
| 16 | TSDiff | 0.2701 | 0.2942 | 0.3516 | 0.5415 | 0.0889* | 0.3093 |
| 15 | TimeGrad | 0.2545 | 0.2853 (0.2845/0.2861) | 0.3980 | 0.5066 (0.4993/0.5139) | 0.1028 | 0.3095 |
| 19 | Moirai | 0.2722 | 0.3043 | 0.3797 | 0.5329 | 0.1102 | 0.3199 |
| 14 | CSDI（官方） | 0.2962* | 0.3290* | 0.4552* | 0.4927* | 0.1050* | 0.3356 |
| 12 | DeepAR-ISQF | 0.2942 | 0.3287 | 0.4302 | 0.5632 | 0.1909 | 0.3614 |
| 17 | DiffSTG | 0.3931 | 0.3201 | 0.4251 (0.4209/0.4294) | 0.6117±0.0581 | 0.2363* | 0.3973 |

#### H=48

| # | 方法 | LOA | JHB | MEL | SPO | SZH | 五城平均 |
|---|---|---:|---:|---:|---:|---:|---:|
| 7 | iTransformer | 0.2616±0.0007 | 0.3448±0.0048 | 0.3734±0.0008 | 0.5201±0.0015 | 0.0839±0.0001 | 0.3167 |
| 13 | MQ-CNN | 0.2651±0.0002 | 0.3356±0.0021 | 0.3683±0.0007 | 0.5088±0.0030 | 0.0859±0.0001 | **0.3128**（并列最低） |
| 6 | PatchTST+quant-pre72 | 0.2635±0.0004 | 0.3285±0.0026 | 0.3695±0.0015 | 0.5142±0.0051 | 0.0885±0.0002 | **0.3128**（并列最低） |
| 10 | Graph WaveNet | 0.2726±0.0017 | 0.3456±0.0016 | 0.3984±0.0039 | 0.5480±0.0054 | 0.0920±0.0002 | 0.3313 |
| 11 | TFT | 0.2549 (0.2542/0.2556) | 0.3407 (0.3338/0.3477) | 0.3657 (0.3652/0.3662) | 0.6036 (0.5204/0.6868) | 0.0967 (0.0939/0.0996) | 0.3323 |
| 8 | STGCN | 0.2927±0.0004 | 0.3426±0.0032 | 0.3984±0.0011 | 0.5453±0.0016 | 0.0985±0.0002 | 0.3355 |
| 9 | STGCN-nograph | 0.2962±0.0002 | 0.3420±0.0016 | 0.4020±0.0006 | 0.5495±0.0008 | 0.0973±0.0002 | 0.3374 |
| 16 | TSDiff | 0.2946 | 0.3419 | 0.3735 | 0.6038 | 0.0959* | 0.3419 |
| 15 | TimeGrad | 0.3060 | 0.3340 (0.3333/0.3347) | 0.4271 | 0.5513 (0.5422/0.5603) | 0.1286 | 0.3494 |
| 18 | Chronos-Bolt | 0.2944 | 0.3704 | 0.3835 | 0.6156 | 0.0917 | 0.3511 |
| 14 | CSDI（官方） | 0.3126* | 0.3757* | 0.4621* | 0.5515* | 0.1147* | 0.3633 |
| 19 | Moirai | 0.3105 | 0.3849 | 0.4176 | 0.6446 | 0.1281 | 0.3771 |
| 12 | DeepAR-ISQF | 0.3214 | 0.3723 | 0.4631 | 0.6029 | 0.2508 | 0.4021 |
| 17 | DiffSTG | 0.4028 | 0.3467 | 0.4414 (0.4363/0.4464) | 0.6419±0.0615 | 0.2367* | 0.4139 |

#### H=72

| # | 方法 | LOA | JHB | MEL | SPO | SZH | 五城平均 |
|---|---|---:|---:|---:|---:|---:|---:|
| 7 | iTransformer | 0.2673±0.0012 | 0.3666±0.0025 | 0.3931±0.0004 | 0.5381±0.0017 | 0.0851±0.0001 | 0.3301 |
| 13 | MQ-CNN | 0.2724±0.0004 | 0.3562±0.0009 | 0.3870±0.0008 | 0.5245±0.0030 | 0.0874±0.0001 | 0.3255 |
| 6 | PatchTST+quant-pre72 | 0.2701±0.0008 | 0.3474±0.0016 | 0.3879±0.0018 | 0.5242±0.0035 | 0.0895±0.0002 | **0.3238**（最低） |
| 10 | Graph WaveNet | 0.2818±0.0019 | 0.3707±0.0012 | 0.4152±0.0039 | 0.5593±0.0013 | 0.0940±0.0001 | 0.3442 |
| 11 | TFT | 0.2567 (0.2566/0.2568) | 0.3624 (0.3517/0.3732) | 0.3816 (0.3809/0.3823) | 0.6626 (0.5311/0.7941) | 0.0982 (0.0948/0.1015) | 0.3523 |
| 8 | STGCN | 0.3008±0.0004 | 0.3702±0.0027 | 0.4152±0.0014 | 0.5687±0.0016 | 0.1008±0.0001 | 0.3511 |
| 9 | STGCN-nograph | 0.3037±0.0001 | 0.3693±0.0008 | 0.4193±0.0005 | 0.5700±0.0008 | 0.0998±0.0002 | 0.3524 |
| 16 | TSDiff | 0.3009 | 0.3726 | 0.4089 | 0.6122 | 0.1066* | 0.3603 |
| 15 | TimeGrad | 0.3188 | 0.3593 (0.3580/0.3606) | 0.4411 | 0.5664 (0.5597/0.5732) | 0.1376 | 0.3646 |
| 18 | Chronos-Bolt | 0.2973 | 0.4134 | 0.4096 | 0.6726 | 0.0937 | 0.3773 |
| 14 | CSDI（官方） | 0.3044* | 0.3928* | 0.4561* | 0.5392* | 0.1177* | 0.3620 |
| 19 | Moirai | 0.3177 | 0.4126 | 0.4389 | 0.6522 | 0.1323 | 0.3907 |
| 12 | DeepAR-ISQF | 0.3283 | 0.3834 | 0.4832 | 0.6021 | 0.2549 | 0.4104 |
| 17 | DiffSTG | 0.3693 | 0.3551 | 0.4553 (0.4536/0.4571) | 0.6337±0.0493 | 0.1919* | 0.4011 |

### 11.1 读表要点

1. **逐站分位头类（#7 iTransformer、#13 MQ-CNN、#6 PatchTST+quant）稳定占据前三**，五城平均 0.284–0.289（H24），比第二个方法组（图主干/TFT/零样本，0.296–0.320）低约 2–10%。
2. **图结构收益很小且不稳定**：#8 STGCN vs #9 STGCN-nograph 只在 LOA/JHB/MEL/SPO 赢 0.5–1.4%，SZH 反而输 1.2%；#10 GWNet 在 SZH 与 #7 持平，其余城市落后 2–5%。站间弱相关在数据层面已经确认，图卷积拿不到额外信息。
3. **零样本 Chronos-Bolt 很强**（H24 五城平均 0.3055），与 TimeGrad/TSDiff 同档；Moirai 弱一档。说明预训练 TSFM 的先验在本数据上有用，但不及逐站微调的分位头。
4. **扩散类整体不具备竞争力**：TSDiff 0.3093、TimeGrad 0.3095 尚可但输 #7/#13；DiffSTG 0.397，且 LOA/SZH 严重欠离散；官方 CSDI 0.336（子集口径）。
5. **DeepAR-ISQF 明显落后**（0.361），SZH 尤其差（0.191）；GLUONTS 逐站单变量 + ISQF 在大规模站点集上效率低。
6. **TFT 的 SPO 方差极大**（H24 0.4655–0.5555，H72 0.5311–0.7941）：SPO 训练集仅 33 站，seed 选择会显著改变结论。主表用 (min/max) 标注，不应当作稳定排名。
7. **DiffSTG 的失败是结构性的，不是单城/单 seed 意外**：JHB（29 站）校准良好（PICP90 0.922），但 MEL（56 站）0.618、LOA（218 站）0.677、SZH（843 站）0.696，且 LOA 的 MPIW50 只有 1.27；SPO 三 seed 在 0.5594–0.6927 间摆动。当前轻量图扩散在站数放大时学不动历史条件。

### 11.2 校准诊断（H=24，多 seed 先逐城平均再五城简单平均）

名义 PICP50=0.50、PICP90=0.90。crpsΣ 会被“真零格白送分”污染，扩散/深度生成方法必须同时看这张表。

| # | 方法 | PICP50 | PICP90 | MPIW50 (kWh/h) | crps_nz |
|---|---|---:|---:|---:|---:|
| 7 | iTransformer | 0.549 | 0.883 | 20.0 | 0.377 |
| 8 | STGCN | 0.546 | 0.849 | 21.1 | 0.391 |
| 9 | STGCN-nograph | 0.546 | 0.850 | 21.0 | 0.394 |
| 10 | Graph WaveNet | 0.568 | 0.880 | 22.2 | 0.394 |
| 11 | TFT | 0.510 | 0.872 | 21.8 | 0.361 |
| 12 | DeepAR-ISQF | 0.368 | 0.809 | 27.9 | 0.496 |
| 13 | MQ-CNN | 0.540 | 0.893 | 20.8 | 0.379 |
| 14 | CSDI（官方） | 0.280 | 0.590 | 20.1 | 0.439 |
| 15 | TimeGrad | 0.375 | 0.780 | 25.4 | 0.418 |
| 16 | TSDiff | 0.439 | 0.857 | 25.2 | 0.370 |
| 17 | DiffSTG | 0.337 | 0.729 | 26.4 | 0.511 |
| 18 | Chronos-Bolt | 0.540 | 0.776 | 20.4 | 0.393 |
| 19 | Moirai | 0.347 | 0.852 | 26.0 | 0.403 |
| 6 | PatchTST+quant-pre72 | 0.512 | 0.885 | 20.1 | 0.382 |

要点：

* 分位头类（#6/#7/#13）与图主干类（#8–#10）PICP90 在 0.85–0.89，接近名义 0.90；**它们的 crpsΣ 优势不是靠区间开得大**（MPIW50 与扩散类同量级）。
* 官方 CSDI 是最严重的欠离散（PICP50 0.28、PICP90 0.59）——与主实验 #3 的结论一致：整块扩散在本数据上结构性欠离散。
* TimeGrad / DiffSTG 也明显欠离散（PICP90 0.77 / 0.72）；TSDiff 相对好（0.857），是唯一接近分位头类的扩散方法。
* DiffSTG 的 `crps_nz` 在扩散类中最高（0.511）：其在正值格上的误差比边际 crpsΣ 显示的更差，符合“零格白送分掩盖正值格问题”的判断；PICP50 仅 0.337 也说明问题在分布宽度而非位置。
* Chronos-Bolt 的 PICP90 0.776 受原生 9 分位夹断影响（见 §7.2），不应据此判定其分布形状差；其 PICP50 0.540 是可信的。
