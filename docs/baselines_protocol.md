# 概率预测 Baseline 统一协议（#7–#19）

> 目的：在现有 #1–#6 之外补齐 14 个 baseline，全部使用**同一批测试窗、同一 clean mask、同一指标、同一 horizon 口径**，形成 20 方法横向对比表，并作为零膨胀方法的试验底座。
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
| `n_scored_windows` | 实际参与指标的有效窗数（排除全不 clean 的窗） |

* CRPS 用 fair CRPS（`crps_fair`）。
* 分位方法统一转成 200 条样本再算 CRPS（`M_SAMPLE=200`，与 #1–#6 一致；`strat_u` 分层均匀采样 + `quant_to_samples_ind`，转前先 `np.sort(quantiles)`）。
* 样本方法（DeepAR / Moirai / CSDI）使用模型自身样本：DeepAR 200、Moirai 200、官方 CSDI 50（受采样成本限制，见 §7）。

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
| 17 | DiffSTG | `train_diffstg.py` | 时空图扩散：[N,72] 联合扩散，每 block「时间膨胀 conv + kNN 稀疏图聚合 + 时间 FiLM」；历史条件用 GRU 末状态（v2；v1 的 `cond_enc(...).mean(dim=3)` 在 RevIN 后≈0，会退化成窗口均值，产物已归档到 `result/baselines/archive_meancond_v1/`）；轻量版 UGnet；DDIM 50 步、η=1 |
| 18 | Chronos-Bolt | `run_zeroshot.py --method chronos_bolt` | 零样本，`chronos-bolt-small`，原生 9 分位 |
| 19 | Moirai | `run_zeroshot.py --method moirai` | 零样本，`moirai-1.0-R-small`，200 条样本→99 经验分位 |
| 20 | UniST | — | 待评估依赖与迁移设定（见 §10） |

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
5. **seed 现状**：新方法先全跑 seed 0；零样本（Chronos/Moirai）确定性，只跑一次。seed 1/2 后续补 mean±std。
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
11. **#20 UniST 未做**：官方实现是 **grid（图像张量）** 模型，且依赖大规模跨场景预训练 + 多模态侧信息；本数据的站级图结构与统一 72h 输入无法在不做实质改造的情况下接入。若必须补，需要按 UniFlow（graph 版）重做数据层与预训练，成本另计。

## 8. 产物

```
result/baselines/
  metrics/{method}/{city}/H{24,48,72}_seed{s}.json   # 指标 + 超参 + 窗口数
  diag/{method}/{city}/seed0.npz                     # 16 窗 × 32 分层站 × 72 × 99 分位，float16
  ckpt/{method}/{city}/seed{s}.pth
  logs/*.log
```

诊断子集固定 seed 0、按零率分层选站，供开发零膨胀方法时看预测形状；不落全量预测。

## 9. 复现

```bash
# #7–#10（seed 0，五城）
for c in LOA JHB MEL SPO SZH; do
  for m in itransformer stgcn stgcn_nograph gwnet; do
    python3 script/baselines/train_quant_backbone.py --method $m --city $c --device 0
  done
done

# #11/#12
python3 script/baselines/train_gluonts.py --method tft   --city MEL --device 0
python3 script/baselines/train_gluonts.py --method deepar --city MEL --device 0

# #14（官方 CSDI；SZH 用 128 窗）
python3 script/baselines/csdi_official.py --city MEL --device 0 --n_eval_windows 256
python3 script/baselines/csdi_official.py --city SZH --device 0 --n_eval_windows 128

# #18/#19
python3 script/baselines/run_zeroshot.py --method chronos_bolt --city MEL --device 0
python3 script/baselines/run_zeroshot.py --method moirai --city MEL --device 0

# 汇总（seed 0，crpsΣ）
python3 script/baselines/summarize.py
python3 script/baselines/summarize.py --markdown
```

## 10. 当前状态（2026-10-25 00:10）

* #7 / #13 / #6（iTransformer / MQ-CNN / PatchTST+quant-pre72）：五城 seed 0/1/2 完成；#8–#10 STGCN / STGCN-nograph / GWNet 仅 seed 0（seed 2 未跑）。
* #11 TFT：五城 seed 0 完成；seed 1 四城完成、SZH 进行中。
* #12 DeepAR：主表用 ISQF 版本（`--distr isqf`），五城 seed 0 完成；StudentT 默认版本在多数城市退化（预测塌到 0，SPO crpsΣ 0.87），仅作参考、不进主表。
* #13 MQ-CNN：五城 seed 0/1/2 完成。
* #14 官方 CSDI：五城完成（LOA/JHB/MEL/SPO 256 窗，SZH 128 窗，`nsample=50`）。
* #18/#19 Chronos-Bolt / Moirai：五城完成（确定性，单次）。
* #6 pre72（PatchTST+quant）：五城 seed 0/1/2 完成。
* #15 TimeGrad：LOA/JHB/MEL/SPO 完成，SZH 全量运行中。
* #16 TSDiff：五城完成（SZH 为 256 窗子集），H24 crpsΣ LOA/JHB/MEL/SPO/SZH = 0.2701/0.2942/0.3516/0.5415/0.0889*。
* #17 DiffSTG：v3（每 block 历史 FiLM）JHB/MEL/SPO 完成，H24 crpsΣ 0.3201 / 0.4294 / 0.6927（SPO 严重塌缩：PICP50 0.056、PICP90 0.384）；LOA 全量、SZH 256 窗运行中。v1/v2 已归档，不进主表。
* #20 UniST：未实现（原因见 §7.11）。
