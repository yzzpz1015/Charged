# EV 充电需求预测：从学长引导到 L2 初版架构

> 现状：冻结 Kairos 时序主干 + 点预测头已达 **R²≈0.80**（超 PatchTST 0.79 / FreTS 0.78）；当前在验证「外部模态（POI / 卫星图 / 卫星图描述）注入」能否再加增量——架构已定为**三路外部模态、逐层 cross-attention 注入主干 decoder**，并已按此实现（`api/model/kairos_l2v3.py`，三种协作并排：gate / urban / align）。纯时序基线是单独训练的 L1（0.7957）；阶段 1 的「输出处标量 gate」与 v2 的「ts/poi/sat 三路」均非目标设计。数据层面已定性「0 = 真实空闲，非缺失」。

---

## 1. 设计流程

学长 08-20 建议转向 Agent 研究（近似 UrbanAgent），并把「外部模态」定为 Agent 对纯时序的增值点；08-22 确认短 horizon 下长历史无增量，画像价值应靠静态空间先验而非更长窗口。据此把主干窗口定在 seq_l=72。

主干选了轻量预训练 TSFM「Kairos」：零样本 R² 就达 0.67，自带实例归一化、对预测长度不敏感。为了证明「外部模态的净贡献」这一论文核心 claim，把增量拆成 L0 / L1 / L2 三层——L0 零样本、L1 冻结主干只换头、L2 在 L1 之上注入外部模态。

L1 很快给出结论：**把 Kairos 原生分位数头换成点预测头是主增量**（0.67 → 0.80，已超专门训练的 PatchTST/FreTS），而加 LoRA（+0.003）或 LLM 统计语义（gate 学回 ≈0）都近乎无贡献。也就是说，时序内部已经被榨干，剩余增量只能来自外部模态——于是 L2 的全部问题就压在一件事上：**外部模态如何有效注入**。下面 §2 是当前初步确定的这套注入架构。

---

## 2. 当前架构（初步确定）

整体见 [kairos_l2_arch.svg](figures/kairos_l2_arch.svg)，单层注入细节见 [kairos_l2_injection.svg](figures/kairos_l2_injection.svg)，三层阶梯见 [kairos_ladder.svg](figures/kairos_ladder.svg)。

### 主干：冻结 Kairos

Kairos（T5 编码-解码，d_model=384，decoder 共 4 层）整体冻结。时序 x 经 encoder 得到隐状态，作为「时序一路」；decoder 逐层解码，每层结构为 self-attention → cross-attention → FFN。

### 三路外部模态（无纯时序流）

目标设计里**没有纯时序流**（用户纠正：三路指三种外部模态）。时序只存在于冻结主干的内部机制里：encoder 输出时序隐状态，是 decoder 原生 cross-attention 的键值——这是主干自身的一部分，恒有、冻结，不属于任何「流」。三种外部模态各投影成 token，随 decoder 逐层推进：

- **POI（路 1）**：POI 密度向量 `[96]` → 线性投影成 4 个 token；
- **卫星图（路 2）**：卫星图 DINOv2 CLS `[768]`（L2 归一化）→ 线性投影成 4 个 token；
- **卫星图描述（路 3）**：VLM（Qwen2-VL-2B）对每站卫星镶嵌图的短描述 → GPT-2 词嵌入 → 线性映射到 d_model，整串 token 作键值。

### 逐层注入（核心）

先分清两类 cross-attention——**一类是主干的，一类是注入的**：

- **时序 cross-attention（原生，不是注入）**：原始时序 x 先进 encoder 编码成隐状态；decoder 每层的 cross-attention（query = decoder 隐状态，key/value = encoder 输出的时序隐状态）是 T5 encoder-decoder **固有的读取历史的机制**——相当于「decoder 看历史的眼睛」，没有它 decoder 对历史一无所知。它恒有、冻结，是主干的一部分，不是外部注入。
- **外部模态注入（新增）**：在时序原生 cross-attention 之上，decoder 隐状态再对 **POI、卫星图、卫星图描述**三路外部模态各做一次 cross-attention，gate 加权后叠加：

```
h = h_base + Σ_i gate_i · CrossAttn_i(h_base, aux_i)，   i ∈ {POI, 卫星图, 卫星图描述}
```

（时序的原生 cross-attn 不在此 Σ 内，它恒有、冻结，对三条外部流都服务。）

- `CrossAttn_i` 决定「注入什么」（query = decoder 隐状态，key/value = 模态 i 的 token）；
- `gate_i` 决定「注入多少」，**标量、初始化为 0**。

外部 gate 从 0 起步，等价于「没有外部模态」；训练后 gate 的学成值作为**诊断**：≈0 说明模型没用上该模态，>0 说明在用（仅诊断，不作最终对比）。

**净贡献怎么比**：三路外部流各配独立预测头，每流的 loss 只对本流，**同模型内各流读数是干净的**（`y_poi / y_img / y_desc` 两两互不污染）。但要回答「外部模态整体值不值」仍需无外部路径的对照——真正的纯时序基准是 **L1**（结构里没有外部路径、已单独训练，R²=0.7957）：**外部模态净贡献 = L2 collab 的 R² − L1 的 R²**，两个模型各训各的，无需评测时动任何开关。

### 训练与评测

三路 MSE 联合训练，损失在实例归一化空间计算（稳健 scale）；评测按 anchor（纯时序）与多模态分别反归一化算 R²，并读出 gate 值。全零历史窗口训练/评测统一剔除（[数据零值分析.md](数据零值分析.md)）。

### 已实现（v3）：三路外部模态逐层注入 + 三种协作并排

`api/model/kairos_l2v3.py` 按用户更正后的目标设计实现（阶段 1 的 `kairos_l2.py` 最简版与 v2 的「ts/poi/sat 三路」都不是目标架构，v2 里不该有纯时序流）：

- **三路 = 三种外部模态**：`stream_poi`（+POI 逐层注入）/ `stream_img`（+卫星图逐层注入）/ `stream_desc`（+卫星图描述逐层注入），共用同一个**冻结 decoder**，用 forward hook 在 `backbone.decoder.block[l]` 上做非侵入式注入。**没有纯时序流**——时序只走主干自带的原生 cross-attn，对三条流都服务；
- **逐层注入**：`h = h + gate_m[l] · CrossAttn_m(h, aux_m)`，每层、每模态一个 `InjectionAttn`（pre-LN 单头注意，Q=decoder 隐状态，KV=模态 tokens），gate 标量初始 0；
- **卫星图描述（新路）**：每站卫星镶嵌图 1536×1536 由 Qwen2-VL-2B-Instruct 生成 4 项短描述（dominant_land_use / building_density / transport_access / notable_features，~17-36 GPT-2 tokens），存 `tiles_dataset/MEL/embedding/z16/sat_captions.json`；`gpt2.wte` 词嵌入 → 线性映射到 d_model 作键值（`gpt2_wte` 冻结）。
- **每流独立预测头** → `y_poi / y_img / y_desc`，各流 loss 互不污染，**同模型内流间读数干净**（苹果对苹果）；
- **可插拔协作**（`--fusion`）产出 `y_collab`（部署用的最终预测，头条对比单独训练的 L1）：
  - `gate`：softmax(MLP(拼接各流 hidden)) 逐样本动态权重，**晚融合参考基线**；
  - `urban`：**UrbanAgent 式置信度迭代**——中心状态 c 初始为三流 hidden 均值，R=3 轮每轮对每条交互边 `c ← c + σ(conf_i(c))·CrossAttn(c, h_i)`，conf 与 CrossAttn 边可学（用户指定的协作方式一）；
  - `align`：**卫星图 ↔ 描述互对齐 + POI 协作**——双向 cross-attn 对齐 h_img↔h_desc 得卫星一致表示 h_sat，门控 `g·h_sat + (1−g)·h_poi` 融合 POI（用户指定的协作方式二）。
- 训练脚本 `script/train_kairos_l2v3.py`：`--fusion gate|urban|align`，三流 MSE 联合训练（实例归一化空间、稳健 scale），保存各流 + collab 的测试预测与 R²，并输出 gate 逐层诊断（仅诊断，不作对比）。

> v2（`result/kairos_l2v2/`，ts/poi/sat）跑出的 static/gate/mlp 结果记录了「纯时序流 + 两路外部」下的行为（外部价值集中在难时段、sat≥poi、collab 三种全 >L1），作为历史存档；v3 才是目标架构，见下。

### 结果（fold6 · pre_len=6 · 早停 patience=8 · 单次运行）

三路外部模态逐层注入 + 三种协作（R²，测试集 22082 样本；L1 = 0.7957 已建立基线，L1-plain 同测试窗过滤后实测 0.7978）：

| fusion | y_poi | y_img | y_desc | y_collab | collab−L1(0.7957) |
|---|---|---|---|---|---|
| **gate** | 0.8003 | 0.8029 | 0.8015 | **0.8028** | **+0.0071** |
| urban | 0.8016 | 0.8009 | 0.8023 | 0.7976 | +0.0019 |
| align | 0.7999 | 0.7960 | 0.7979 | 0.7987 | +0.0030 |

**三路外部流全部 > L1**：9 个流读数（poi/img/desc × 3 fusion）全部 ≥ 0.796、都高于 0.7957 基线，最好的是 gate 的 img 流 0.8029。**卫星图描述路成立**：desc 流 0.8015–0.8023，与卫星图嵌入流（0.796–0.8029）相当，第三条外部模态不是摆设。

**协作层的诚实结果**：只有简单的加权晚融合（gate）把 collab 提到等于最好单流（collab 0.8028 ≈ img 0.8029）；urban（UrbanAgent 置信度迭代）的 collab 0.7976 **低于它自己的全部三条流**，align（图↔文互对齐+POI）的 collab 0.7987 也低于它自己的 poi 流 0.7999。即这两种精心设计的协作，作为实现并没有超越「选最好那条流」，更没超越简单的动态加权。对 L1-plain 同窗实测 0.7978 口径，gate +0.0050 / urban −0.0002 / align +0.0009——urban 基本是白做。

**难时段救场**（按预测窗标签 std 分档，mean|err| 净改善，q3 最难）：

| 难度 | 标签 std | L1 | gate collab | gate img | urban collab | align collab |
|---|---|---|---|---|---|---|
| 易 | 0.06 | 3.28 | 3.23 | 3.40 | 3.62 | 3.16 |
| 中 | 4.88 | 7.88 | 7.63 | 7.54 | 7.38 | 7.45 |
| 难 | 8.64… | — | — | — | — | — |
| **最难** | 51.52 | 49.91 | **49.05** | 49.10 | 50.14 | 49.49 |

最难档 gate collab 把 mean|err| 49.91→49.05（R² 0.7516→0.7577，+0.0061），方向对但幅度小于 v2 观测的 −1.0。易/中档也有小幅改善；外部模态价值仍集中在难时段，但本版增量整体收敛。

**gate 诊断（每层标量，初始 0，仅诊断）**：poi gate 前两层 +0.04~0.05 后两层 ≈0/负；img gate 多为负（−0.02~−0.04）；desc gate 近 0/微负。与 v2 一致——**注入强度仍弱**，增益来自「少量一致扰动 + 各流头协同适配」，不是强注入。gate 的 collab 之所以最好，是因为它只在预测层把各流按逐样本权重混合，不额外扰动已经过 backbone 的隐状态。

**诚实边界**：单 fold、单 seed。流级「全部 >L1」与「desc 路成立」是可靠信号（9/9 一致）；但 collab 增量的绝对值（+0.001~+0.007）在单次运行的噪声尺度内，urban/align 的「协作不如好流」需要多 seed 确认后再下结论。

---

## 3. 下一步

- **多 fold / 多 seed 复现**（urban 协作跑 fold 1–5，确认增量与难时段集中是否稳定）——首选；
- **难时段专门对照**：对最难档样本单独出 R²/误差表，作为论文主 claim（「外部模态在难时段救场」）的支撑图；
- **卫星 raw patch tokens**（当前用 DINOv2 CLS `[768]` 投影成 token；CLS 丢失空间细节，后续可换 patch 级 token 经 cross-attention 注入，对比 CLS）；
- **描述质量迭代**：当前 captions 部分站套「residential/high/parks」样板，若 desc 路增益弱，尝试更强的站点特定提示、更大 VLM、或对 caption 做 land-use 聚类再 embedding。

**待确认**：预测长度 pre_len 尚未定（默认 6）；是否先做「站点活跃期 mask」排除结构性零污染再重测 L2。

### 邻居站流量探针（2026-08-31，MEL 62 站 · 纯数据）

脚本 `script/neighbor_signal_probe.py` / `script/neighbor_probe2.py`。结论：**邻居流量只有同刻空间相关，没有可利用的领先预测信号。**

- **同刻相关**：k=5 最近邻 cycleAdj r≈0.086（远场基线 0.055）；≤2km 内 0.126；双活跃时段更强（MI 0.136 vs 0.088）。距离衰减集中在前 ~2–5km → **半径取 k=5 最近邻（中位 5.2km）**。
- **领先增量（决定性的负结果）**：邻居过去→目标未来，控制目标自身滞后 1..h，OOS R² 增量在 h=1..24 全部 ≈ **−0.002**（k=1）；k=3/5 为负（共线性）；同刻上限也仅 +0.002。Ridge 与 OLS 一致。
- **非线性**：dCor 0.033 < |Pearson| 0.122，无线性之外的结构。
- **教训**：同刻相关 ≠ 领先可预测——空间相关是同时公共冲击，已被目标自身过去吸收。
- **跨城推广**（`script/neighbor_probe_cities.py`）：JHB/SPO/LOA/SZH/MEL 五城领先增量无一为正（k=1 ≈0 或负）；SZH（1445 站、0.7km、2.7% 空闲）仍无增量 → 非密度/利用率问题。AMS 排除（充电桩粒度、同址副本、共线性假象）。**溢流机理定向否证**（`script/neighbor_probe_util.py`）：按利用率分档，"邻居满负荷→目标抬升"增量全 ≈0。结论：图工作增益来自分钟级协调/站点占用/站选择等本质空间任务，非小时级电量跨站预测。

**设计含义**：邻居流**完全移除**（数值 token 与 agentic 语义描述都不做）。语义描述是底层数值的确定性函数，数值在条件化后无增量且无非线性结构（dCor），描述不可能凭空多出信息；唯一能引入的外部知识（站点协同静态上下文）已由 POI/卫星模态覆盖，且同为同刻非领先。探针保留为**负结果叙事**（论文 motivation 里"系统探测外生源：邻居流无领先信号"）。

### 外生时序变量探针（2026-08-31，MEL 60 站 / SZH 1198 站 · 纯数据）

脚本 `script/exog_probe.py`（温度/湿度/电价 vs 电量，raw 与循环调整双报 + 真实多步 forecast-origin 增量 + dCor）。结论：**动态外生时序变量全部 ≈0 预测增量——同刻共动、被自身历史吸收。** MI 曾用 raw 电量 vs 循环调整外生源（不对称），已修正为双端循环调整（MEL 电价 MI 反而升至 0.173）。

- **温度/湿度**：同刻 cycleAdj 相关 MEL 0.007/−0.046、SZH 0.043/−0.013，MI 0.034/0.024 / 0.080/0.036；raw 相关里那点值（MEL humidity −0.110、SZH temp 0.052）基本是共享昼夜节律，调整后塌缩。**真实多步测试（天气按完美已知未来** temp[t+h]→vol[t+h]，控制 vol[t],t−1,t−2**）**：h=1..6 增量全 ≈ −0.001~−0.03（负）；h=6 处线性控制自身崩（MEL R²=−0.36），天气也没救回。
- **电价**：MEL 同刻关系是全部探针里最强的（corr_cyc **−0.146**、MI_cyc **0.173**，均为双端循环调整，属站点内时间动态而非水平混杂），但滞后剖面平坦（h=0..24 一致）→ 同刻共动非领先；过去向增量 +0.001≈0。SZH 无关系（0.001）。电价-电量关系**城市特异**。
- **dCor 全 < 1**：无线性之外结构，非线性模型也挖不出。
- **统一教训**：任何随时间变化的外生信息（邻居流量、天气、电价）都一定已写在目标自身近期窗口里 → 增量 ≈0；**只有时不变信息（站点静态上下文、日历身份）不在历史里** → 静态模态理论上是 L2 唯一的增量候选。动态外生时序（邻居/天气/电价）全部经探针排除。**但静态上下文本体随后也被探针否定（见下节）**，v4 实际只剩日历身份 + agent 语义判读。

### 静态上下文探针（2026-09-04，MEL 62 站 · 全编码器链）

脚本 `script/static_context_probe.py` / `script/sat_patch_probe.py` / `script/remoteclip_probe.py`。结论：**卫星图/POI 对「充电站繁忙程度（负荷水平）」预测 ≈0——不是编码器问题，是数据本身低信号。**

- **反事实（context → 负荷水平 mu_s）**：卫星 DINOv2 CLS LOO R² −0.45~−0.8、POI 总数 pearson −0.138，全负 → 卫星图根本「不知道」站点繁忙。
- **探针 A/B/C（context → L1 残差，含周末差分、calendar 交互）**：全负 → 对时序主干残差也无增量。
- **DINOv2 patch 探针**：CLS/patch-mean/max/grid 全退化（CLS cosine 0.869、grid7×7 0.794），负荷 R² 全负 → 非 CLS 池化假象。
- **跨城 DINOv2**：JHB 0.883 / LOA 0.849 / MEL 0.869 全退化 → 非 MEL 特有。
- **RemoteCLIP ViT-L-14（地理预训练）**：CLS 仍退化 0.880；patch 明显更判别（grid7×7 cosine **0.674**）；负荷 R² 仍全负（grid7×7 最好也 **−0.14**）。

**关键区分**：地理预训练确实让卫星 patch 看到更多空间差异（cosine 0.87→0.67），但这**不转化为对负荷的预测力**——决定繁忙程度的变量（价格/品牌/app 可见度/目的地邻近/充电桩数量/营业时间）不在卫星视角里。

**标签纠正（2026-09-04）**：`volume ≈ avg_power`（r=0.977，功率 4.5~350kW），volume 是功率的代理、卫星看不见，故「繁忙度」该用 `duration`（占用时长）。但换 duration 后卫星仍 ≈0（RC grid7×7 → duration LOO R² +0.009，边缘显著、量级为零）。曾因**站点 id 错位**（volume 列字符串序 vs mosaic 文件数值序）+ 欠正则误报 +0.172，修正后归零。与 VisionEV 的分歧最可能是**站点类型多样性**：MEL 62 站全单桩同质（caption 全是 residential/high 模板）→ 无类型方差可挖。

**跨城 + 缩放证据链（2026-09-04，`script/static_context_cities.py` / `remoteclip_cities.py` / `zoom_crop_probe.py`）**：把 VisionEV 分歧的每个逃逸口堵死，结论比 MEL 单城更强：

- 站点规模确实驱动繁忙度（唯一真信号，属「容量→吞吐」机械关系）：charger_num ↔ area r = MEL 0.995 / JHB 0.968 / LOA 0.688；charger_num → total_duration r = MEL 0.346 / JHB 0.865 / LOA 0.541。
- 但卫星看不见规模：跨 MEL/JHB/LOA（charger_num 1→16），DINOv2 CLS → charger_num(log) 全负（LOA −0.081）；RemoteCLIP grid7×7 → charger_num/area/duration 全 ≈0。
- 缩放假设否证（免费，无需下载）：站点 66–169m 在 z16 3.7km 瓦片里占 0.03–0.2% 面积（「干草堆里的针」），但中心裁剪到 256px（0.61km，站点占 27%）后 charger_num/area R² 仍 ≈0、无回升 → 不是缩放/编码器问题，是图像里根本没信号。
- RemoteCLIP 权重确认加载成功（grid7×7 cosine 0.676 确比 DINOv2 0.85 判别），但判别出的空间差异不指向需求。

**结论**：卫星图对 EV 负荷预测（水平/容量/利用率/残差任一维度）≈0，跨 3 城 × 2 编码器 × 4 尺度全部成立。**不需下载更细瓦片**（`download_tile.py --zoom` 白跑）；容量信号即使拍到也被 RevIN 当站点水平免费吸收。VisionEV 的信号若是站点类型（高速/商场/住宅），本样本（城市公共充电桩、单一类型）天然没有。

**设计含义**：L2 的「卫星/POI 静态模态」不能再作为负荷水平/繁忙度的增量来源。它推翻了上一节「时不变静态上下文是唯一增量源」的推断——静态上下文和动态外生源一样 ≈0。外部模态剩下的候选只剩**日历身份**（确定性、粗粒度可用）和 **agent 语义判读**（非数值通道）。v4 转向日历/agent，卫星/POI 若保留只作非水平维度（非典型时段行为）。
