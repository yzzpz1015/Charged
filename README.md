# Charged — EV 充电需求预测

多站点电动汽车充电量（volume ≈ 充电功率代理）小时级预测。核心科学问题：**在微调（fine-tuning）之上，外部模态（卫星图 / POI / 文本）与历史检索能否再产生预测增量？**

> 阶段性结论：**只有 fine-tuning 稳健**（冻结 Kairos TSFM + 点预测头，R²≈0.79–0.80）；检索 / 邻居 / 外生时序 / 静态上下文 / 多模态逐层注入，随预测 horizon 拉长（6→24）要么归零要么转负。完整论证见 [当前进度9.6.md](当前进度9.6.md)。

---

## 1. 数据与权重在哪

**代码在 GitHub，大文件在 HuggingFace。** 仓库 `.gitignore` 排除了 `data/`、`tiles_dataset/`、`api/model/weights/`，这三者需从 HF 单独下载。

| 内容 | 位置 | 说明 |
|---|---|---|
| 源代码 | `github.com/yzzpz1015/Charged` | `git clone git@github.com:yzzpz1015/Charged.git` |
| 数据集 | HF Dataset [`yzzpz/Charged-data`](https://huggingface.co/datasets/yzzpz/Charged-data) | 公开；含 `data/`（6 城时序）+ `tiles_dataset/`（卫星图） |
| 模型权重 | 官方 HF 模型库 | GPT-2、SigLIP v2、Kairos_23m（见下） |

### 下载数据（在项目根目录执行）

```bash
# 直接下载（落地到 data/ 与 tiles_dataset/）
hf download yzzpz/Charged-data --repo-type dataset --local-dir .

# 国内镜像（若直连慢）
HF_ENDPOINT=https://hf-mirror.com hf download yzzpz/Charged-data --repo-type dataset --local-dir .
```

### 下载模型权重

```bash
python script/download_weights.py   # GPT-2 + SigLIP v2，走 hf-mirror
```

Kairos 主干（`mldi-lab/Kairos_23m`）在训练/推理时经 `--checkpoint` 自动从 HF 下载（`--kairos_repo` 指定本地缓存路径，默认 `/tmp/Kairos_repo`）。

---

## 2. 数据说明

- **`data/<CITY>/`**：6 城（AMS / JHB / LOA / MEL / SPO / SZH）小时级充电数据，每城 10 个 CSV（volume / duration / e_price / s_price / sites / chargers / distance / weather / poi / info）。干净版本两级：`<CITY>_remove_zero/`（剔除全零充电站）、`<CITY>_remove_const/`（再剔除「恒值」站：恒 0 闲置残漏 + 恒非 0 常量填充伪影，**训练默认用此**）。被剔除站点的排除清单见 `data/constant_sites.csv`（539 站，含 `kind` 字段区分 zero/fill）。字段说明见 `data/README.md`。
- **`tiles_dataset/<CITY>/`**：JHB / MEL / LOA 三城的卫星瓦片（`tiles/` 与拼接 `mosaic/` PNG，z16）及 DINOv2 预计算嵌入（`embedding/z16/`），索引见 `tiles_dataset/site_tile_index.csv`。

---

## 3. 复现核心实验

依赖：`pip install -r requirements.txt`（若有）；PyTorch + `transformers` + `huggingface_hub`。

```bash
# 零样本基线（Kairos 原生分位数头）
python script/zero_shot_kairos.py --city MEL --seq_l 72 --pre_len 6

# L1 —— 冻结 Kairos + 微调点预测头（唯一稳健的增益来源）
python script/train_kairos_l1.py --city MEL --variant plain --seq_l 72 --pre_len 6 --fold 6

# L1 变体：LoRA 头 / 统计文本 token 化
python script/train_kairos_l1.py --city MEL --variant lora --seq_l 72 --pre_len 6
python script/train_kairos_l1.py --city MEL --variant l1b  --seq_l 72 --pre_len 6
```

> `--data_path` 默认 `data/`，运行时自动 append `--data_suffix`（默认 `_remove_const`，指向已剔恒值站的干净版）。传 `--data_suffix _remove_zero` 可复现历史 R²（见下表）。

探针脚本（均为负结果，用于论文严谨性论证）见 `script/`：邻居流 `neighbor_probe*.py`、外生时序 `exog_probe.py`、静态上下文 `static_context_*.py` / `sat_patch_probe.py`、检索 `retrieval_probe.py` / `kairos_retrieval_teacher.py`、多模态注入 `train_kairos_l2v3.py`。

---

## 4. 分层实验框架（R²，MEL fold6）

| 层级 | 定义 | pre_len=6 | pre_len=24 |
|---|---|---|---|
| L0 | Kairos_23m 零样本 | 0.6705 | 0.6646 |
| **L1** | 冻结 Kairos + 点预测头 | **0.7982** | **0.7901** |
| L2 | L1 + 三路外部模态逐层注入 | 0.8028（正则假象） | 0.7695（有害） |
| PatchTST（训练） | 专门训练基线 | 0.7866 | 0.7739 |
| FreTS（训练） | 频域基线 | 0.7845 | 0.7870 |

> 上表为 `_remove_zero` 上的历史结果（MEL fold6）。自 `_remove_const` 成为默认后，重跑会在已剔恒值站的数据上略变（MEL 仅剔 2 站，几乎不变；SZH 剔 519/1379 站，变化显著）。

---

## 5. 目录结构

```
api/
  model/     # 模型实现（kairos_l1 / kairos_l2v3 / patchtst / frets / ev_vlm …）
  dataset/   # 数据加载与 collate
  trainer/   # train/valid/test 循环
  parsing/   # CLI 参数
script/      # 训练与探针脚本入口
data/        # 时序数据（从 HF 下载）
tiles_dataset/  # 卫星图与 DINOv2 嵌入（从 HF 下载）
result/      # 实验输出
当前进度9.6.md  # 阶段性结论与完整负结果叙事
```
