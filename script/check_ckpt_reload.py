# -*- coding: utf-8 -*-
"""回归测试：``train.pth`` 在**新进程**里重放，必须逐元素等于训练进程写下的 ``predict.npy``。

为什么需要这个
--------------
``PredictionTrainer.test()`` 是在训练进程里跑的——模型的 BatchNorm
``running_mean/running_var`` 是 buffer，内存里已经是训练累积的正确值，所以
**训练进程内的 test() 看不出 checkpoint 缺东西**。只有当新进程
``load_model('train.pth')`` 时，缺失的 buffer 才会退回初始化值（mean=0/var=1），
预测静默偏移（实测 SPO raw MAE 10.9 vs 正确 2.99，差 3.6 倍）。

2026-10-05 修了 `api/trainer/common.py` 的保存逻辑（可训练参数 ∪ 全部 buffer）。
本脚本把这条不变量固定下来。

用法：``python3 script/check_ckpt_reload.py [--cities LOA ...]``
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

BASE = "/home/data/yzzpz/Charged"
sys.path.insert(0, BASE)

SEQ_L, PRE_LEN = 72, 72


def check(city, device):
    from api.dataset.common import EVDataset
    from api.model.config import PredictionModel

    hits = sorted(glob.glob(f"{BASE}/result/univariate/{city}/patchtst/"
                            f"volume-None-site-{SEQ_L}-{PRE_LEN}-6*/predict.npy"),
                  key=os.path.getmtime)
    if not hits:
        return f"{city:<5} 跳过（没有 {SEQ_L}-{PRE_LEN} 的 predict.npy）"
    run_dir = os.path.dirname(hits[-1])
    saved = np.load(os.path.join(run_dir, 'predict.npy'))       # 训练进程写下的（正确）

    ds = EVDataset(feature='volume', auxiliary='None',
                   data_path=f"{BASE}/data/{city}_remove_const/")
    ds.split_cross_validation(fold=6, total_fold=6, train_ratio=0.8, valid_ratio=0.1)
    ds.create_loaders(seq_l=SEQ_L, pre_len=PRE_LEN, batch_size=16,
                      device=device, model_name='patchtst')
    model = PredictionModel(num_node=ds.feat.shape[1], n_fea=1, model_name='patchtst',
                            seq_l=SEQ_L, pre_len=PRE_LEN)
    model.load_model(os.path.join(run_dir, 'train.pth'))        # 新进程 reload
    model.model.to(device).eval()

    te = ds.test_loader.dataset
    outs = []
    with torch.no_grad():
        for b in DataLoader(te, batch_size=16, shuffle=False):
            outs.append(model.model(b[0].to(device)).float().cpu().numpy())
    pred = np.concatenate(outs, 0)
    # 比较在**原空间**做（与 predict.npy 同空间），逐元素
    M, N, P = pred.shape
    a = pred.transpose(0, 2, 1).reshape(-1, N)
    a = ds.scaler.inverse_transform(a).reshape(M, P, N).transpose(0, 2, 1)

    if a.shape != saved.shape:
        return f"{city:<5} 形状不一致：reload {a.shape} vs saved {saved.shape}"
    d = np.abs(a - saved)
    rel = d.max() / max(np.abs(saved).mean(), 1e-9)
    ok = rel < 1e-4
    return (f"{city:<5} {'OK ' if ok else 'FAIL'}  形状 {a.shape}  "
            f"最大绝对差 {d.max():.3e}  相对 {rel:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cities', nargs='*',
                    default=['LOA', 'MEL', 'JHB', 'AMS', 'SPO', 'SZH'])
    ap.add_argument('--device', type=int, default=0)
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print("train.pth 新进程重放 vs 训练进程 predict.npy（应逐元素相等）")
    print("-" * 78)
    bad = 0
    for c in args.cities:
        msg = check(c, device)
        print(msg)
        if 'FAIL' in msg:
            bad += 1
    print("-" * 78)
    print("全部通过 ✓" if bad == 0 else f"{bad} 个城市不一致 ✗")


if __name__ == '__main__':
    main()
