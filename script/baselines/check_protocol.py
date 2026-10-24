# -*- coding: utf-8 -*-
"""协议门：新底座必须复现现有评测的数字，否则后面所有方法都不可比。

1. ``--what kairos``：Kairos+quant MEL，pre72 窗、H=24，目标 **0.3542**。
   （README 消融表的 0.3479 查无实据：同值出现在旧 LOA pre24 残差基线日志里，
   是抄写串行；现有 ``kairos_prob_eval.quant_pass`` 在同一窗集上给出
   ``crps_norm_sum = 0.354216``，与主表 ③ 的 0.3542 一致。）
2. ``--what patchtst``：现有 pre24 checkpoint 在 **pre72 窗集**上重算 j=23，
   与 ``eval_patch_quant.py`` 的 pre24 结果对账（窗集不同，数值不应相等；
   这里只确认流程一致并记录 pre72 数值）。

用法::

    python3 script/baselines/check_protocol.py --what kairos --city MEL --device 0
"""
import argparse
import os
import sys

import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "script"))

import common as C  # noqa: E402


def load_kairos_quant(city, out_root, device, n_quant=99):
    sys.path.insert(0, "/tmp/Kairos_repo")
    import tsfm.model.kairos  # noqa: F401
    from transformers import AutoModel
    from api.model.kairos_l1 import KairosL1
    bb = AutoModel.from_pretrained("mldi-lab/Kairos_23m",
                                   trust_remote_code=True).to(device).eval()
    qs = np.arange(1, n_quant + 1) / (n_quant + 1)
    model = KairosL1(bb, pre_len=C.SEQ_L, quantiles=qs).to(device)
    ck = os.path.join(out_root, city, f"quant_pre{C.SEQ_L}_fold{C.FOLD}.pth")
    model.load_state_dict(torch.load(ck, map_location=device))
    return model.eval(), qs


@torch.no_grad()
def infer_kairos(model, raw, starts, qs, device, batch_rows=512):
    """返回 [M,N,72,Q]（raw 单位），按批生成以免 [M*N,72,99] 爆内存。"""
    M, N = len(starts), raw.shape[1]
    out = np.empty((M, N, C.PRE_LEN, len(qs)), np.float32)
    w = max(1, batch_rows // N)
    off = np.arange(C.SEQ_L)[None, :]
    for c0 in range(0, M, w):
        c1 = min(c0 + w, M)
        X = raw[starts[c0:c1, None] + off].transpose(0, 2, 1).reshape(-1, C.SEQ_L)
        yn, (loc, scale) = model(torch.from_numpy(X.astype(np.float32)).to(device))
        pred = model.denorm(yn, loc, scale).float().cpu().numpy()
        out[c0:c1] = pred.reshape(c1 - c0, N, C.PRE_LEN, len(qs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["kairos", "patchtst"], default="kairos")
    ap.add_argument("--city", default="MEL")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--kairos_root", default=os.path.join(BASE, "result", "kairos_l1_clean"))
    ap.add_argument("--patch_root", default=os.path.join(BASE, "result", "patchtst_quant_clean"))
    args = ap.parse_args()
    dev = C.torch_device(args.device)

    ds, raw, starts, keep, smean = C.load_city(args.city)
    n, m = C.check_protocol(args.city, ds, raw, starts)
    print(f"[{args.city}] 站 {n}  测试窗 {m}  历史 {C.SEQ_L}  目标 {C.PRE_LEN}")

    if args.what == "kairos":
        model, qs = load_kairos_quant(args.city, args.kairos_root, dev)
        pred = infer_kairos(model, raw, starts["te"], qs, dev)
        met = C.StreamingMetrics(raw, starts["te"], keep["te"], smean, qs)
        met.update_quantiles(0, pred)
        res = met.result()
        h24 = res["H24"]["crps_norm_sum"]
        print(f"[kairos+quant {args.city}] H24 crpsΣ = {h24:.6f}  (目标 0.354216 ±1e-4)")
        for k, v in res.items():
            print(f"  {k}: crpsΣ {v['crps_norm_sum']:.6f}  crpsΣsum "
                  f"{v['crps_sum_norm']:.6f}  crps_nz {v['crps_nz']:.6f}")
        if args.city == "MEL" and abs(h24 - 0.354216) > 1e-4:
            raise SystemExit(f"协议门失败：{h24:.6f} 与 0.354216 差超过 1e-4")
        print("协议门通过")

    else:
        from api.model.patchtst_quant import PatchTSTQuant
        qs = np.load(os.path.join(args.patch_root, args.city,
                                  "qlevels_patch_quant_pre24_fold6.npy"))
        model = PatchTSTQuant(seq_len=C.SEQ_L, pre_len=24, quantiles=list(qs)).to(dev)
        model.load_state_dict(torch.load(
            os.path.join(args.patch_root, args.city, "patch_quant_pre24_fold6.pth"),
            map_location=dev))
        model.eval()
        M, N = len(starts["te"]), raw.shape[1]
        met = C.StreamingMetrics(raw, starts["te"], keep["te"], smean, qs,
                                 horizons=(24,))
        off = np.arange(C.SEQ_L)[None, :]
        w = max(1, 512 // N)
        with torch.no_grad():
            for c0 in range(0, M, w):
                c1 = min(c0 + w, M)
                X = raw[starts["te"][c0:c1, None] + off].transpose(0, 2, 1).reshape(-1, C.SEQ_L)
                yn, (loc, scale) = model(torch.from_numpy(X.astype(np.float32)).to(dev))
                pred = model.denorm(yn, loc, scale).float().cpu().numpy()
                met.update_quantiles(c0, pred.reshape(c1 - c0, N, 24, len(qs)))
        r = met.result()["H24"]
        print(f"[patchtst+quant(pre24 ckpt) {args.city}] pre72窗 H24 crpsΣ "
              f"{r['crps_norm_sum']:.4f}  (pre24 窗旧值 0.3483，窗集不同不可直接比)")


if __name__ == "__main__":
    main()
