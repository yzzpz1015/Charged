# -*- coding: utf-8 -*-
"""#11 TFT / #12 DeepAR（GluonTS 0.14.4，逐站单变量）。

* 训练序列只到训练期结束（按协议 2/1/3 计数切分），避免把 valid/test 喂进模型；
* 验证序列只覆盖验证期（``start`` 从验证窗最早起点前 72h 开始）；
* 测试：每个测试窗把 72h 历史作为 entry 的 ``target`` 送进 predictor，输出未来 72 步；
* DeepAR 用 ``num_parallel_samples=200`` 的**模型自身样本**（非负约束）→
  ``StreamingMetrics.update_samples``；TFT 直接训练 99 个分位 → ``update_quantiles``。

用法::

    python3 script/baselines/train_gluonts.py --method deepar --city MEL --device 0
    python3 script/baselines/train_gluonts.py --method tft --city MEL --device 0
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common as C  # noqa: E402

FREQ = "h"


from gluonts.torch.distributions import ISQFOutput as _ISQFOutput  # noqa: E402


class ISQFLoc0Output(_ISQFOutput):
    """ISQFOutput 的补丁版：只给 scale 时补 loc=0。

    GluonTS 0.14 的 ``ISQFOutput.distribution`` 在 loc=None 时会建出
    ``AffineTransform(loc=None)``；训练用 ``CRPS`` 时
    ``TransformedISQF.crps`` 调 ``_inverse`` 直接 `Tensor - None` 崩溃。
    DeepAR 正是只传 scale 的调用方式。
    """

    def distribution(self, distr_args, loc=None, scale=None):
        if scale is not None and loc is None:
            loc = torch.zeros_like(scale)
        return super().distribution(distr_args, loc, scale)


def _entries(raw, ds, row0, row1):
    """[row0,row1) 的逐站序列（start 用真实时间戳）。"""
    t0 = pd.Period(ds.time[int(row0)], freq=FREQ)
    return [{"start": t0, "target": raw[int(row0):int(row1), n].astype(np.float32)}
            for n in range(raw.shape[1])]


def _test_entries(raw, ds, starts):
    """每个测试窗 × 每站一条 72h 历史。返回 entries + (win_idx, station_idx)。"""
    out, meta = [], []
    for m, s in enumerate(starts):
        t0 = pd.Period(ds.time[int(s)], freq=FREQ)
        for n in range(raw.shape[1]):
            out.append({"start": t0, "target": raw[int(s):int(s) + C.SEQ_L, n].astype(np.float32)})
            meta.append((m, n))
    return out, meta


def build_estimator(args, n_quant):
    from gluonts.torch.model.deepar import DeepAREstimator
    from gluonts.torch.model.tft import TemporalFusionTransformerEstimator

    root = os.path.join(C.RESULT_ROOT, "ckpt", args.tag, args.city)
    os.makedirs(root, exist_ok=True)
    tkw = dict(max_epochs=args.epochs, accelerator="gpu", devices=[int(args.device)],
               enable_progress_bar=False, enable_model_summary=False, logger=False,
               gradient_clip_val=1.0, check_val_every_n_epoch=1,
               default_root_dir=root)   # 否则 ckpt 落到 cwd/checkpoints/ 且并发互相覆盖
    if args.method == "deepar":
        from gluonts.torch.distributions import StudentTOutput
        from gluonts.torch.modules.loss import CRPS, NegativeLogLikelihood

        if args.distr == "isqf":
            # ISQF：直接学分位函数，用 CRPS（积分 pinball）训练。
            # StudentT(nu=2) 在零膨胀数据上会退化成"位置≈0+重尾"，点预测塌陷。
            qk = [float(x) for x in args.isqf_qk.split(",")]
            distr_output = ISQFLoc0Output(num_pieces=args.isqf_pieces, qk_x=qk)
            loss = CRPS()
        else:
            distr_output = StudentTOutput()
            loss = NegativeLogLikelihood()
        return DeepAREstimator(
            freq=FREQ, prediction_length=C.PRE_LEN, context_length=C.SEQ_L,
            num_layers=args.layers, hidden_size=args.hidden, dropout_rate=0.1,
            num_parallel_samples=C.M_SAMPLE, nonnegative_pred_samples=True,
            distr_output=distr_output, loss=loss,
            batch_size=args.batch_size, num_batches_per_epoch=args.steps_per_epoch,
            patience=args.patience, trainer_kwargs=tkw,
            lr=args.lr, weight_decay=1e-4, scaling=True, default_scale=1.0,
        )
    qs = [round(i / (n_quant + 1), 4) for i in range(1, n_quant + 1)]
    return TemporalFusionTransformerEstimator(
        freq=FREQ, prediction_length=C.PRE_LEN, context_length=C.SEQ_L,
        quantiles=qs, num_heads=args.nheads, hidden_dim=args.hidden,
        variable_dim=args.hidden, dropout_rate=0.1,
        batch_size=args.batch_size, num_batches_per_epoch=args.steps_per_epoch,
        patience=args.patience, trainer_kwargs=tkw,
        lr=args.lr, weight_decay=1e-4,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["deepar", "tft"])
    ap.add_argument("--city", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--steps_per_epoch", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--nheads", type=int, default=4)
    ap.add_argument("--n_quant", type=int, default=99)
    ap.add_argument("--distr", choices=["studentt", "isqf"], default="studentt",
                    help="DeepAR 输出分布；isqf = 隐式分位（CRPS 训练）")
    ap.add_argument("--isqf_pieces", type=int, default=10)
    ap.add_argument("--isqf_qk", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--eval_batch_windows", type=int, default=8)
    ap.add_argument("--limit_test", type=int, default=0)
    ap.add_argument("--no_diag", action="store_true")
    args = ap.parse_args()
    # 结果目录名带分布标签，避免覆盖已判废的 StudentT 结果
    args.tag = args.method + ("_isqf" if args.method == "deepar"
                              and args.distr == "isqf" else "")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    ds, raw, starts, keep, smean = C.load_city(args.city)
    N, _ = C.check_protocol(args.city, ds, raw, starts)
    qs_np = np.arange(1, args.n_quant + 1) / (args.n_quant + 1)

    tr_end = int(starts["tr"][-1]) + C.SEQ_L + C.PRE_LEN
    va_end = int(starts["va"][-1]) + C.SEQ_L + C.PRE_LEN
    va_start = max(0, int(starts["va"][0]) - C.SEQ_L)
    train_data = _entries(raw, ds, 0, tr_end)
    valid_data = _entries(raw, ds, va_start, va_end)
    print(f"[{args.tag}/{args.city}/seed{args.seed}] 站 {N} "
          f"train 行 [0,{tr_end}) valid [{va_start},{va_end}) "
          f"test 窗 {len(starts['te'])}", flush=True)

    from gluonts.dataset.common import ListDataset
    train_ds = ListDataset(train_data, freq=FREQ)
    valid_ds = ListDataset(valid_data, freq=FREQ)
    est = build_estimator(args, args.n_quant)
    t0 = time.time()
    predictor = est.train(train_ds, validation_data=valid_ds)
    print(f"  trained in {time.time()-t0:.0f}s", flush=True)

    out_m, out_d, out_c = C.out_dirs(args.tag, args.city)
    torch.save(predictor.network.state_dict(), os.path.join(out_c, f"seed{args.seed}.pth"))

    te = starts["te"]
    if args.limit_test:
        te = te[:args.limit_test]
    entries, meta = _test_entries(raw, ds, te)
    met = C.make_metrics(args.city, raw, te, keep["te"][:len(te)], smean, qs_np,
                         with_diag=not args.no_diag)
    M = len(te)
    chunk_w = max(1, args.eval_batch_windows)
    per_chunk = chunk_w * N
    t1 = time.time()
    for c0 in range(0, len(entries), per_chunk):
        c1 = min(c0 + per_chunk, len(entries))
        fc = list(predictor.predict(
            ListDataset(entries[c0:c1], freq=FREQ)))
        m0 = meta[c0][0]
        w = min(chunk_w, M - m0)
        if args.method == "deepar":
            S = np.empty((w, N, C.PRE_LEN, C.M_SAMPLE), np.float32)
            for k, f in enumerate(fc):
                m, n = meta[c0 + k]
                smp = np.asarray(f.samples, np.float32)
                # GluonTS 0.14 返回 (num_samples, pred_len)，而 StreamingMetrics 要 [T, S]
                if smp.shape == (C.M_SAMPLE, C.PRE_LEN):
                    smp = smp.T
                S[m - m0, n] = smp[:C.PRE_LEN, :C.M_SAMPLE]
            met.update_samples(m0, S)
        else:
            Q = np.empty((w, N, C.PRE_LEN, args.n_quant), np.float32)
            for k, f in enumerate(fc):
                m, n = meta[c0 + k]
                Q[m - m0, n] = np.stack(
                    [f.quantile(float(q))[:C.PRE_LEN] for q in qs_np], -1)
            met.update_quantiles(m0, Q)
        if (c0 // per_chunk) % 5 == 0:
            done = min(c1 // N, M)
            el = time.time() - t1
            print(f"    {done}/{M} 窗  {el:.0f}s  ETA {el/max(done,1)*(M-done)/60:.0f}min",
                  flush=True)

    res = met.result()
    extra = dict(elapsed_s=round(time.time() - t0, 1), n_test_windows=len(te),
                 epochs=args.epochs, hidden=args.hidden, batch_size=args.batch_size,
                 num_parallel_samples=C.M_SAMPLE if args.method == "deepar" else 0,
                 n_quant=args.n_quant if args.method == "tft" else 0,
                 gluonts="0.14.4", freq=FREQ, distr=args.distr if args.method=="deepar" else "")
    C.save_metrics(args.tag, args.city, args.seed, res, extra)
    if not args.no_diag:
        C.save_diag(args.tag, args.city, args.seed, met)


if __name__ == "__main__":
    main()
