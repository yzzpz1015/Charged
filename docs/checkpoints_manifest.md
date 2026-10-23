# Charged checkpoints

Checkpoints for **probabilistic forecasting of hourly station-level EV charging volume**
(Kairos_23m backbone; clean split, `seq_l=72`; H=24/48/72; crpsΣ main metric).

Code: https://github.com/yzzpz1015/Charged
Data: https://huggingface.co/datasets/yzzpz/Charged-data

## Contents

| Directory | Files | Role |
|---|---|---|
| `kairos_l1_clean/<CITY>/plain_pre72_fold6.pth` | 6 cities | ① point prediction |
| `kairos_l1_clean/<CITY>/quant_pre72_fold6.pth` | 6 cities | ③ per-station quantile head (main method) |
| `patchtst_quant_clean/<CITY>/patch_quant_pre24_fold6.pth` | MEL, SPO | backbone × head ablation |
| `csdi_clean/<CITY>_csdi.pth` | 5 cities | ④ block CSDI |
| `csdi_clean/<CITY>_zidiff.pth` | 5 cities | ⑤ absorbing-state diffusion |
| `timediff_nomix_clean/<CITY>_timediff.pth` | MEL, SPO | TimeDiff reference (no future mixup) |

Cities: `LOA JHB MEL SPO SZH AMS`. AMS is reported separately
(~95.9% of stations frozen in the test period) and excluded from cross-city averages.

## Notes

- The Kairos backbone (`mldi-lab/Kairos_23m`) is downloaded automatically by the training code.
- `result/timediff_clean/` (future-mixup version) is intentionally **not** included:
  its sampler collapses (MPIW50 ≈ 0.02) and its metrics are invalid.
- Evaluation protocol and per-city numbers: see the GitHub README.
