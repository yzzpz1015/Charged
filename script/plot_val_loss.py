"""Plot val_loss curves for all experiments."""
import re, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_log(path):
    epochs, trains, vals = [], [], []
    stages = []
    with open(path) as f:
        for line in f:
            m = re.search(r'Epoch\s+(\d+):\s+train=([\d.]+),\s+val=([\d.]+)', line)
            if m:
                epochs.append(int(m.group(1)))
                trains.append(float(m.group(2)))
                vals.append(float(m.group(3)))
            if 'Switched to stage' in line:
                m2 = re.search(r'at epoch (\d+)', line)
                if m2:
                    s = line[line.find('stage') + 6]
                    if s.isdigit():
                        stages.append((int(m2.group(1)), int(s)))
    return epochs, trains, vals, stages

def add_plot(ax, path, label, color, show_stages=False):
    if not os.path.exists(path):
        print(f"  SKIP {label}: no file")
        return
    ep, tr, vl, stages = parse_log(path)
    if not ep:
        print(f"  SKIP {label}: no data")
        return
    ax.plot(ep, vl, color=color, linewidth=1.5, label=f'{label} val')
    ax.plot(ep, tr, color=color, alpha=0.2, linewidth=1, label=f'{label} train')
    if show_stages and stages:
        sc = {1: 'green', 2: 'orange', 3: 'red'}
        for s_ep, s_num in stages:
            ax.axvline(x=s_ep, color=sc.get(s_num, 'gray'), linestyle='--', alpha=0.4)
            ax.text(s_ep, vl[0]*1.05, f'S{s_num}', fontsize=7, color=sc.get(s_num, 'gray'), ha='right', rotation=90)
    print(f"  {label}: {len(ep)}ep, val {vl[0]:.3f} -> {vl[-1]:.3f}")

RES = '/home/data/yzzpz/Charged/result/univariate/MEL'
EV = f'{RES}/ev_vlm'

# Left: EV-VLM staged (3 pre_len)
entries_left = [
    ('EV-VLM staged pre=6',  f'{EV}/volume-None-site-72-6-6#2/logging.txt', '#1f77b4'),
    ('EV-VLM staged pre=12', f'{EV}/volume-None-site-72-12-6#1/logging.txt', '#ff7f0e'),
    ('EV-VLM staged pre=24', f'{EV}/volume-None-site-72-24-6#1/logging.txt', '#2ca02c'),
]

# Right: pre_len=6 all models + gate ablation
entries_right = [
    ('EV-VLM staged',   f'{EV}/volume-None-site-72-6-6#2/logging.txt', '#1f77b4'),
    ('EV-VLM e2e',      f'{EV}/volume-None-site-72-6-6/logging.txt', '#d62728'),
    ('FreTS',           f'{RES}/frets/volume-None-site-72-6-6/logging.txt', '#9467bd'),
    ('PatchTST',        f'{RES}/patchtst/volume-None-site-72-6-6/logging.txt', '#8c564b'),
    # Gate + RelationGAT experiments
    ('GAT gate=0',      '/tmp/ev_vlm_gat.log', '#e377c2'),
    ('GAT gate=-1',     '/tmp/ev_vlm_gate_n1.log', '#bcbd22'),
    ('GAT gate=0 #2',   '/tmp/ev_vlm_gate_0.log', '#17becf'),
    ('GAT gate=-1 #2',  '/tmp/ev_vlm_gat_n1.log', '#ff9896'),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

print("=== Left: EV-VLM staged ===")
for label, path, color in entries_left:
    add_plot(ax1, path, label, color, show_stages=True)
ax1.set_title('EV-VLM Staged (pre_len=6/12/24)', fontsize=12)
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)

print("\n=== Right: pre_len=6 all models ===")
for label, path, color in entries_right:
    add_plot(ax2, path, label, color)
ax2.set_title('pre_len=6 All Models', fontsize=12)
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)

plt.suptitle('EV-VLM Results (MEL, seq_l=72)', fontsize=14, fontweight='bold')
plt.tight_layout()
out = '/home/data/yzzpz/Charged/result/val_loss_comparison.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'\nSaved to {out}')
