# -*- coding: utf-8 -*-
"""把城市宏观图 + 单站连续窗口图 + 预测分布形状图整合成一个自包含 HTML。

零膨胀讨论用的参考材料：数据长什么样、预测分布长什么样。
图片全部 base64 内嵌，生成的 HTML 单文件即可转发。

用法::

    python script/build_zero_inflation_report.py
    # 输出 docs/zero_inflation_report.html

依赖 ``/tmp/zi_stats.json`` 里的逐城统计；该文件由一次性脚本生成。
若不存在，脚本会现场重算（约 1 分钟）。
"""
import base64
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f'{BASE}/docs/zero_inflation_report.html'
MACRO_DIR = f'{BASE}/result/figs_zero_inflation'
SERIES_DIR = f'{BASE}/result/figs'
STATS = '/tmp/zi_stats.json'

CITIES = ['AMS', 'LOA', 'JHB', 'MEL', 'SPO', 'SZH']
CITY_CN = {'AMS': 'Amsterdam', 'LOA': 'Los Angeles', 'JHB': 'Johannesburg',
           'MEL': 'Melbourne', 'SPO': 'São Paulo', 'SZH': 'Shenzhen'}
SERIES = {
    'JHB': ['prob_series_JHB_H24_s120+168_st7.png',
            'prob_series_JHB_H24_s120+168_st11.png'],
    'LOA': ['prob_series_LOA_H24_s120+168_st89.png',
            'prob_series_LOA_H24_s120+168_st60.png'],
    'MEL': ['prob_series_MEL_H24_s120+168_st3.png',
            'prob_series_MEL_H24_s120+168_st51.png'],
    'SPO': ['prob_series_SPO_H24_s120+168_st31.png',
            'prob_series_SPO_H24_s120+168_st24.png'],
    'SZH': ['prob_series_SZH_H24_s120+168_st240.png',
            'prob_series_SZH_H24_s120+168_st195.png'],
}
SPO_EXTRA = ['prob_series_SPO_H24_st5.png',
             'prob_series_SPO_H24_st21.png',
             'prob_series_SPO_H24_st31.png']
SHAPES = [
    ('prob_shape_35_MEL_H24_seed0.png',
     'MEL：③ 分位头 vs ⑤ 吸收态扩散，12 个格子（上 6 格真值为 0，下 6 格真值 >0）。'),
    ('prob_shape_SPO_H24.png',
     'SPO：① 点+残差池 / ③ 分位头 / ④ 整块扩散，随机抽 12 格，含真值为 0 的格子。'),
    ('prob_shape_SPO_H24_oracle.png',
     'SPO：满分门卫（oracle）口径，只比较“画师”本身的分布形状。'),
]


def img64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


def load_stats():
    if os.path.exists(STATS):
        return json.load(open(STATS))
    sys.path.insert(0, BASE)
    sys.path.insert(0, f'{BASE}/script')
    import numpy as np
    import prob_baseline as pb
    import kairos_prob_eval as ke
    from api.dataset.common import EVDataset
    ser = {'JHB': [7, 11], 'LOA': [60, 89], 'MEL': [3, 51],
           'SPO': [24, 31], 'SZH': [195, 240]}
    out = {}
    for c in CITIES:
        ds = EVDataset(feature='volume', auxiliary='None',
                       data_path=f'{BASE}/data/{c}_remove_const/')
        ds.split_cross_validation(fold=6, total_fold=6,
                                  train_ratio=0.8, valid_ratio=0.1)
        raw = ds.feat
        tot, act = raw.sum(1), (raw > 0).sum(1)
        rec = {'N': int(raw.shape[1]), 'T': int(raw.shape[0]),
               'zero_share': float((raw == 0).mean()),
               'total_zero_hours': int((tot == 0).sum()),
               'total_min': float(tot.min()),
               'active_med': float(np.median(act)),
               'splits': {}}
        for sid, name in [(0, 'train'), (1, 'valid'), (2, 'test')]:
            m = ds.split_of_row == sid
            rec['splits'][name] = {
                'hours': int(m.sum()),
                'zero_share': float((raw[m] == 0).mean()),
                'total_min': float(tot[m].min()),
                'active_med': float(np.median(act[m])),
            }
        zs = (raw == 0).mean(0)
        rec.update(station_zero_med=float(np.median(zs)),
                   station_zero_p90=float(np.percentile(zs, 90)),
                   station_gt50=float((zs > 0.5).mean()))
        rec['series'] = []
        if c in ser:
            y0, t0 = pb.load(c)
            ok, trm, *_ = pb.split_rows(y0, t0)
            cols, _ = pb.live_cols(y0, trm)
            y = y0[:, cols]
            starts = ds.window_starts(2, 72, 24)
            j = 23
            lag = pb.LAG * int(np.ceil((j + 1) / pb.LAG))
            s_te = np.asarray(starts) + 72 + j
            m_te = ok[s_te - lag]
            sel = np.arange(120, min(120 + 168, len(m_te)))
            sel = sel[m_te[sel]]
            for st in ser[c]:
                rec['series'].append({
                    'sid': st,
                    'win_zero': float((y[s_te[sel], st] == 0).mean()),
                    'overall_zero': float((y[:, st] == 0).mean()),
                    'win_mean': float(y[s_te[sel], st].mean()),
                    'win_max': float(y[s_te[sel], st].max()),
                    'n': int(len(sel)),
                })
        out[c] = rec
    json.dump(out, open(STATS, 'w'), ensure_ascii=False, indent=1)
    return out


def pct(x):
    return f'{100 * x:.1f}%'


def stat_table(S):
    rows = []
    for c in CITIES:
        r = S[c]
        note = ' <span class="tag">单列</span>' if c == 'AMS' else ''
        rows.append(
            f'<tr><td><b>{c}</b>{note}</td><td>{r["N"]}</td>'
            f'<td>{pct(r["zero_share"])}</td>'
            f'<td>{pct(r["station_zero_med"])}</td>'
            f'<td>{pct(r["station_zero_p90"])}</td>'
            f'<td>{100 * r["station_gt50"]:.1f}%</td>'
            f'<td>{r["total_min"]:,.0f}</td>'
            f'<td>{r["active_med"]:.0f}</td>'
            f'<td>{pct(r["splits"]["test"]["zero_share"])}</td></tr>')
    return '\n'.join(rows)


def series_block(c, S):
    files = SERIES[c]
    stats = {s['sid']: s for s in S[c]['series']}
    figs = []
    for f in files:
        sid = int(f.rsplit('_st', 1)[1].split('.')[0])
        s = stats.get(sid, {})
        kind = '稀疏/零多的样例站' if s.get('win_zero', 0) >= 0.5 else '连续/零少的样例站'
        figs.append(f'''
<figure>
  <img src="data:image/png;base64,{img64(f"{SERIES_DIR}/{f}")}" alt="{c} station {sid}">
  <figcaption><b>{c} · 站序 {sid}</b>（{kind}）——该 168h 窗口内零值率
    <b>{pct(s.get("win_zero", 0))}</b>，全期 {pct(s.get("overall_zero", 0))}，
    窗口均值 {s.get("win_mean", 0):.1f} kWh/h，峰值 {s.get("win_max", 0):.1f} kWh/h。
    <span class="path">{f}</span></figcaption>
</figure>''')
    return '\n'.join(figs)


def main():
    S = load_stats()
    macro = {}
    for c in CITIES:
        macro[c] = img64(f'{MACRO_DIR}/{c.lower()}_macro.png')

    macro_blocks = []
    for c in CITIES:
        extra = ''
        if c == 'AMS':
            extra = ('<p class="note">AMS 测试期约 95.9% 的站处于“数值冻结”（非零但不再变化），'
                     '因此只单列，不进入跨城平均。</p>')
        macro_blocks.append(f'''
<div class="city-block">
  <h3>{c} · {CITY_CN[c]}</h3>
  <figure><img src="data:image/png;base64,{macro[c]}" alt="{c} macro">
  <figcaption>六个月的<b>城市级</b>时序：上=全部站求和；中=有量站数；下=站均量。
    粗线为日均。<span class="path">result/figs_zero_inflation/{c.lower()}_macro.png</span>
  </figcaption></figure>{extra}
</div>''')

    series_blocks = []
    for c in ['LOA', 'JHB', 'MEL', 'SPO', 'SZH']:
        series_blocks.append(f'''
<div class="city-block">
  <h3>{c} · {CITY_CN[c]}</h3>
  {series_block(c, S)}
</div>''')

    extra_spo = '\n'.join(f'''
<figure><img src="data:image/png;base64,{img64(f"{SERIES_DIR}/{f}")}" alt="{f}">
<figcaption><span class="path">{f}</span>（随机抽格版本，非连续 168h）</figcaption></figure>'''
                        for f in SPO_EXTRA)

    shape_blocks = '\n'.join(f'''
<figure>
  <img src="data:image/png;base64,{img64(f"{SERIES_DIR}/{f}")}" alt="{f}">
  <figcaption>{desc}<br><span class="path">{f}</span></figcaption>
</figure>''' for f, desc in SHAPES)

    html = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EV 充电量：零膨胀与预测形状参考</title>
<style>
:root {{ --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#8a8984;
         --line:#e6e5e1; --blue:#2a78d6; --orange:#eb6834; --green:#1baf7a; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink);
  font:15px/1.65 -apple-system,"Segoe UI","Noto Sans CJK SC","Noto Sans CJK JP",sans-serif; }}
header {{ max-width:1180px; margin:0 auto; padding:48px 28px 20px; }}
h1 {{ font-size:30px; line-height:1.25; margin:0 0 8px; letter-spacing:-.02em; }}
.sub {{ color:var(--ink2); margin:0 0 22px; }}
.callout {{ background:#fff; border:1px solid var(--line); border-left:3px solid var(--blue);
  border-radius:6px; padding:16px 20px; margin:22px 0; }}
.callout h2 {{ font-size:16px; margin:0 0 8px; }}
.callout ol {{ margin:8px 0 0; padding-left:22px; }}
.callout li {{ margin:6px 0; }}
nav {{ max-width:1180px; margin:0 auto 10px; padding:0 28px; }}
nav a {{ color:var(--blue); text-decoration:none; margin-right:18px; font-size:14px; }}
nav a:hover {{ text-decoration:underline; }}
main {{ max-width:1180px; margin:0 auto; padding:0 28px 70px; }}
section {{ margin-top:46px; }}
h2 {{ font-size:22px; margin:0 0 6px; padding-bottom:8px; border-bottom:1px solid var(--line); }}
h3 {{ font-size:17px; margin:30px 0 10px; }}
.lede {{ color:var(--ink2); margin:10px 0 18px; }}
table {{ width:100%; border-collapse:collapse; background:#fff; font-size:13.5px; margin:12px 0 8px; }}
th,td {{ padding:9px 10px; border-bottom:1px solid var(--line); text-align:right; }}
th:first-child, td:first-child {{ text-align:left; }}
thead th {{ color:var(--ink2); font-weight:600; background:#f6f5f2; }}
tbody tr:hover {{ background:#faf9f6; }}
.tag {{ font-size:11px; color:var(--muted); border:1px solid var(--line);
  border-radius:3px; padding:0 4px; margin-left:4px; }}
.city-block {{ margin-top:20px; }}
figure {{ margin:18px 0 26px; background:#fff; border:1px solid var(--line);
  border-radius:6px; padding:10px; }}
figure img {{ display:block; width:100%; height:auto; border-radius:3px; }}
figcaption {{ color:var(--ink2); font-size:13px; padding:9px 6px 2px; }}
.path {{ color:var(--muted); font-family:ui-monospace,Menlo,Consolas,monospace; font-size:11.5px; }}
.note {{ color:var(--ink2); font-size:13px; background:#f6f5f2; padding:10px 14px;
  border-radius:5px; margin:4px 0 0; }}
.legend-help {{ color:var(--ink2); font-size:13.5px; }}
footer {{ max-width:1180px; margin:0 auto; padding:24px 28px 60px; color:var(--muted);
  font-size:12px; border-top:1px solid var(--line); }}
@media print {{ body {{ font-size:11pt; }} figure {{ break-inside:avoid; }}
  section {{ break-before:auto; }} nav {{ display:none; }} }}
</style></head>
<body>
<header>
  <h1>EV 充电量预测：零膨胀与预测形状参考</h1>
  <p class="sub">数据长什么样 · 零在哪个尺度出现 · 现有概率预测给出什么形状的分布<br>
  2026-10-22 · clean 口径（停摆行剔除，可用行按计数 2/1/3 划分）</p>

  <div class="callout">
    <h2>先说结论（给零膨胀建模参考）</h2>
    <ol>
      <li><b>零只在站级出现，城市级几乎看不到。</b>六城的“全城合计”都没有任何一小时为 0；
        单个站可以连续几十小时为 0，但其它站同时在线，求和后总是一条正值曲线。</li>
      <li><b>站与站差异极大。</b>同一城市里，中位站的零值率从 0%（SZH）到 30.7%（MEL）；
        站级零值率 p90 在 JHB/LOA/SPO 达 39%~45%，样例站窗口内零值率 0%~73%。</li>
      <li><b>零是真实空闲，不是缺失。</b>volume 与 duration 同步为 0；停摆行（该小时 &gt;80% 站同时为 0）
        已整体剔除，不在训练/验证/测试中。图里的 0 是站点没车充，不是没上报。</li>
      <li><b>现有分布形状对零的处理不一致。</b>分位头给出的是连续区间，在真值为 0 的格子上仍覆盖正的量级；
        扩散类在部分零格会向 0 集中，但不同实现差异明显（见第 4 节）。这正是“零格应不应该有原子、
        以及零格该不该和正格共用一套分布”的问题。</li>
    </ol>
  </div>
</header>

<nav>
  <a href="#overview">1 · 零值统计</a>
  <a href="#macro">2 · 城市宏观</a>
  <a href="#series">3 · 单站 168h</a>
  <a href="#shape">4 · 预测形状</a>
</nav>

<main>
<section id="overview">
  <h2>1 · 零值统计（当前 clean 数据，剔退化站后）</h2>
  <p class="lede">“零值率”按“站 × 小时”计。N 为训练期退化站剔除后的站数
    （死站/常数平台按训练期统计剔除，与基线同口径）。</p>
  <table>
    <thead><tr>
      <th>城市</th><th>站数 N</th><th>整体零值率</th><th>站级零值率中位</th>
      <th>站级零值率 p90</th><th>零值率 &gt;50% 的站</th>
      <th>全城合计最小值<br>(kWh/h)</th><th>活跃站数中位</th><th>测试段零值率</th>
    </tr></thead>
    <tbody>
{stat_table(S)}
    </tbody>
  </table>
  <p class="note">六城的“全城合计 = 0”小时数均为 <b>0</b>。AMS 测试期约 95.9% 的站数值冻结
    （非零但不再变化），只单列、不参与跨城平均。SZH 站数多、单站量小，聚合后零值率极低。</p>
</section>

<section id="macro">
  <h2>2 · 城市宏观：为什么聚合后看不到零</h2>
  <p class="lede">每张图三行：① 全部站求和（细线=小时，粗线=日均）；② 有量站数（volume&gt;0）；
    ③ 站均量 = 总量 ÷ 有量站数。灰底是停摆行（&gt;80% 站同时为 0，按规则不进划分）。
    中行的波动就是被聚合掩盖掉的站级零：即使总量曲线平稳，也常有 3~10 个站同时在 0。</p>
{''.join(macro_blocks)}
</section>

<section id="series">
  <h2>3 · 单站 168 小时连续窗口（数据 + 三种预测）</h2>
  <p class="lede">每城两个站：一个爱归零、一个几乎不归零；横轴 168 个连续目标小时，
    每点都是提前 24h 的预测。三行分别是 ① 点预测+残差池、③ 分位头、④ 整块扩散；
    黑实线=真值，灰虚线=主干点预测。④只有 24 条采样链（评测口径），①③各 200 条，
    所以绿色点稀不代表区间更窄。文件名中的 <code>s120+168</code> 表示从第 120 个测试窗口起取 168 个。</p>
{''.join(series_blocks)}
  <h3>SPO 的随机抽格版本（补充）</h3>
{extra_spo}
</section>

<section id="shape">
  <h2>4 · 预测分布形状（零格 vs 正格）</h2>
  <p class="lede">每个小面板 = 一个“站 × 窗口 × 提前量”的格子。点=采样样本；
    粗/细横条=25~75% / 10~90% 区间；白条=中位数；黑色箭头=真值（▽）。
    先看真值为 0 的格子：分布有没有在 0 处形成原子、还是把质量铺到正数；
    再看真值 &gt;0 的格子：区间是否覆盖、右尾是否足够。</p>
{shape_blocks}
  <p class="note">观察：③ 分位头在真值为 0 的格子上通常给出一段连续的正区间，没有 0 原子；
    扩散类在部分零格会把样本堆压到 0 附近（如 SPO 窗 115），但并非所有实现都如此。
    这解释了为什么“边际 CRPS”与“只看真值&gt;0 的条件 CRPS”会给出不同结论：
    零格的质量怎么放，会单独影响一大块指标。</p>
</section>
</main>

<footer>
  数据：<code>data/&lt;city&gt;_remove_const/</code>；分窗：seq_l=72 / pre_len=24 / fold=6；
  城市宏观图由 <code>script/viz_city_macro.py</code> 于 2026-10-22 用当前数据重画；
  单站与形状图由 <code>script/prob_viz.py</code> / <code>prob_viz_multi.py</code> 生成。
  本页为自包含 HTML，图片已内嵌，可直接转发或打印为 PDF。
</footer>
</body></html>'''
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'[html] {OUT}  {os.path.getsize(OUT) / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
