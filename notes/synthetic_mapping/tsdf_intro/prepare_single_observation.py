"""教师示意：单帧平面观测如何更新一行TSDF采样位置。非完整三维融合。"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parent
depth, truncation = 3.10, .05
z = np.arange(3.02, 3.161, .02)
sdf = depth-z
observed = sdf > -truncation
values = np.full(z.shape, np.nan)
values[observed] = np.minimum(sdf[observed]/truncation, 1.)
values[np.abs(values) < 1e-12] = 0
assert np.allclose(values[:-1], [1, 1, .8, .4, 0, -.4, -.8])
assert np.isnan(values[-1])

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(13, 6.3))
fig.suptitle('TSDF：一次深度观测，给附近的空间格子写入什么？', fontsize=21, y=.96)
fig.text(.5, .87, '教师设定：相机正对箱面，观测深度 3.10 米；截断距离设为 5 厘米',
         ha='center', fontsize=13, color='#42516a')
ax = fig.add_axes([.07, .20, .90, .59])
ax.set_xlim(-1.8, 8)
ax.set_ylim(-1.7, 2.6)
ax.axis('off')
ax.text(-1.65, 1.95, '相机在左侧', fontsize=12)
ax.annotate('', xy=(4, 1.8), xytext=(-1.5, 1.8),
            arrowprops=dict(arrowstyle='->', color='#273745', lw=2))
ax.text(1.0, 2.05, '朝箱面看 →', fontsize=12, ha='center')
ax.plot([4, 4], [-.12, 2.35], color='#173c30', linewidth=3, zorder=4)
ax.text(4.15, 2.26, '本帧观测表面', fontsize=12, color='#173c30')
ax.text(-1.7, .5, '格中存值', fontsize=12, va='center')
for j, (pos, val) in enumerate(zip(z, values)):
    color = '#dedede' if np.isnan(val) else ('#f5ecd6' if val == 0 else ('#a8dced' if val > 0 else '#f3be96'))
    ax.add_patch(Rectangle((j-.5, 0), 1, 1, facecolor=color, edgecolor='white', lw=2))
    label = '未知' if np.isnan(val) else ('0' if val == 0 else f'{val:+.1f}')
    ax.text(j, .5, label, ha='center', va='center', fontsize=15, fontweight='bold', zorder=5)
    ax.text(j, -.38, f'{pos:.2f}', ha='center', fontsize=12)
ax.text(3.5, -.88, '格子中心到相机的前向位置 z（米）', ha='center', fontsize=12)
ax.text(.8, 1.26, '蓝色：观测表面前方', ha='center', fontsize=12, color='#16627e')
ax.text(5.5, 1.26, '橙色：表面后方近邻', ha='center', fontsize=12, color='#99511b')
fig.text(.5, .15, '表面附近：存值 =（观测深度 − 格子前向位置）÷ 0.05 米；存值本身没有单位',
         ha='center', fontsize=12)
fig.text(.5, .085, '前方超过 5 厘米时记 +1；后方超过 5 厘米，本帧不更新（本例此前无观测，所以灰色未知）',
         ha='center', fontsize=11, color='#42516a')
fig.text(.5, .03, '只画同一观察方向附近的一行格子；负值是表面定位的局部表示，不代表看见了箱体内部。',
         ha='center', fontsize=11, color='#42516a')
fig.savefig(OUT/'single_observation.png', dpi=140)
plt.close(fig)
(OUT/'single_observation.json').write_text(json.dumps(dict(
    kind='teacher-set single-frame projective TSDF illustration',
    depth_m=depth, truncation_m=truncation, sample_z_m=z.tolist(),
    tsdf=[float(v) if np.isfinite(v) else None for v in values],
    weights=observed.astype(int).tolist(),
    note='Fixed camera; initially unknown samples. Not a complete 3D integration experiment.'
), ensure_ascii=False, indent=2), encoding='utf-8')
