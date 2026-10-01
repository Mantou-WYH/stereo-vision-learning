"""解释回退：箱子固定，朝向固定，只比较两个左右站位的预期图像。"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
focal, cx, cy = 240., 200., 125.
front_z = 3.
world = np.array([[-.5, -.45, front_z], [.5, -.45, front_z],
                  [.5, .45, front_z], [-.5, .45, front_z], [0., 0., front_z]])


def project(camera_x):
    p = world - np.array([camera_x, 0., 0.])
    return np.column_stack([focal*p[:, 0]/p[:, 2]+cx, focal*p[:, 1]/p[:, 2]+cy])


observed = project(.5)
candidates = [project(0.), project(.5)]
assert np.array_equal(observed, candidates[1])
assert np.allclose(candidates[0]-observed, np.tile([40., 0.], (5, 1)))
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
fig, axes = plt.subplots(1, 3, figsize=(12, 5))
fig.subplots_adjust(left=.035, right=.985, top=.73, bottom=.25, wspace=.13)
fig.suptitle('箱子位置已知：哪种相机站位能拍出眼前这张照片？', fontsize=19, y=.95)
fig.text(.5, .84, '箱子和标记A都不动；相机朝向固定，这个简化例子只改变左右站位。',
         ha='center', fontsize=11)
for ax, uv, title in zip(axes, [observed]+candidates,
        ['已经拍到的新照片', '猜想①：相机站在箱子正前方', '猜想②：相机站得偏右']):
    ax.set_facecolor('#edf0f4')
    ax.fill(uv[:4, 0], uv[:4, 1], facecolor='#b58250', edgecolor='#67472e', lw=2)
    ax.axvline(cx, color='#5374a5', ls='--', lw=1.3)
    ax.scatter(*uv[4], color='#ffd335', edgecolor='#52371f', s=100, zorder=3)
    ax.annotate('A', uv[4], xytext=(9, -8), textcoords='offset points', fontsize=13)
    ax.set(xlim=(0, 400), ylim=(250, 0), title=title)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
fig.text(.19, .19, '这是要解释的观测，始终不变', ha='center', fontsize=11)
fig.text(.51, .19, '画出的箱子在中央：对不上', ha='center', fontsize=11, color='#b54d2a')
fig.text(.83, .19, '画出的箱子偏左：能对上', ha='center', fontsize=11, color='#17775e')
fig.text(.5, .07, '中、右两图是按候选站位计算的“应该拍到的样子”；蓝色虚线是图像中心。',
         ha='center', fontsize=10, color='#42516a')
fig.savefig(OUT/'position_intuition.png', dpi=150)
plt.close(fig)
(OUT/'position_intuition_checks.json').write_text(json.dumps(dict(
    kind='Simplified 1D position hypothesis comparison, not full PnP.',
    known_scene='Stationary front rectangle Z=3m, fixed camera orientation and intrinsics.',
    hidden_camera_x_m=.5, candidate_camera_x_m=[0., .5],
    observed_marker_uv_px=observed[4].tolist(),
    candidate_marker_uv_px=[a[4].tolist() for a in candidates],
    checks='Right candidate exactly matches fixed observation; centered candidate shifted 40px.',
    evidence='Teacher preparation only; understanding pending.'), indent=2), encoding='utf-8')
print('简化站位对照图已保存。')
