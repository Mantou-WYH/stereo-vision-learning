"""两组教师候选位姿的投影与独立固定观测比较；不是PnP求解。"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from two_view_mapping_lab import build_views, FX, FY, CX, CY

OUT = Path(__file__).resolve().parent
data = np.load(OUT/'correspondences.npz')
points = data['world_points_m'].copy()
observed = data['second_uv_px'].copy()
first, second = build_views()


def project(camera, rotation):
    p = (points-camera) @ rotation
    assert np.all(p[:, 2] > 0)
    return np.column_stack([FX*p[:, 0]/p[:, 2]+CX, FY*p[:, 1]/p[:, 2]+CY])


predictions = [project(first['camera'], first['R_wc']),
               project(second['camera'], second['R_wc'])]
assert np.allclose(predictions[0], data['first_uv_px'])
assert np.allclose(predictions[1], observed, atol=1e-12)
errors = [np.linalg.norm(p-observed, axis=1) for p in predictions]
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
fig, axes = plt.subplots(1, 2, figsize=(12, 6.8))
fig.subplots_adjust(left=.07, right=.97, bottom=.29, top=.77, wspace=.28)
fig.suptitle('新照片固定不动：换候选位姿，看预测点能否落到观测点上', fontsize=19, y=.96)
fig.text(.5, .875, '黑色实心点：图中已观察到的位置；彩色空心圈：候选位姿算出的投影',
         ha='center', fontsize=12)
for ax, predicted, color, title in zip(axes, predictions, ['#eb6f2f', '#00b580'],
        ['候选①：认为相机还在第1帧的位置和朝向', '候选②：使用生成本例观测的真实位姿']):
    ax.imshow(second['picture'], interpolation='nearest', alpha=.7)
    ax.scatter(observed[:, 0], observed[:, 1], s=30, c='#162435', label='固定的图像观测', zorder=4)
    ax.scatter(predicted[:, 0], predicted[:, 1], s=140, facecolors='none',
               edgecolors=color, linewidths=2.5, label='候选位姿的预测', zorder=5)
    for label, o, p in zip('ABCD', observed, predicted):
        ax.plot([o[0], p[0]], [o[1], p[1]], color=color, lw=1.7)
        ax.annotate(label, o, xytext=(-13, -12), textcoords='offset points',
                    fontsize=12, color='#162435')
    ax.set(title=title, xlabel='u（像素）', ylabel='v（像素）')
    ax.legend(loc='lower right', fontsize=9)
fig.text(.5, .195, f'A：观测(71.3, 70.8)；候选①预测(88.9, 68.0)，相差{errors[0][0]:.1f}像素；候选②重合。',
         ha='center', fontsize=12)
fig.text(.5, .12, '已知三维点、图像观测、相机内参均固定；改变的是候选相机的位置和朝向。',
         ha='center', fontsize=11)
fig.text(.5, .04, '两个候选均由教师给定；第二候选来自模拟真值，本步未运行优化、PnP或特征匹配。',
         ha='center', fontsize=10, color='#42516a')
fig.savefig(OUT/'candidate_projection.png', dpi=150)
plt.close(fig)
np.savez_compressed(OUT/'candidate_projection.npz', world_points_m=points,
                    observed_uv_px=observed, unchanged_pose_predicted_uv_px=predictions[0],
                    true_pose_predicted_uv_px=predictions[1],
                    unchanged_pose_error_px=errors[0], true_pose_error_px=errors[1])
report = dict(kind='Teacher-given candidate projections; no pose optimization.',
              observed_source='Saved second_uv_px, fixed before candidate projection.',
              unchanged_pose=dict(camera_m=first['camera'].tolist(), R_wc=first['R_wc'].tolist(),
                                  error_px=errors[0].tolist(), squared_error_sum_px2=float((errors[0]**2).sum())),
              true_pose=dict(camera_m=second['camera'].tolist(), R_wc=second['R_wc'].tolist(),
                             error_px=errors[1].tolist()),
              evidence='Teacher validation; student understanding and execution pending.')
(OUT/'candidate_checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
