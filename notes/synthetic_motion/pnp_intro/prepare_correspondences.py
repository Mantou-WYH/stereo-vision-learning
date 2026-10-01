"""L09输入示意：既有三维标记与下一幅图像中的二维位置；不求解位姿。"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from two_view_mapping_lab import build_views, FX, FY, CX, CY

OUT = Path(__file__).resolve().parent
points = np.array([[-.5, -.35, 3.1], [.4, -.35, 3.1],
                   [.4, .35, 3.1], [-.5, .35, 3.1]])
views = build_views()
uvs = []
for view in views:
    camera_points = (points-view['camera']) @ view['R_wc']
    uv = np.column_stack([FX*camera_points[:, 0]/camera_points[:, 2]+CX,
                          FY*camera_points[:, 1]/camera_points[:, 2]+CY])
    assert np.all(camera_points[:, 2] > 0)
    pixels = np.floor(uv+.5).astype(int)
    assert np.all(view['face'][pixels[:, 1], pixels[:, 0]] == 4)
    reconstructed = np.column_stack([(uv[:, 0]-CX)*camera_points[:, 2]/FX,
                                      (uv[:, 1]-CY)*camera_points[:, 2]/FY,
                                      camera_points[:, 2]]) @ view['R_wc'].T + view['camera']
    assert np.allclose(reconstructed, points)
    uvs.append(uv)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
fig, axes = plt.subplots(1, 2, figsize=(12, 6.5))
fig.subplots_adjust(left=.08, right=.96, bottom=.3, top=.76, wspace=.28)
fig.suptitle('同一批固定标记：已有三维位置 → 新照片中的二维位置', fontsize=20, y=.96)
fig.text(.5, .87, '模拟箱子保持静止，相机换了位置；A—D表示相同物理标记', ha='center', fontsize=12)
colors = ['#ffcc33', '#e953b6', '#64efbf', '#ffffff']
for ax, view, uv, title in zip(axes, views, uvs,
                              ['第1帧：这些点已有三维坐标', '第2帧：找出同名点的像素位置']):
    ax.imshow(view['picture'], interpolation='nearest')
    for label, color, pixel in zip('ABCD', colors, uv):
        ax.scatter(*pixel, s=95, facecolor=color, edgecolor='#24354b', zorder=3)
        ax.annotate(label, pixel, xytext=(7, -13), textcoords='offset points', fontsize=12,
                    color='#142034', bbox=dict(facecolor='white', alpha=.9, edgecolor='none', pad=1))
    ax.set(title=title, xlabel='u（像素）', ylabel='v（像素）')
fig.text(.5, .195, f'A的已知地图坐标：(-0.50, -0.35, 3.10) 米  →  第2帧A：({uvs[1][0, 0]:.1f}, {uvs[1][0, 1]:.1f}) 像素',
         ha='center', fontsize=13)
fig.text(.5, .115, 'B、C、D同样配成“一个三维点 → 一个二维位置”；再加相机内参，才构成PnP的输入。',
         ha='center', fontsize=11)
fig.text(.5, .035, '本图三维点和对应关系由教师设定；已知相机位姿仅用于生成观测，本步尚未估计相机运动。',
         ha='center', fontsize=10, color='#42516a')
fig.savefig(OUT/'correspondences.png', dpi=150)
plt.close(fig)
np.savez_compressed(OUT/'correspondences.npz', world_points_m=points,
                    first_uv_px=uvs[0], second_uv_px=uvs[1],
                    K=np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1.]]))
report = dict(kind='Teacher-generated four planar correspondences, no PnP solve or feature tracker.',
              checks='Positive camera Z; visible front pixels in both views; backprojection roundtrip.',
              point_source='Teacher-set map positions, not student reconstructed points.',
              second_image_source='Independent virtual camera view, not an actual new ETH3D frame.',
              next_goal='Use a candidate pose to project these known points and compare observed image positions.')
(OUT/'checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print('对应关系示意与数据已保存；没有执行PnP求解。')
