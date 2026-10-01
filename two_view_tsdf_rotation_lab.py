"""L08：理想深度不变，仅第二帧估计朝向绕地图Y轴偏1度。"""
import argparse
import json
from pathlib import Path
import numpy as np
from two_view_mapping_lab import build_views
from two_view_tsdf_lab import make_grid, integrate, extract, STEP, ORIGIN, MU
from two_view_tsdf_noise_lab import front_surface

OUT = Path(__file__).resolve().parent / 'notes/synthetic_mapping/tsdf_rotation_error'
ANGLE_DEG = 1.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    views = build_views()
    angle = np.deg2rad(ANGLE_DEG)
    error_R = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                        [-np.sin(angle), 0, np.cos(angle)]])
    assert np.allclose(error_R.T @ error_R, np.eye(3))
    assert np.isclose(np.linalg.det(error_R), 1)
    grid = make_grid()
    surfaces, masks, counts = [], [], []
    for name, wrong in [('correct', False), ('wrong_rotation', True)]:
        field = np.zeros(grid.shape[:-1], np.float32)
        weights = np.zeros(field.shape, np.uint8)
        for n, view in enumerate(views, 1):
            observation = dict(view)
            observation['R_wc'] = view['R_wc'].copy()
            if n == 2 and wrong:
                observation['R_wc'] = error_R @ view['R_wc']
            assert np.array_equal(observation['depth'], view['depth'], equal_nan=True)
            assert np.array_equal(observation['camera'], view['camera'])
            integrate(observation, grid, field, weights)
            if n == 1:
                if not wrong:
                    first_field, first_weights = field.copy(), weights.copy()
                else:
                    assert np.array_equal(field, first_field)
                    assert np.array_equal(weights, first_weights)
        z, valid = front_surface(field, weights)
        surfaces.append(z.copy()); masks.append(valid.copy())
        vertices, faces = extract(field, weights)
        counts.append(dict(vertices=len(vertices), triangles=len(faces)))
        np.savez_compressed(OUT / f'{name}.npz', tsdf=field, weights=weights,
                            vertices_m=vertices, triangles=faces, front_z_m=z,
                            front_valid=valid, origin_m=ORIGIN, spacing_m=STEP,
                            truncation_m=MU)
    second = views[1]
    wrong_points = second['camera_points'] @ (error_R @ second['R_wc']).T + second['camera']
    assert np.allclose(np.linalg.norm(wrong_points-second['camera'], axis=1),
                       np.linalg.norm(second['world_points']-second['camera'], axis=1))
    normal = error_R @ np.array([0., 0., 1.])
    front = second['face'][second['valid']] == 4
    anchor = error_R @ (np.array([0., 0., 3.1])-second['camera']) + second['camera']
    assert np.max(abs((wrong_points[front]-anchor) @ normal)) < 1e-12
    x, y = grid[:, :, 0, 0], grid[:, :, 0, 1]
    roi = (abs(x) < .6) & (abs(y) < .45)
    common = roi & masks[0] & masks[1]
    metrics = []
    for name, z, valid in zip(['correct', 'wrong_rotation'], surfaces, masks):
        e = (z[common]-3.1)*1000
        metrics.append(dict(condition=name, available_positions=int((roi & valid).sum()),
                            common_positions=int(common.sum()), mae_mm=float(abs(e).mean()),
                            rmse_mm=float(np.sqrt(np.mean(e**2)))))
    report = dict(date='2026-10-01', angle_deg=ANGLE_DEG,
                  rotation='R_wc_est = R_y_world(+1 degree) @ R_wc_true; camera center unchanged.',
                  inputs='Same two ideal simulated depths in both conditions; no random noise or re-rendering.',
                  reference='Teacher-set front plane mapZ=3.10m; only used in evaluation.',
                  roi_positions=int(roi.sum()), metrics=metrics, mesh_counts=counts,
                  checks='Proper rotation; unchanged depth/center; first-frame field identical; radii preserved; rotated plane equation satisfied.',
                  evidence='Teacher preparation; no student run or independent implementation.',
                  limits='One two-view synthetic rotation error; no pose estimation/correction or real-map accuracy.')
    (OUT / 'checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez_compressed(OUT / 'inputs_and_evaluation.npz', first_depth_m=views[0]['depth'],
                        second_depth_m=second['depth'], first_camera_m=views[0]['camera'],
                        second_camera_m=second['camera'], first_R_wc=views[0]['R_wc'],
                        second_R_wc_true=second['R_wc'], second_R_wc_est=error_R @ second['R_wc'],
                        wrong_second_world_points_m=wrong_points, second_front=front,
                        x_m=x, y_m=y, roi=roi, common=common)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    import matplotlib
    if args.save_only:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5))
    fig.subplots_adjust(left=.09, right=.96, bottom=.24, top=.76, wspace=.3)
    fig.suptitle('箱子没有倾斜，估错的相机朝向把观测放斜了', fontsize=20, y=.95)
    fig.text(.5, .86, '俯视箱子正面的一条截面：横轴左右位置，纵轴前后位置，单位均为米',
             ha='center', fontsize=12)
    for ax in axes:
        ax.axhline(3.1, color='#495769', ls='--', lw=2, label='人为真实箱面 Z=3.10米')
        ax.set(xlim=(-.6, .6), ylim=(3.09, 3.17), xlabel='地图 X：左右位置（米）',
               ylabel='地图 Z：前后位置（米）')
        ax.grid(alpha=.2)
    q = wrong_points[front]
    keep = (abs(q[:, 0]) < .6) & (abs(q[:, 1]) < .03)
    axes[0].scatter(q[keep, 0], q[keep, 2], s=16, color='#dc7745',
                    label='第2帧按错误朝向放入地图')
    axes[0].set_title('融合前：第2帧观测已放斜')
    j = np.argmin(abs(y[0]))
    line = (abs(x[:, j]) < .6) & masks[1][:, j]
    axes[1].plot(x[line, j], surfaces[1][line, j], color='#218c6d', lw=2.5,
                 label='两帧融合提取的表面截线')
    axes[1].set_title('融合后：表面也产生倾斜与偏移')
    for ax in axes:
        ax.legend(loc='upper right', fontsize=9)
    fig.text(.5, .135, '仅第2帧估计朝向绕地图 Y 轴偏1°；相机位置、深度和实际箱面不变。',
             ha='center', fontsize=11)
    fig.text(.5, .07, '为看清厘米级偏差，纵轴显示范围较窄；图中角度不代表实际倾角。',
             ha='center', fontsize=10, color='#42516a')
    fig.savefig(OUT / 'comparison.png', dpi=150)
    if args.save_only:
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    main()
