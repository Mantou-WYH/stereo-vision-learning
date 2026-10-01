"""L09：沿用图像跟踪得到的PnP位姿，放置第二帧点云。"""
import argparse
import json
from pathlib import Path
import numpy as np
from two_view_mapping_lab import render, look_at, LOW, FX, FY, CX, CY

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'notes/synthetic_motion/tracked_pnp'
OUT = ROOT/'notes/synthetic_motion/estimated_pose_mapping'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save-only', action='store_true')
    args = parser.parse_args()
    report0 = json.loads((SOURCE/'checks.json').read_text(encoding='utf-8'))
    C0 = np.array([1.4, 0., 0.])
    R0 = look_at(C0, np.array([0., 0., 3.9]))
    C1_true = np.array(report0['truth_camera_position_in_first_m'])
    angle = np.deg2rad(report0['truth_yaw_in_first_deg'])
    R1_true = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                        [-np.sin(angle), 0., np.cos(angle)]])
    # 只重建同一模拟器的两帧理想深度观察，不重新跟踪/匹配/求位姿。
    first = render(C0, R0)
    second = render(C0+R0@C1_true, R0@R1_true)
    with np.load(SOURCE/'inputs.npz') as saved:
        assert np.array_equal(first['depth'], saved['first_depth_m'], equal_nan=True)
    with np.load(SOURCE/'estimated_pose.npz') as saved:
        C1_est = saved['camera_position_in_first_m'].copy()
        R1_est = saved['R_wc_in_first'].copy()
    points0 = first['camera_points']
    points1 = second['camera_points']
    # 三维旋转和平移将第二相机坐标变为第一相机坐标。
    placed_est = points1 @ R1_est.T + C1_est
    placed_true = points1 @ R1_true.T + C1_true  # 独立对照与事后评价。
    assert np.allclose(placed_true @ R0.T + C0, second['world_points'], atol=1e-12, rtol=0)
    assert np.allclose((placed_est-C1_est) @ R1_est, points1, atol=1e-12, rtol=0)
    faces1 = second['face'][second['valid']]
    front1 = faces1 == 4
    # 回到原世界仅用于检查与已知箱面Z=3.1的法向偏差。
    front_error = abs((placed_est @ R0.T+C0)[front1, 2]-LOW[2])
    true_front_error = abs((placed_true @ R0.T+C0)[front1, 2]-LOW[2])
    displacement = np.linalg.norm(placed_est-placed_true, axis=1)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT/'placed_clouds.npz', first_xyz_m=points0,
                        second_xyz_in_second_m=points1, second_xyz_estimated_in_first_m=placed_est,
                        second_xyz_reference_in_first_m=placed_true,
                        first_depth_m=first['depth'], second_depth_m=second['depth'],
                        estimated_C_in_first_m=C1_est, estimated_R_wc_in_first=R1_est)
    report = dict(kind='Place unchanged second-frame depth cloud using saved tracked-PnP pose',
                  first_points=len(points0), second_points=len(points1),
                  mean_second_front_normal_error_m=float(front_error.mean()),
                  max_second_front_normal_error_m=float(front_error.max()),
                  mean_reference_front_normal_error_m=float(true_front_error.mean()),
                  median_point_placement_error_m=float(np.median(displacement)),
                  display_section='Same physical second-frame samples selected by reference |Y|<0.03m; display only',
                  evidence='Teacher preparation/run; student run and implementation pending',
                  limits='Ideal synthetic depth; concatenation only, no averaging/TSDF/mesh or real sequence')
    (OUT/'checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    import matplotlib
    if args.save_only:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(13, 9))
    fig.suptitle('算出的相机位姿，决定第二帧点云放在哪里', fontsize=19, y=.96)
    fig.text(.5, .905, '同两帧理想深度；只比较第二帧采用设定位姿或已保存的PnP估计位姿。',
             ha='center', fontsize=12)
    color0, color1 = '#239ac6', '#e88331'
    clouds = [placed_true, placed_est]
    labels = ['设定位姿：作为对照', '图像跟踪得到的PnP位姿']
    section0 = abs(points0[:, 1]) < .03
    # 只用于统一观察的剖面选取，不参与放置、求位姿或任何过滤。
    section1 = abs(placed_true[:, 1]) < .03
    for i, (cloud, title) in enumerate(zip(clouds, labels)):
        ax = fig.add_axes([.06+i*.49, .52, .39, .32], projection='3d')
        for p, color, name in [(points0, color0, '第一次拍摄'), (cloud, color1, '第二次拍摄')]:
            sample = p[::2]
            ax.scatter(sample[:, 0], sample[:, 2], sample[:, 1], c=color, s=2,
                       linewidths=0, depthshade=False, label=name)
        ax.set(title=title, xlabel='X 向右（米）', ylabel='Z 向前（米）', zlabel='Y 向下（米）',
               xlim=(-1.2, .7), ylim=(2.9, 4.5), zlim=(.85, -.85))
        ax.set_box_aspect((1.9, 1.6, 1.7))
        ax.view_init(elev=19, azim=-60)
        ax.legend(fontsize=8, loc='upper left')
        detail = fig.add_axes([.09+i*.49, .18, .35, .21])
        for p, mask, color, name in [(points0, section0, color0, '第一次拍摄'),
                                      (cloud, section1, color1, '第二次拍摄')]:
            detail.scatter(p[mask, 0], p[mask, 2], c=color, s=9, label=name)
        detail.set(xlim=(-.3, .15), ylim=(3.20, 3.46), xlabel='X（米）', ylabel='Z（米）',
                   title='放大箱面中部的水平薄切片')
        detail.grid(alpha=.25)
        detail.legend(fontsize=8, loc='upper right')
    fig.text(.5, .08, f'估计位姿下，第二帧正面点偏离已知箱面的平均距离：{front_error.mean()*1000:.2f} 毫米',
             ha='center', fontsize=12)
    fig.text(.5, .025, '青色/橙色在本图区分拍摄次数；全部坐标为第一相机坐标，单位米。本次只放置，不平均或补洞。',
             ha='center', fontsize=10, color='#42516a')
    fig.savefig(OUT/'comparison.png', dpi=150)
    if args.save_only:
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    main()
