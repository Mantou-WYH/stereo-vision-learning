"""L09：四个理想共面三维—二维对应，用现有OpenCV求解相机位姿。"""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'notes/synthetic_motion/pnp_pose'


def estimate_pose(world_points_m, observed_uv_px, K):
    """求解器只接收地图点、图像观察和相机标定，未传入模拟真实位姿。"""
    import cv2
    ok, rvec, tvec = cv2.solvePnP(np.ascontiguousarray(world_points_m, dtype=np.float64),
                                 np.ascontiguousarray(observed_uv_px, dtype=np.float64),
                                 K, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError('solvePnP未返回成功结果')
    R_cw, _ = cv2.Rodrigues(rvec)
    R_wc = R_cw.T
    camera_m = -R_wc @ tvec.reshape(3)
    predicted, _ = cv2.projectPoints(world_points_m, rvec, tvec, K, np.zeros(5))
    return camera_m, R_wc, rvec, tvec, predicted.reshape(-1, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save-only', action='store_true')
    args = parser.parse_args()
    data = np.load(ROOT/'notes/synthetic_motion/pnp_intro/correspondences.npz')
    points, observed, K = data['world_points_m'], data['second_uv_px'], data['K']
    camera, rotation, rvec, tvec, predicted = estimate_pose(points, observed, K)
    error_px = np.linalg.norm(predicted-observed, axis=1)
    assert np.all(((points-camera) @ rotation)[:, 2] > 0)
    # 模拟真实位姿在求解后读取，仅评价、展示；没有传入estimate_pose。
    from two_view_mapping_lab import build_views
    first, second = build_views()
    position_error_m = float(np.linalg.norm(camera-second['camera']))
    rotation_error_deg = float(np.rad2deg(np.arccos(np.clip(
        (np.trace(rotation.T @ second['R_wc'])-1)/2, -1, 1))))
    assert position_error_m < 1e-6
    assert rotation_error_deg < 1e-4
    assert error_px.max() < 1e-5
    yaw_deg = float(np.rad2deg(np.arctan2(rotation[0, 2], rotation[2, 2])))
    OUT.mkdir(parents=True, exist_ok=True)
    import cv2
    report = dict(kind='Teacher-run OpenCV solvePnP ITERATIVE, planar ideal four-point example.',
                  opencv_version=cv2.__version__, inputs='Only world_points_m, observed_uv_px, K; zero distortion.',
                  estimated_camera_position_m=camera.tolist(), estimated_yaw_deg=yaw_deg,
                  camera_position_reference_m=second['camera'].tolist(),
                  position_error_m=position_error_m, rotation_error_deg=rotation_error_deg,
                  reprojection_error_px=error_px.tolist(),
                  conversion='OpenCV R_cw/t_cw -> R_wc=R_cw.T, camera_center=-R_cw.T @ t_cw.',
                  evidence='Teacher preparation; student execution/independent implementation pending.',
                  limits='Synthetic ideal planar case; no tracking, noise, outlier handling, or general unique-pose claim.')
    (OUT/'checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez_compressed(OUT/'estimated_pose.npz', camera_position_m=camera, R_wc=rotation,
                        rvec_cw=rvec, tvec_cw=tvec, world_points_m=points, observed_uv_px=observed,
                        predicted_uv_px=predicted, K=K, reprojection_error_px=error_px)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    import matplotlib
    if args.save_only:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    fig.subplots_adjust(left=.08, right=.96, top=.76, bottom=.27, wspace=.3)
    fig.suptitle('这次实际求解：只给点与图像对应，让PnP找相机位姿', fontsize=19, y=.95)
    fig.text(.5, .865, '左：求解结果投回新照片；右：求解得到的相机位置与朝向（俯视地图）',
             ha='center', fontsize=12)
    ax = axes[0]
    ax.imshow(second['picture'], interpolation='nearest', alpha=.75)
    ax.scatter(observed[:, 0], observed[:, 1], s=28, c='#162435', label='固定图像观察', zorder=3)
    ax.scatter(predicted[:, 0], predicted[:, 1], s=150, facecolors='none', edgecolors='#00a67c',
               linewidths=2.5, label='求解位姿的预测投影', zorder=4)
    for label, pixel in zip('ABCD', observed):
        ax.annotate(label, pixel, xytext=(-12, -12), textcoords='offset points', fontsize=12)
    ax.set(title='观察点与预测点重合', xlabel='u（像素）', ylabel='v（像素）')
    ax.legend(loc='lower right', fontsize=9)
    ax = axes[1]
    ax.add_patch(Rectangle((-.8, 3.1), 1.6, 1.6, facecolor='#bddbe3', edgecolor='#3e6d77'))
    ax.text(0, 3.9, '固定箱子', ha='center', fontsize=12)
    ax.scatter([0], [0], c='#69788b', marker='^', s=65, label='第1帧相机 / 地图原点')
    ax.scatter([second['camera'][0]], [second['camera'][2]], facecolors='none',
               edgecolors='#dd7945', s=170, linewidths=2, label='教师真实位置（仅核对）')
    ax.scatter([camera[0]], [camera[2]], marker='x', c='#00a67c', s=85,
               linewidths=2.5, label='PnP估计位置')
    tip = camera + 1.4*rotation[:, 2]
    ax.annotate('', xy=tip[[0, 2]], xytext=camera[[0, 2]],
                arrowprops=dict(arrowstyle='->', color='#00a67c', lw=2.5))
    ax.text(tip[0]-.1, tip[2]+.18, '估计朝向', ha='center', color='#17775e')
    ax.set(xlim=(-1.2, 3.3), ylim=(-.5, 5.1), xlabel='地图 X（米）',
           ylabel='地图 Z（米）', title='位移与旋转一同求出')
    ax.set_aspect('equal'); ax.grid(alpha=.2); ax.legend(loc='upper right', fontsize=8)
    display_camera = np.where(abs(camera)<.0005, 0., camera)
    fig.text(.5, .165, f'估计位置：X={display_camera[0]:.3f}米，Y={display_camera[1]:.3f}米，Z={display_camera[2]:.3f}米；朝向：向左约{abs(yaw_deg):.1f}°',
             ha='center', fontsize=13)
    fig.text(.5, .09, '模拟真值未作为求解器输入；本例无噪声且对应正确，结果只说明此例成功。',
             ha='center', fontsize=11, color='#42516a')
    fig.savefig(OUT/'comparison.png', dpi=150)
    if args.save_only:
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    main()
