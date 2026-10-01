"""L09：固定3D点，给2D定位加小误差，观察共同位姿的投影残差。"""
import argparse
import json
from pathlib import Path
import numpy as np
from pnp_pose_lab import estimate_pose

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'notes/synthetic_motion/pnp_noise'
OFFSET_PX = np.array([[.8, -.4], [-.6, .5], [.3, -.7], [-.4, .6]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save-only', action='store_true')
    args = parser.parse_args()
    data = np.load(ROOT/'notes/synthetic_motion/pnp_intro/correspondences.npz')
    points, clean, K = data['world_points_m'], data['second_uv_px'], data['K']
    observed = clean + OFFSET_PX
    C, R, rvec, tvec, predicted = estimate_pose(points, observed, K)
    residual = predicted-observed
    errors = np.linalg.norm(residual, axis=1)
    initial_sse = float(np.sum(OFFSET_PX**2))
    solved_sse = float(np.sum(residual**2))
    assert solved_sse < initial_sse
    assert np.all(((points-C) @ R)[:, 2] > 0)
    assert errors.min() > .1
    OUT.mkdir(parents=True, exist_ok=True)
    report = dict(kind='Teacher-run PnP with fixed small per-point pixel offsets; no outliers.',
                  offsets_px=OFFSET_PX.tolist(), residual_lengths_px=errors.tolist(),
                  total_squared_reprojection_error_px2=solved_sse,
                  reference_pose_squared_error_px2=initial_sse,
                  estimated_camera_position_m=C.tolist(),
                  input_change='Only observed 2D positions; same world points, K, zero distortion and solver.',
                  evaluation='Clean projections used only after solving to compare reference-pose residual.',
                  evidence='Teacher preparation; student understanding and execution pending.',
                  limits='Four planar synthetic points, one preset error pattern; no general accuracy/uniqueness claim.')
    (OUT/'checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez_compressed(OUT/'result.npz', world_points_m=points, K=K,
                        observed_uv_px=observed, clean_uv_px=clean, offsets_px=OFFSET_PX,
                        predicted_uv_px=predicted, residual_uv_px=residual,
                        camera_position_m=C, R_wc=R, rvec_cw=rvec, tvec_cw=tvec)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    import matplotlib
    if args.save_only:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from two_view_mapping_lab import build_views
    picture = build_views()[1]['picture']
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.subplots_adjust(left=.1, right=.94, top=.8, bottom=.18, wspace=.3, hspace=.55)
    fig.suptitle('定位有一点误差：同一套位姿让四个点整体接近观察', fontsize=18, y=.965)
    fig.text(.5, .9, '新图A—D附近的放大窗口；黑点：带误差的固定观察，绿圈：求解后的投影',
             ha='center', fontsize=11)
    for ax, label, o, p, error in zip(axes.flat, 'ABCD', observed, predicted, errors):
        ax.imshow(picture, interpolation='nearest', alpha=.55)
        ax.plot([o[0], p[0]], [o[1], p[1]], color='#18725a', lw=2)
        ax.scatter(*o, color='#182334', s=55, zorder=4)
        ax.scatter(*p, facecolors='none', edgecolors='#009774', s=140, linewidths=2.5, zorder=5)
        ax.set(xlim=(o[0]-1.25, o[0]+1.25), ylim=(o[1]+1.25, o[1]-1.25),
               xlabel='u（像素）', ylabel='v（像素）', title=f'{label}：仍相差 {error:.2f} 像素')
        ax.set_aspect('equal'); ax.grid(alpha=.25)
        ax.ticklabel_format(style='plain', useOffset=False)
    fig.text(.5, .095, '三维点和标定不变；只给二维观察加小幅定位误差，没有加入错误对应。',
             ha='center', fontsize=11)
    fig.text(.5, .04, '六个运动量共用一套位姿来解释所有观察；本例保留小残差，未做到每个点完全重合。',
             ha='center', fontsize=10, color='#42516a')
    fig.savefig(OUT/'comparison.png', dpi=150)
    if args.save_only:
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    main()
