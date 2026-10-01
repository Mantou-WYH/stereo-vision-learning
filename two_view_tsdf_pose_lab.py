"""L08：同一带噪深度，第二帧地图位置误差如何传入TSDF表面。"""
import argparse
import json
from pathlib import Path
import numpy as np
from two_view_mapping_lab import build_views
from two_view_tsdf_lab import make_grid, integrate, extract, STEP, ORIGIN, MU
from two_view_tsdf_noise_lab import front_surface, OUT as NOISE_OUT

OUT = Path(__file__).resolve().parent / 'notes/synthetic_mapping/tsdf_pose_error'
SHIFT_M = np.array([0., 0., .04])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    views = build_views()
    grid = make_grid()
    field = np.zeros(grid.shape[:-1], np.float32)
    weights = np.zeros(field.shape, np.uint8)
    stages = []
    for n, view in enumerate(views, 1):
        data = np.load(NOISE_OUT / f'input_{n}.npz')
        noisy = dict(view)
        noisy['depth'] = data['depth_m'].copy()
        noisy['camera'] = data['camera_m'].copy()
        noisy['R_wc'] = data['R_wc'].copy()
        assert np.array_equal(noisy['R_wc'], view['R_wc'])
        assert np.array_equal(noisy['camera'], view['camera'])
        if n == 2:
            noisy['camera'] += SHIFT_M
            # 同一相机内点：错误位姿仅把整帧地图Z放远4厘米。
            correct = view['camera_points'] @ noisy['R_wc'].T + view['camera']
            wrong = view['camera_points'] @ noisy['R_wc'].T + noisy['camera']
            assert np.allclose(wrong - correct, SHIFT_M)
        stages.append(integrate(noisy, grid, field, weights))
        if n == 1:
            baseline = np.load(NOISE_OUT / 'after_1.npz')
            assert np.array_equal(weights, baseline['weights'])
            assert np.array_equal(field, baseline['tsdf'])
        np.savez_compressed(OUT / f'input_{n}.npz', depth_m=noisy['depth'],
                            camera_m=noisy['camera'], R_wc=noisy['R_wc'])
    vertices, faces = extract(field, weights)
    bad_z, bad_valid = front_surface(field, weights)
    np.savez_compressed(OUT / 'after_2_wrong_pose.npz', tsdf=field, weights=weights,
                        vertices_m=vertices, triangles=faces, front_z_m=bad_z,
                        front_valid=bad_valid, origin_m=ORIGIN, spacing_m=STEP, truncation_m=MU)
    evaluation = np.load(NOISE_OUT / 'evaluation.npz')
    first, correct = evaluation['first_front_z_m'], evaluation['fused_front_z_m']
    x, y, roi = evaluation['x_m'], evaluation['y_m'], evaluation['roi']
    common = evaluation['common'] & bad_valid
    surfaces = [first, correct, bad_z]
    metrics = []
    for name, z in zip(['first_frame', 'two_correct_poses', 'second_pose_shifted'], surfaces):
        error = (z[common] - 3.1) * 1000
        metrics.append(dict(condition=name, common_positions=int(common.sum()),
                            roi_available_positions=int((roi & np.isfinite(z)).sum()),
                            mae_mm=float(abs(error).mean()), mean_signed_error_mm=float(error.mean()),
                            rmse_mm=float(np.sqrt(np.mean(error ** 2)))))
    report = dict(kind='teacher-run TSDF pose translation comparison',
                  depths='Exactly the saved noisy input_1/input_2 from tsdf_noise, seed 20260930.',
                  change='Only second estimated camera position in world/map coordinates is shifted.',
                  second_pose_shift_m=SHIFT_M.tolist(), rotation='unchanged',
                  reference_front_z_m=3.1, roi_positions=int(roi.sum()),
                  evaluation='Same interior map positions; first positive-to-negative Z crossing.',
                  metrics=metrics, integration_stages=stages,
                  checks='First-frame field identical to noise baseline; all second-frame world points shift by +4cm Z.',
                  evidence='Teacher verification; student execution and independent implementation pending.',
                  limits='Two synthetic frames; no pose estimator or correction; no real-map accuracy claim.')
    (OUT / 'checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez_compressed(OUT / 'evaluation.npz', x_m=x, y_m=y, roi=roi, common=common,
                        first_front_z_m=first, correct_front_z_m=correct, wrong_front_z_m=bad_z)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    import matplotlib
    if args.save_only:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#bdbdbd')
    fig = plt.figure(figsize=(14, 7))
    fig.suptitle('两张深度图相同：只把第2帧的相机位置估错4厘米', fontsize=20, y=.96)
    fig.text(.5, .87, '相机实际没有再移动；融合程序把它的位置沿地图 Z 向前多算了4厘米',
             ha='center', fontsize=12)
    ix, iy = np.flatnonzero(roi.any(axis=1)), np.flatnonzero(roi.any(axis=0))
    extent = [x[ix[0], 0]-STEP/2, x[ix[-1], 0]+STEP/2,
              y[0, iy[-1]]+STEP/2, y[0, iy[0]]-STEP/2]
    for k, (left, surface, title) in enumerate(zip([.055, .385, .715], surfaces,
            ['只加入第1帧', '两帧：相机位置正确', '两帧：第2帧位置估错'])):
        ax = fig.add_axes([left, .32, .26, .43])
        residual = np.where(common, (surface - 3.1) * 1000, np.nan)
        im = ax.imshow(residual[np.ix_(ix, iy)].T, extent=extent,
                       interpolation='nearest', cmap=cmap,
                       norm=TwoSlopeNorm(vmin=-60, vcenter=0, vmax=60))
        ax.set(title=title, xlabel='地图 X（米）', ylabel='地图 Y（米）')
        fig.text(left+.13, .21, f"平均绝对偏离 {metrics[k]['mae_mm']:.1f} 毫米",
                 ha='center', fontsize=14)
    cax = fig.add_axes([.25, .115, .5, .025])
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal', extend='both')
    cbar.set_label('箱面沿地图 Z 的位置误差（毫米）：蓝偏近，白接近真值，红偏远')
    fig.text(.5, .02, f'相同 {common.sum():,} 处箱面位置、相同色阶；深度噪声与融合参数均保持原值。灰色为无共同估计。',
             ha='center', fontsize=10, color='#42516a')
    fig.savefig(OUT / 'comparison.png', dpi=150)
    if args.save_only:
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    main()
