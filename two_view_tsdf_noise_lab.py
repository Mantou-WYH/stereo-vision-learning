"""L08：已知位姿下，两帧独立深度噪声的TSDF融合对照。"""
import argparse
import json
from pathlib import Path
import numpy as np
from two_view_mapping_lab import build_views
from two_view_tsdf_lab import make_grid, integrate, extract, STEP, ORIGIN, MU

OUT = Path(__file__).resolve().parent / 'notes/synthetic_mapping/tsdf_noise'
SEED = 20260930
NOISE_STD_M = .02


def front_surface(field, weights):
    """沿地图Z寻找第一个正到负零交叉；不按参考面挑选交叉。"""
    crossings = ((weights[:, :, :-1] > 0) & (weights[:, :, 1:] > 0)
                 & (field[:, :, :-1] > 0) & (field[:, :, 1:] <= 0))
    valid = crossings.any(axis=2)
    k = crossings.argmax(axis=2)
    i, j = np.indices(k.shape)
    left, right = field[i, j, k], field[i, j, k + 1]
    fraction = np.divide(left, left - right, out=np.zeros_like(left),
                         where=left != right)
    depth = ORIGIN[2] + (k + fraction) * STEP
    depth[~valid] = np.nan
    return depth, valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    views = build_views()
    rng = np.random.default_rng(SEED)
    grid = make_grid()
    field = np.zeros(grid.shape[:-1], np.float32)
    weights = np.zeros(field.shape, np.uint8)
    surfaces, masks, stages = [], [], []
    for number, view in enumerate(views, 1):
        noisy = dict(view)
        depth = view['depth'].copy()
        depth[view['valid']] += rng.normal(0, NOISE_STD_M, int(view['valid'].sum()))
        assert np.array_equal(np.isnan(depth), np.isnan(view['depth']))
        assert np.all(depth[view['valid']] > 0)
        noisy['depth'] = depth
        # 保存本帧输入；积分仅使用带噪深度、内参和位姿。
        np.savez_compressed(OUT / f'input_{number}.npz', depth_m=depth,
                            valid=view['valid'], camera_m=view['camera'], R_wc=view['R_wc'])
        info = integrate(noisy, grid, field, weights)
        vertices, faces = extract(field, weights)
        surface, valid = front_surface(field, weights)
        surfaces.append(surface.copy())
        masks.append(valid.copy())
        info.update(frames=number, vertices=len(vertices), triangles=len(faces))
        stages.append(info)
        np.savez_compressed(OUT / f'after_{number}.npz', tsdf=field, weights=weights,
                            vertices_m=vertices, triangles=faces, front_z_m=surface,
                            front_valid=valid, origin_m=ORIGIN, spacing_m=STEP,
                            truncation_m=MU)

    # 评价限定正面内部，避开边角；两次使用完全相同的地图位置。
    x, y = grid[:, :, 0, 0], grid[:, :, 0, 1]
    roi = (abs(x) < .6) & (abs(y) < .45)
    common = roi & masks[0] & masks[1]
    assert common.any()
    reference_front_z_m = 3.1  # 人为合成箱面的真值，只在评价阶段使用。
    for stage, surface, mask in zip(stages, surfaces, masks):
        errors = surface[common] - reference_front_z_m
        stage.update(roi_available_positions=int((roi & mask).sum()),
                     common_positions=int(common.sum()),
                     mae_mm=float(abs(errors).mean() * 1000),
                     rmse_mm=float(np.sqrt(np.mean(errors ** 2)) * 1000))
    report = dict(seed=SEED, noise_std_m=NOISE_STD_M,
                  noise='Independent zero-mean Gaussian noise at each valid forward-depth pixel.',
                  poses='Unchanged exact teacher-set poses from two_view_mapping_lab.',
                  reference_front_z_m=reference_front_z_m, roi_positions=int(roi.sum()),
                  evaluation='First positive-to-negative map-Z zero crossing; same common interior positions.',
                  grid_spacing_m=STEP, truncation_m=MU, stages=stages,
                  evidence='Teacher preparation only; no student execution or independent acceptance.',
                  limits='One synthetic seed; no pose error or systematic bias; no real-stereo accuracy claim.')
    (OUT / 'checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez_compressed(OUT / 'evaluation.npz', x_m=x, y_m=y, roi=roi, common=common,
                        first_front_z_m=surfaces[0], fused_front_z_m=surfaces[1],
                        reference_front_z_m=reference_front_z_m)
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
    fig = plt.figure(figsize=(12, 6.5))
    fig.suptitle('同一块箱面：加入第2帧后，表面起伏是否减小？', fontsize=20, y=.97)
    fig.text(.5, .89, '两个相机位姿保持准确；每帧深度加入独立随机误差（标准差2厘米）',
             ha='center', fontsize=12)
    ix, iy = np.flatnonzero(roi.any(axis=1)), np.flatnonzero(roi.any(axis=0))
    images = []
    for k, left in enumerate([.075, .555]):
        ax = fig.add_axes([left, .275, .37, .48])
        residual = np.where(common, (surfaces[k] - reference_front_z_m) * 1000, np.nan)
        patch = residual[np.ix_(ix, iy)].T
        extent = [x[ix[0], 0]-STEP/2, x[ix[-1], 0]+STEP/2,
                  y[0, iy[-1]]+STEP/2, y[0, iy[0]]-STEP/2]
        images.append(ax.imshow(patch, extent=extent, interpolation='nearest',
                                cmap=cmap, norm=TwoSlopeNorm(vmin=-60, vcenter=0, vmax=60)))
        ax.set(title='只加入第1帧' if k == 0 else '加入第1帧，再加入第2帧',
               xlabel='箱面左右位置：地图 X（米）', ylabel='箱面上下位置：地图 Y（米）')
        fig.text(left+.185, .18, f"平均绝对偏离：{stages[k]['mae_mm']:.1f} 毫米",
                 ha='center', fontsize=15)
    cax = fig.add_axes([.25, .105, .5, .025])
    cbar = fig.colorbar(images[0], cax=cax, orientation='horizontal', extend='both')
    cbar.set_label('相对真值平面的地图 Z 偏差（毫米）：蓝色偏近，白色贴近，红色偏远')
    fig.text(.5, .015, f'同一块正面内部、相同的 {common.sum():,} 个地图位置；颜色表示位置误差，不是外观或深度。',
             ha='center', fontsize=10, color='#42516a')
    fig.savefig(OUT / 'comparison.png', dpi=150)
    if args.save_only:
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    main()
