"""L08：已对齐的两帧点云，按2厘米体素保存均值与观测点数。"""
import argparse
import json
from pathlib import Path
import numpy as np
from two_view_mapping_lab import build_views

OUT = Path(__file__).resolve().parent / 'notes/synthetic_mapping/voxel_merge'
SIZE = .02
ORIGIN = np.array([-1.003, -1.003, 2.997])


def merge_voxels(points, size=SIZE, origin=ORIGIN):
    """格子只由三维坐标决定，不使用表面真值或帧编号。"""
    keys = np.floor((points-origin)/size).astype(np.int64)
    cells, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((len(cells), 3))
    np.add.at(sums, inverse, points)
    return cells, sums/counts[:, None], counts, inverse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save-only', action='store_true')
    args = parser.parse_args()
    first, second = build_views()
    p1, p2 = first['world_points'], second['world_points']
    points = np.concatenate([p1, p2])
    cells, means, counts, inverse = merge_voxels(points)
    source = np.r_[np.zeros(len(p1), dtype=int), np.ones(len(p2), dtype=int)]
    source_counts = np.zeros((len(cells), 2), dtype=int)
    np.add.at(source_counts, (inverse, source), 1)
    # 表面标签仅用于教师核验，不参与合并。
    faces = np.concatenate([v['face'][v['valid']] for v in (first, second)])
    front_counts = np.bincount(inverse, weights=(faces == 4), minlength=len(cells)).astype(int)
    side_counts = np.bincount(inverse, weights=(faces == 1), minlength=len(cells)).astype(int)
    assert counts.sum() == len(points)
    assert np.array_equal(source_counts.sum(axis=1), counts)
    assert np.allclose(np.sum(means*counts[:, None], axis=0), points.sum(axis=0))
    assert np.allclose(means[(front_counts > 0) & (side_counts == 0), 2], 3.1)
    assert np.allclose(means[(side_counts > 0) & (front_counts == 0), 0], .8)
    shared = int(np.all(source_counts > 0, axis=1).sum())
    mixed = int(((front_counts > 0) & (side_counts > 0)).sum())
    report = dict(input_points=len(points), output_points=len(means), voxel_size_m=SIZE,
                  grid_origin_m=ORIGIN.tolist(), shared_by_two_frames=shared,
                  only_second_frame=int(((source_counts[:, 0] == 0) & (source_counts[:, 1] > 0)).sum()),
                  cells_with_side_observations=int((side_counts > 0).sum()),
                  cells_mixing_front_and_side=mixed, reduction_percent=100*(1-len(means)/len(points)),
                  evidence='Teacher preparation; student execution and independent implementation pending.',
                  limitation='Synthetic ideal depths and known poses. Voxel averaging, no TSDF or accuracy improvement claim.')
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT/'map.npz', input_xyz_m=points, input_frame=source,
                        voxel_keys=cells, xyz_m=means, point_counts=counts,
                        frame_point_counts=source_counts, input_to_voxel=inverse,
                        voxel_size_m=SIZE, origin_m=ORIGIN)
    (OUT/'checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    import matplotlib
    if args.save_only:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    blue, orange, green = '#078ac5', '#ed842e', '#168361'
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('两帧已经对齐：把重复采样整理成一份地图', fontsize=22, y=.98)
    fig.text(.5, .92, '同一合成箱子，深度与位姿固定；空间小格边长 2 厘米，代表点取格内均值',
             ha='center', fontsize=13, color='#42516a')
    fig.text(.5, .89, '蓝色：第1帧原始点　橙色：第2帧原始点　绿色：合并后的代表点',
             ha='center', fontsize=12)
    for left, merged in [(.045, False), (.545, True)]:
        ax = fig.add_axes([left, .47, .42, .37], projection='3d')
        for p, color in ([(means, green)] if merged else [(p1, blue), (p2, orange)]):
            ax.scatter(p[:, 0], p[:, 2], p[:, 1], c=color, s=2, linewidths=0, depthshade=False)
        title = f'合并后：{len(means):,} 个代表点' if merged else f'直接拼接：{len(points):,} 个点'
        ax.set(title=title, xlabel='地图 X（米）', ylabel='地图 Z（米）', zlabel='Y 向下（米）',
               xlim=(-1, 1), ylim=(2.9, 4.9), zlim=(.8, -.8))
        ax.set_box_aspect((2, 2, 1.6))
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([3.1, 4, 4.7]); ax.set_zticks([-.6, 0, .6])
        ax.view_init(elev=22, azim=-53)
    # 正面一块12厘米见方区域。格线使用同一个三维网格的X/Y边界。
    lo, hi = -.103, .017
    def patch_mask(p):
        return ((p[:, 0] >= lo) & (p[:, 0] < hi) & (p[:, 1] >= lo) & (p[:, 1] < hi)
                & np.isclose(p[:, 2], 3.1))
    for left, merged in [(.14, False), (.64, True)]:
        ax = fig.add_axes([left, .145, .22, .235])
        for p, color in ([(means, green)] if merged else [(p1, blue), (p2, orange)]):
            q = p[patch_mask(p)]
            ax.scatter(q[:, 0], q[:, 1], c=color, s=25 if merged else 15,
                       marker='D' if merged else 'o', zorder=3)
        for edge in np.arange(lo, hi+.001, SIZE):
            ax.axvline(edge, color='#cbd5df', linewidth=.8)
            ax.axhline(edge, color='#cbd5df', linewidth=.8)
        ax.set(xlim=(lo, hi), ylim=(hi, lo), xlabel='地图 X（米）', ylabel='地图 Y（米）',
               title='同一块正面放大：每格一个代表点' if merged else '正面局部放大：两帧点落入小格')
        ax.set_aspect('equal')
        ax.set_xticks([-.10, -.05, 0]); ax.set_yticks([-.10, -.05, 0])
    fig.text(.5, .075, f'点数减少 {report["reduction_percent"]:.1f}%　|　{shared:,} 个格子收到两帧观测　|　正面与右侧面仍保留',
             ha='center', fontsize=13, color='#42516a')
    fig.text(.5, .03, '上图显示全部输入点与全部代表点；空格不生成点。均值合并不保证精度提升，也不会恢复未观测表面。',
             ha='center', fontsize=11, color='#42516a')
    fig.savefig(OUT/'comparison.png', dpi=130)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.save_only:
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    main()
