"""固定P1/P2，仅比较5×5与3×3；全图与共同作答区分开统计。"""
import numpy as np
import matplotlib.pyplot as plt
from real_sgbm_lab import load_pair, compute_disparity
from sgbm_reference_lab import load_reference
from sgbm_error_lab import evaluate


def main(save_path=None):
    left, right = load_pair()
    estimates = [compute_disparity(left, right, block_size=w) for w in (5, 3)]
    reference = load_reference(estimates[0].shape)
    reports = [evaluate(e, reference) for e in estimates]
    shared = np.isfinite(reference) & np.isfinite(estimates[0]) & np.isfinite(estimates[1])
    assert shared.any()
    print('只改变窗口5×5/3×3；P1=200、P2=800及其他参数固定。')
    for w, estimate, report in zip((5, 3), estimates, reports):
        stats = report[-1]
        shared_mae = np.abs(estimate[shared]-reference[shared]).mean()
        print(f"{w}×{w}: 作答比例{stats['coverage']:.2%}；各自作答处平均误差{stats['mae']:.3f}px；误差>2px比例{stats['bad2']:.2%}；共同作答处平均误差{shared_mae:.3f}px")
    print(f'共同作答区：{shared.sum()}个位置；仅用于配对比较，不代表缺失处也正确。')
    print('上述是整图统计，不单独证明叶片边缘得到改善。参考降采样仍沿用教学协议。')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
    disparity_colors = plt.get_cmap('viridis').copy()
    error_colors = plt.get_cmap('magma').copy()
    for cmap in (disparity_colors, error_colors):
        cmap.set_bad('#b8b8b8')
    for col, (w, estimate, report) in enumerate(zip((5, 3), estimates, reports)):
        error, _, _, _, stats = report
        top = axes[0, col].imshow(estimate, cmap=disparity_colors, vmin=0, vmax=128, interpolation='nearest')
        axes[0, col].set_title(f"{w}×{w}：视差；作答比例 {stats['coverage']:.1%}", fontsize=11)
        bottom = axes[1, col].imshow(error, cmap=error_colors, vmin=0, vmax=10, interpolation='nearest')
        axes[1, col].set_title(f"{w}×{w}：误差；各自作答处平均 {stats['mae']:.2f} 像素", fontsize=11)
    fig.colorbar(top, ax=list(axes[0]), shrink=.8, label='视差（像素）；灰色：没有估计')
    fig.colorbar(bottom, ax=list(axes[1]), shrink=.8, extend='max', label='误差（像素）；灰色：无法比较')
    for ax in axes.ravel():
        ax.set_xlabel('横向位置 u')
        ax.set_ylabel('纵向位置 v')
    fig.suptitle('同一盆芦荟：左列5×5，右列3×3；下排越亮表示错得越多', fontsize=14)
    if save_path:
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    main()
