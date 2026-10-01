"""固定有遮挡的合成输入，对比窗口跨越深度边界的影响。"""
import numpy as np
import matplotlib.pyplot as plt
from white_wall_lab import match_with_gap


def make_scene():
    rng = np.random.default_rng(20260924)
    left = rng.integers(20, 236, (16, 64)).astype(float)
    truth = np.full_like(left, 4)
    truth[:, 32:] = 10
    right = np.full_like(left, np.nan)
    owner = np.full(left.shape, -1, dtype=int)
    # 先投影远背景，再投影近物体；近物体覆盖重叠处的远背景。
    for u in range(left.shape[1]):
        d = int(truth[0, u])
        r = u - d
        if r >= 0:
            right[:, r] = left[:, u]
            owner[:, r] = u
    visible = np.zeros_like(left, dtype=bool)
    for u in range(left.shape[1]):
        r = u - int(truth[0, u])
        if r >= 0:
            visible[:, u] = owner[:, r] == u
    return left, right, truth, visible


def main(save_path=None):
    left, right, truth, visible = make_scene()
    outputs = [match_with_gap(left, right, radius=r) for r in (2, 6)]
    # 公共评估区远离图像边缘，仅观察物体边界附近，避免边缘窗口大小混淆。
    region = np.zeros_like(visible)
    region[:, 20:44] = True
    for width, (raw, kept, _) in zip((5, 13), outputs):
        valid = np.isfinite(kept)
        assert np.all(kept[valid] == truth[valid])
        assert not np.any(valid & ~visible)
        unknown_visible = np.count_nonzero(region & visible & ~valid)
        print(f'宽{width}：边界附近共同区域内，仍可见但被拒绝的位置{unknown_visible}个。')
        print(f'宽{width}：同一区域原始错误候选{np.count_nonzero(region & (raw != truth))}个。')
    assert np.isnan(outputs[0][1][:, 24:34]).all()
    assert np.isnan(outputs[1][1][:, 20:38]).all()
    print('左图u=26至31在右图被近物体遮挡：这部分不是扩大窗口导致的。')
    print('严格零SSD过滤仅用于本无噪声例子；未知区域变宽不等于真实物体变宽。')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), layout='constrained')
    gray = plt.get_cmap('gray').copy()
    gray.set_bad('#b4cedc')
    colors = plt.get_cmap('viridis').copy()
    colors.set_bad('#b8b8b8')
    for ax, data, boundary, title in zip(axes[0], [left, right], [31.5, 21.5],
            ['左图：左边远（d=4），右边近（d=10）', '右图：近物体盖住了一部分背景']):
        ax.imshow(data, cmap=gray, vmin=0, vmax=255, interpolation='nearest')
        ax.axvline(boundary, color='#ff7520', linestyle='--')
        ax.set_title(title, fontsize=11)
    for col, (width, (raw, kept, _)) in enumerate(zip((5, 13), outputs)):
        for row, data, name in [(1, raw, '硬选结果'), (2, kept, '严格过滤后：灰色是未知')]:
            ax = axes[row, col]
            plot = ax.imshow(data, cmap=colors, vmin=0, vmax=16, interpolation='nearest')
            ax.axvline(31.5, color='#ff7520', linestyle='--')
            ax.set_title(f'宽{width}：{name}', fontsize=11)
            fig.colorbar(plot, ax=ax, shrink=.8, ticks=[0, 4, 10, 16], label='视差（像素）')
    for ax in axes.ravel():
        ax.set_xlabel('横向位置 u')
        ax.set_ylabel('行号 v')
    fig.suptitle('看橙色边界附近：窗口越宽，越容易同时包含两种位移', fontsize=14)
    if save_path:
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    main()
