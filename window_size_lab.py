"""固定无噪声输入，对比横向5与13像素窗口；不使用二维窗口。"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from white_wall_lab import match_with_gap


def main(save_path=None):
    rng = np.random.default_rng(20260924)
    left = rng.integers(20, 236, (24, 48)).astype(float)
    left[6:18, 16:36] = 128
    right = np.full_like(left, np.nan)
    right[:, :-6] = left[:, 6:]
    results = [match_with_gap(left, right, radius=r)[1] for r in (2, 6)]
    for width, result in zip((5, 13), results):
        assert np.all(result[np.isfinite(result)] == 6)
        count = np.isfinite(result[6:18, 16:36]).sum()
        print(f'窗口宽{width}：白墙区域240个位置中保留{count}个，未知{240-count}个。')
    assert np.isfinite(results[0][6:18, 16:36]).sum() == 48
    assert np.isfinite(results[1][6:18, 16:36]).sum() == 144
    print('窗口更宽，能借到更远的纹理；墙中央仍未知。')
    print('同时图像两侧灰边变宽，因为大窗口更容易超出图像边界。')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#b8b8b8')
    for column, (width, result) in enumerate(zip((5, 13), results)):
        ax = axes[0, column]
        ax.imshow(left, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        # 两个窗口中心同为(u=19,v=10)，大窗口能跨过白墙左边缘。
        ax.add_patch(Rectangle((19-width//2-.5, 9.5), width, 1,
                               fill=False, edgecolor='#ff7b25', linewidth=2))
        ax.set_title(f'同一张左图：橙框宽{width}像素，同一中心', fontsize=11)
        ax = axes[1, column]
        im = ax.imshow(result, cmap=cmap, vmin=0, vmax=12, interpolation='nearest')
        ax.add_patch(Rectangle((15.5, 5.5), 20, 12, fill=False,
                               edgecolor='#ff7b25', linewidth=1.5))
        ax.set_title(f'宽{width}的结果：橙框内灰色表示未知', fontsize=11)
        fig.colorbar(im, ax=ax, shrink=.7, ticks=[0, 6, 12], label='视差（像素）')
    for ax in axes.ravel():
        ax.set_xlabel('横向位置 u')
        ax.set_ylabel('行号 v')
    fig.suptitle('只改变窗口宽度：白墙内部的未知区域会缩小吗？', fontsize=14)
    if save_path:
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    main()
