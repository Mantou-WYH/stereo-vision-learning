"""无噪声白墙实验：SSD为零仍可能有多个同样好的候选。"""
import numpy as np
import matplotlib.pyplot as plt
from ssd_assignment import compute_ssd


def match_with_gap(left, right, radius=2):
    raw = np.full_like(left, np.nan)
    kept = np.full_like(left, np.nan)
    gaps = np.full_like(left, np.nan)
    for v in range(left.shape[0]):
        for u in range(radius, left.shape[1] - radius):
            patch = left[v, u-radius:u+radius+1]
            positions, costs = [], []
            for r in range(radius, min(u+1, right.shape[1]-radius)):
                candidate = right[v, r-radius:r+radius+1]
                if not np.isfinite(candidate).all():
                    continue
                positions.append(r)
                costs.append(compute_ssd(patch, candidate))
            if not costs:
                continue
            best = np.argmin(costs)
            # 与前例相同：先选出最小SSD，差别是此次还检查第二名。
            raw[v, u] = u - positions[best]
            if len(costs) < 2:
                continue
            ordered = np.sort(costs)
            gaps[v, u] = ordered[1] - ordered[0]
            # 本例无噪声：只保留零代价且没有并列第一的结果。
            # 这不是实际相机图像的通用阈值。
            if ordered[0] == 0 and gaps[v, u] > 0:
                kept[v, u] = raw[v, u]
    return raw, kept, gaps


def main(save_path=None):
    rng = np.random.default_rng(20260924)
    left = rng.integers(20, 236, (24, 48)).astype(float)
    left[6:18, 16:36] = 128  # 只把这块纹理改成均匀灰色。
    right = np.full_like(left, np.nan)
    right[:, :-6] = left[:, 6:]  # 全图真实位移仍为6，不加噪声。
    raw, kept, gaps = match_with_gap(left, right)
    # 真值仅供验算，不进入搜索及拒绝条件。
    assert np.all(kept[np.isfinite(kept)] == 6)
    assert np.isnan(kept[6:18, 18:34]).all()
    assert np.all(gaps[6:18, 18:34] == 0)
    assert raw[10, 24] == 12 and np.isnan(kept[10, 24])
    print("示例：左图(v=10,u=24)，真实视差6；硬选得到12；并列检查后为未知。")
    print("白墙内部多个窗口SSD都为0，最好与第二好之差为0。")
    print("灰色不表示距离0，只表示没有保留估计。当前例子特意不加噪声。")
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    gray = plt.get_cmap('gray').copy()
    gray.set_bad('#b4cedc')
    colors = plt.get_cmap('viridis').copy()
    colors.set_bad('#b8b8b8')
    for ax, data, title in zip(axes[0], [left, right],
                              ['左图：中间改成均匀灰色，模拟白墙', '右图：整体左移6像素（无噪声）']):
        ax.imshow(data, cmap=gray, vmin=0, vmax=255, interpolation='nearest')
        ax.set_title(title, fontsize=11)
    for ax, data, title in zip(axes[1], [raw, kept],
                              ['硬选最小SSD：白墙内出现错误视差', '拒绝并列最优：白墙内部留为未知']):
        plot = ax.imshow(data, cmap=colors, vmin=0, vmax=24, interpolation='nearest')
        ax.set_title(title, fontsize=11)
        fig.colorbar(plot, ax=ax, shrink=.7, ticks=[0, 6, 12, 24], label='视差（像素）')
    for ax in axes.ravel():
        ax.set_xlabel('横向位置 u')
        ax.set_ylabel('行号 v')
    fig.suptitle('真实位移都是6：有最小值，不等于能确定对应位置', fontsize=14)
    if save_path:
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    main()
