"""L05：固定同一份带噪声数据，只改变最低SSD的接受上限。"""
import numpy as np
import matplotlib.pyplot as plt

from window_search_assignment import make_scene
from row_disparity_lab import match_row


def filter_candidates(disparities, costs, threshold):
    result = np.full(disparities.size, np.nan)
    for u_left in range(disparities.size):
        # 只检查最小SSD有多大。本轮没有加入唯一性差距检查。
        if costs[u_left] <= threshold:
            result[u_left] = disparities[u_left]
    return result


def main(save_path=None):
    left, right, true_disparity = make_scene()
    # 第一步：只改变右图亮度，不移动像素，真实位移仍是6。
    rng = np.random.default_rng(20260924)
    available = np.isfinite(right)
    right[available] += rng.normal(0, 1, available.sum())

    # 第二步：搜索只做一次，两种过滤规则共用这份候选及SSD。
    disparities, costs = match_row(left, right, radius=2)

    # 第三步：比较两种规则。30仅是本固定教学例子的演示值。
    thresholds = [0, 30]
    outputs = [filter_candidates(disparities, costs, t) for t in thresholds]

    # 真值只用于下面的检查与说明，不参与搜索或过滤。
    u = np.arange(left.size)
    overlap = (u >= true_disparity + 2) & (u < left.size - 2)
    assert np.all(disparities[overlap] == true_disparity)
    assert not np.isfinite(outputs[0]).any()
    assert np.array_equal(np.isfinite(outputs[1]), overlap)
    print("同一份图像、同一次搜索，仅改变SSD上限。")
    print(f"例：左中心24仍选中右中心{24-int(disparities[24])}，但SSD变为{costs[24]:.2f}。")
    for threshold, output in zip(thresholds, outputs):
        kept = np.isfinite(output)
        print(f"SSD上限{threshold}：保留正确候选{np.count_nonzero(kept & overlap)}/38个；保留边缘候选{np.count_nonzero(kept & ~overlap)}/6个。")
    print("另有4个左窗口不完整位置，两种规则下都未计算。")
    print("30不是通用阈值；本例不能证明放宽上限总能提高准确度。")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), layout="constrained", sharex=True, sharey=True)
    titles = ["旧规则：SSD必须为0 → 正确候选也全部被拒绝",
              "允许少量差异：SSD不超过30 → 本例保留38个正确候选"]
    for ax, output, title in zip(axes, outputs, titles):
        ax.plot(u, output, "o", color="#3575aa", markersize=5)
        ax.set_title(title)
        ax.set_xlim(-0.5, 47.5)
        ax.set_ylim(-0.8, 8)
        ax.set_yticks([0, 2, 4, 6, 8])
        ax.set_ylabel("保留的视差（像素）")
        ax.grid(alpha=0.2)
    axes[0].text(24, 4, "没有蓝点：没有候选被保留", ha="center")
    axes[1].set_xlabel("左图像素位置 uL；没有蓝点表示未知，不是视差0")
    if save_path:
        fig.savefig(save_path, dpi=130)
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
