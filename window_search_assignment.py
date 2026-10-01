"""L05：从图像行提取窗口，复用 SSD 自动搜索匹配。只需完成一个 TODO。"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from ssd_assignment import compute_ssd


def extract_patch(row, u, radius):
    """截取以索引 u 为中心、左右各 radius 个像素的一维窗口。"""
    # 边界检查由教师提供；本轮不处理截断窗口或补边。
    if u - radius < 0 or u + radius >= row.size:
        raise ValueError("窗口超出图像边界")

    # TODO：把下面一行替换成 return row[起始索引:结束索引]
    # 起始索引为 u-radius；结束索引不包含在切片内。
    return row[u - radius : u + radius + 1]


def make_scene():
    """教师提供的理想数据：右图中纹理向左移动6像素，没有噪声。"""
    rng = np.random.default_rng(20260923)
    left_row = rng.integers(20, 236, size=48).astype(float)
    true_disparity = 6
    right_row = np.full_like(left_row, np.nan)
    right_row[:-true_disparity] = left_row[true_disparity:]
    # 右侧缺失区保留 NaN（未知），不循环回绕，也不用于匹配。
    return left_row, right_row, true_disparity


def show_result(left_row, right_row, u_left, best_u_right, radius,
                candidate_positions, costs, left_patch, best_patch, save_path=None):
    """教师提供的绘图。本轮不用修改。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), layout="constrained",
                             gridspec_kw={"height_ratios": [1, 1, 1.2, 2]})
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("#b4cedc")
    for ax, row, center, title in [
        (axes[0], left_row, u_left, "左图一行：固定参考窗口"),
        (axes[1], right_row, best_u_right, "右图一行：自动选出的窗口；浅蓝区无对应数据"),
    ]:
        ax.imshow(row[np.newaxis, :], cmap=cmap, vmin=0, vmax=255,
                  interpolation="nearest", aspect="auto")
        ax.add_patch(Rectangle((center-radius-0.5, -0.48), 2*radius+1, 0.96,
                               fill=False, edgecolor="#ed793a", linewidth=2))
        ax.set_title(f"{title}（中心 u={center}）")
        ax.set_yticks([])
        ax.set_xticks(np.arange(0, row.size, 6))
        ax.set_xlabel("横向像素索引 u")

    axes[2].imshow(np.stack([left_patch, best_patch]), cmap="gray", vmin=0,
                   vmax=255, interpolation="nearest", aspect="auto")
    axes[2].set_yticks([0, 1], ["左参考窗口", "右匹配窗口"])
    axes[2].set_xticks(np.arange(2*radius+1))
    axes[2].set_xlabel("窗口内部索引")
    axes[2].set_title("取出的两个窗口：灰度值逐个对应")

    axes[3].plot(candidate_positions, costs, "o-", color="#3575aa", markersize=4)
    axes[3].axvline(best_u_right, color="#ed793a", linestyle="--",
                    label=f"最佳 uR={best_u_right}，视差={u_left-best_u_right} px")
    axes[3].set_xlabel("候选右窗口中心 uR（像素）")
    axes[3].set_ylabel("SSD（灰度值平方）")
    axes[3].grid(alpha=0.2)
    axes[3].legend()
    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path, dpi=130)
        plt.close(fig)


def main(save_path=None):
    left_row, right_row, true_disparity = make_scene()
    u_left = 24
    radius = 2  # 左右各2个，加中心1个，共5个像素。
    left_patch = extract_patch(left_row, u_left, radius)

    # 本次合成数据只搜索非负视差；不把此限制推广到所有标定模型。
    candidate_positions = np.arange(radius, u_left + 1)
    costs = []
    for u_right in candidate_positions:
        right_patch = extract_patch(right_row, u_right, radius)
        costs.append(compute_ssd(left_patch, right_patch))

    costs = np.array(costs)
    best_index = np.argmin(costs)
    best_u_right = candidate_positions[best_index]
    best_patch = extract_patch(right_row, best_u_right, radius)
    disparity = u_left - best_u_right

    print("左参考窗口：", left_patch)
    print("右匹配窗口：", best_patch)
    print(f"左中心={u_left}，右中心={best_u_right}，估计视差={disparity} px")
    print(f"最低 SSD={costs[best_index]:g}，合成真值视差={true_disparity} px")
    assert left_patch.shape == (2 * radius + 1,), "请检查切片的结束索引"
    assert np.array_equal(left_patch, left_row[22:27]), "参考窗口位置有误"
    assert np.isfinite(costs).all(), "搜索范围包含无效像素"
    assert disparity == true_disparity, "估计视差与真值不一致"
    print("窗口提取与搜索检查通过")
    show_result(left_row, right_row, u_left, best_u_right, radius,
                candidate_positions, costs, left_patch, best_patch, save_path)


if __name__ == "__main__":
    main()
