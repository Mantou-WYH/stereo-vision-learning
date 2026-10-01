"""L05：从一个左像素扩展到一行。教师提供外层循环，复用已有核心函数。"""
import numpy as np
import matplotlib.pyplot as plt

from ssd_assignment import compute_ssd
from window_search_assignment import extract_patch, make_scene


def match_row(left_row, right_row, radius):
    # NaN 表示尚无结果，不把缺失视差写成0。
    disparity_row = np.full(left_row.size, np.nan)
    minimum_costs = np.full(left_row.size, np.nan)

    # 新增外层循环：每轮换一个左像素。两端取不满窗口的位置跳过。
    for u_left in range(radius, left_row.size - radius):
        left_patch = extract_patch(left_row, u_left, radius)
        costs = []
        positions = []

        # 原来的内层搜索：为当前左像素寻找右图对应位置。
        # 只搜索非负视差，这是本次合成实验的约定。
        for u_right in range(radius, min(u_left + 1, right_row.size - radius)):
            right_patch = extract_patch(right_row, u_right, radius)
            if not np.isfinite(right_patch).all():
                continue  # 右图缺失数据不能参与比较。
            costs.append(compute_ssd(left_patch, right_patch))
            positions.append(u_right)

        if not costs:
            continue
        best_index = np.argmin(costs)
        best_u_right = positions[best_index]

        # 新增存储：每个结果写回对应的左像素位置。
        disparity_row[u_left] = u_left - best_u_right
        minimum_costs[u_left] = costs[best_index]

    # 这些是未经可靠性过滤的候选结果，有最小值不代表存在真实匹配。
    return disparity_row, minimum_costs


def main(save_path=None):
    left_row, right_row, true_disparity = make_scene()
    radius = 2
    disparities, costs = match_row(left_row, right_row, radius)
    positions = np.arange(left_row.size)

    # 本轮新增：先全部标为未知，只保留 SSD 为0的候选。
    # 仅用于当前无噪声、整数灰度合成实验，不是实际图像的通用判据。
    filtered_disparities = np.full(left_row.size, np.nan)
    for u_left in range(left_row.size):
        if costs[u_left] == 0:
            filtered_disparities[u_left] = disparities[u_left]

    # 以下真值范围仅用于评估和绘图，不传入 match_row。
    # 左中心6、7虽然有对应像素，但对应的右窗口越过图像左边界。
    full_overlap = (positions >= true_disparity + radius) & (positions < left_row.size-radius)
    assert np.all(disparities[full_overlap] == true_disparity)
    assert np.all(costs[full_overlap] == 0)
    assert np.isnan(disparities[:radius]).all()
    assert np.isnan(disparities[-radius:]).all()
    print("左图中心8至45：完整对应窗口区域，预计视差6。")
    print("实际估计：", disparities[full_overlap])
    print("左图中心2至7的原始候选：", disparities[2:8])
    print("这些边缘候选缺少真实对应或完整右窗口，不能当作可靠测量。")
    print("左图中心0、1、46、47取不满左窗口：保留NaN（未计算）。")
    assert np.array_equal(np.isfinite(filtered_disparities), full_overlap)
    assert np.all(filtered_disparities[full_overlap] == true_disparity)
    print("过滤后左中心2至7：", filtered_disparities[2:8])
    print("原始候选44个；过滤后保留38个，拒绝6个；另有4个未计算。")
    print("过滤前后数值检查通过。NaN表示未知，不是视差0。")
    print("注意：SSD为0也可能有重复纹理歧义，本轮未加入唯一性检查。")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), layout="constrained",
                             gridspec_kw={"height_ratios": [0.7, 0.7, 2, 2]})
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("#b4cedc")
    for ax, row, title in zip(axes[:2], [left_row, right_row],
                              ["左图一行", "右图一行：纹理向左移6像素；浅蓝为缺失数据"]):
        ax.imshow(row[np.newaxis, :], cmap=cmap, vmin=0, vmax=255,
                  interpolation="nearest", aspect="auto")
        ax.set_title(title)
        ax.set_yticks([])
        ax.set_xlabel("图像中的横向索引 u")
        ax.set_xticks(np.arange(0, 48, 6))

    ax = axes[2]
    ax.plot(positions, disparities, "o-", color="#3575aa", markersize=4,
            label="逐像素搜索得到的原始视差")
    ax.plot(positions[full_overlap], np.full(full_overlap.sum(), true_disparity),
            "--", color="#e18325", linewidth=2, label="完整窗口对应区：真值6")
    ax.axvspan(-0.5, 1.5, color="#777777", alpha=0.15, label="左窗口不完整：未计算")
    ax.axvspan(45.5, 47.5, color="#777777", alpha=0.15)
    ax.axvspan(1.5, 7.5, color="#cf7161", alpha=0.18,
               label="无真实对应或无完整右窗口：候选可能错误")
    ax.set_xlim(-0.5, 47.5)
    ax.set_ylim(-1, max(9, float(np.nanmax(disparities)) + 3))
    ax.set_xlabel("左图像素位置 uL")
    ax.set_ylabel("该位置的视差（像素）")
    ax.set_title("过滤前：没有完整对应的边缘区域也被选出了候选")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=9)
    ax = axes[3]
    ax.plot(positions, filtered_disparities, "o-", color="#3575aa", markersize=4)
    ax.axvspan(-0.5, 1.5, color="#777777", alpha=0.15)
    ax.axvspan(45.5, 47.5, color="#777777", alpha=0.15)
    ax.axvspan(1.5, 7.5, color="#cf7161", alpha=0.18)
    ax.text(4.5, 3, "拒绝\n留空", ha="center")
    ax.set_xlim(axes[2].get_xlim())
    ax.set_ylim(axes[2].get_ylim())
    ax.set_xlabel("左图像素位置 uL")
    ax.set_ylabel("保留的视差（像素）")
    ax.set_title("过滤后：只保留SSD=0的候选；空缺表示未知")
    ax.grid(alpha=0.2)
    if save_path:
        fig.savefig(save_path, dpi=130)
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
