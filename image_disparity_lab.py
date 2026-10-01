"""L05：逐行复用匹配器，把一行视差组成一张视差图。

仍然使用横向5像素窗口，不是二维5×5窗口。本轮未加入唯一性检查。
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from row_disparity_lab import match_row
from noise_threshold_lab import filter_candidates


def make_images():
    """合成上半部位移4、下半部位移10的图；真值仅用于验算。"""
    rng = np.random.default_rng(20260924)
    left_image = rng.integers(20, 236, size=(24, 48)).astype(float)
    true_disparity = np.full_like(left_image, 4)
    true_disparity[12:] = 10
    right_image = np.full_like(left_image, np.nan)
    right_image[:12, :-4] = left_image[:12, 4:]
    right_image[12:, :-10] = left_image[12:, 10:]
    available = np.isfinite(right_image)
    right_image[available] += rng.normal(0, 1, available.sum())
    return left_image, right_image, true_disparity


def match_image(left_image, right_image):
    # 创建与左图同样大小的结果，每个位置先记为未知。
    height, width = left_image.shape
    disparity_image = np.full((height, width), np.nan)

    # 本轮核心：每次取左右图的同一行，复用之前的整行匹配。
    for v in range(height):
        left_row = left_image[v]
        right_row = right_image[v]
        disparities, costs = match_row(left_row, right_row, radius=2)
        filtered_row = filter_candidates(disparities, costs, threshold=30)
        # 把这一行视差存回结果图的第v行。
        disparity_image[v] = filtered_row

    return disparity_image


def disparity_to_depth(disparity, focal_px, baseline_m):
    """本实验采用平行、主点对齐的双目模型；返回前向深度Z（米）。"""
    depth = np.full(disparity.shape, np.nan, dtype=float)
    # 未知、零或负视差不参与除法，避免伪造距离或除以零。
    valid = np.isfinite(disparity) & (disparity > 0)
    depth[valid] = focal_px * baseline_m / disparity[valid]
    return depth


def main(save_path=None):
    left, right, true_disparity = make_images()
    result = match_image(left, right)
    # 教学相机参数，不是由合成图片估计出的真实标定值。
    focal_px = 600.0
    baseline_m = 0.1
    depth = disparity_to_depth(result, focal_px, baseline_m)

    # 真值只用于核对结果，不进入匹配及过滤函数。
    valid = np.isfinite(result)
    columns = np.arange(left.shape[1])[None, :]
    overlap = (columns >= true_disparity + 2) & (columns < left.shape[1] - 2)
    assert valid.any()
    assert np.all(result[valid] == true_disparity[valid])
    assert not np.any(valid & ~overlap)
    print("输入左图形状：", left.shape)
    print("输出视差图形状：", result.shape)
    print(f"保留结果：{valid.sum()}/{result.size}个位置。")
    print("上半部保留的视差：", np.unique(result[:12][valid[:12]]), "像素")
    print("下半部保留的视差：", np.unique(result[12:][valid[12:]]), "像素")
    assert np.all(depth[:12][valid[:12]] == 15.0)
    assert np.all(depth[12:][valid[12:]] == 6.0)
    assert np.array_equal(np.isfinite(depth), valid)
    print("教学参数：焦距600像素，基线0.1米；深度Z = 60 / 视差。")
    print("上半部深度15米（更远），下半部深度6米（更近）。")
    print(f"完整对应窗口区共{overlap.sum()}个位置，其中保留{np.count_nonzero(valid & overlap)}个。")
    print("彩色图中每个格子对应左图同一位置，颜色表示视差，不是原图亮度。")
    print("灰色表示没有保留视差；不是视差0，也不是深度0。")
    print("本轮仍使用横向5像素窗口，逐行计算；尚未换成二维窗口。")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, grid = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    axes = grid.ravel()
    gray = plt.get_cmap("gray").copy()
    gray.set_bad("#b4cedc")
    for ax, data, title in zip(axes[:2], [left, right],
                              ["左图：24行，每行48个像素", "右图：上半部左移4，下半部左移10"]):
        ax.imshow(data, cmap=gray, vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(title, fontsize=11)
    colors = plt.get_cmap("viridis").copy()
    colors.set_bad("#b8b8b8")
    plotted = axes[2].imshow(result, cmap=colors, vmin=0, vmax=12, interpolation="nearest")
    axes[2].set_title("输出：一张视差图，同色代表同视差", fontsize=11)
    fig.colorbar(plotted, ax=axes[2], shrink=0.7, ticks=[0, 4, 10, 12], label="视差（像素）")
    depth_plot = axes[3].imshow(depth, cmap=colors, vmin=0, vmax=18,
                                interpolation="nearest")
    axes[3].set_title("换算结果：深度图（前向距离Z）", fontsize=11)
    fig.colorbar(depth_plot, ax=axes[3], shrink=0.7, ticks=[0, 6, 15, 18], label="深度（米）")
    for ax, labels in [(axes[2], ["4 像素", "10 像素"]),
                       (axes[3], ["15 米：更远", "6 米：更近"])]:
        for v, label in zip([5, 17], labels):
            ax.text(28, v, label, ha="center", va="center", color="white",
                    fontsize=12, bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"))
    axes[2].legend(handles=[Patch(facecolor="#b8b8b8", label="灰色：没有保留结果")],
                   loc="upper center", bbox_to_anchor=(0.5, -0.27), fontsize=10)
    for ax in axes:
        ax.set_xlabel("横向位置 u")
        ax.set_ylabel("行号 v")
        ax.set_xticks([0, 12, 24, 36, 47])
        ax.set_yticks([0, 6, 12, 18, 23])
    fig.suptitle("同一个位置：先匹配得到视差，再用 Z = 600 × 0.1 / 视差换算深度", fontsize=14)
    if save_path:
        fig.savefig(save_path, dpi=130)
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
