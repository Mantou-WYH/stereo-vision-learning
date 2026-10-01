"""L05：在校正后的同一行上滑动小窗口，用 SSD 找到对应位置。"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RNG = np.random.default_rng(7)
ROW_LENGTH = 160
TRUE_DISPARITY = 18
LEFT_CENTER = 105
WINDOW_RADIUS = 5

# 生成一行带有明显纹理的灰度值，并模拟右图向左移动后的观测。
raw = RNG.uniform(15, 240, ROW_LENGTH)
LEFT_ROW = np.convolve(raw, np.ones(5) / 5, mode="same")
LEFT_ROW[96:115] += np.array([0, 8, 18, 35, 60, 90, 120, 145, 165,
                              180, 160, 130, 95, 65, 42, 25, 12, 5, 0])
LEFT_ROW = np.clip(LEFT_ROW, 0, 255)

RIGHT_ROW = np.full(ROW_LENGTH, 90.0)
RIGHT_ROW[:-TRUE_DISPARITY] = LEFT_ROW[TRUE_DISPARITY:]
RIGHT_ROW += RNG.normal(0, 2.0, ROW_LENGTH)
RIGHT_ROW = np.clip(RIGHT_ROW, 0, 255)

SEARCH_CENTERS = np.arange(55, LEFT_CENTER)


def window(row, center):
    return row[center - WINDOW_RADIUS:center + WINDOW_RADIUS + 1]


LEFT_PATCH = window(LEFT_ROW, LEFT_CENTER)
SSD_COSTS = np.array([
    np.sum((LEFT_PATCH - window(RIGHT_ROW, center)) ** 2)
    for center in SEARCH_CENTERS
])
BEST_RIGHT_CENTER = int(SEARCH_CENTERS[np.argmin(SSD_COSTS)])


def build_figure():
    fig, (left_ax, right_ax, cost_ax) = plt.subplots(
        3, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [1, 1, 1.25]}
    )
    fig.subplots_adjust(left=0.10, right=0.96, top=0.90, bottom=0.18, hspace=0.42)
    fig.suptitle("同一行上的窗口匹配：把左侧纹理拿到右图逐位置比较", y=0.97)

    x = np.arange(ROW_LENGTH)
    left_ax.plot(x, LEFT_ROW, color="tab:orange", linewidth=1.8)
    left_ax.axvspan(LEFT_CENTER - WINDOW_RADIUS, LEFT_CENTER + WINDOW_RADIUS,
                    color="tab:orange", alpha=0.25, label="固定的左图小窗口")
    left_ax.axvline(LEFT_CENTER, color="tab:orange", linestyle="--")
    left_ax.set_title("左图第 v 行：截取这一小段纹理")
    left_ax.set_ylabel("灰度")
    left_ax.set_xlim(45, 125)
    left_ax.set_ylim(0, 270)
    left_ax.grid(alpha=0.18)
    left_ax.legend(loc="upper left")

    right_ax.plot(x, RIGHT_ROW, color="tab:blue", linewidth=1.8)
    candidate_span = right_ax.axvspan(0, 1, color="tab:blue", alpha=0.25,
                                      label="正在尝试的右图窗口")
    candidate_line = right_ax.axvline(0, color="tab:blue", linestyle="--")
    right_ax.set_title("右图同一行：拖动窗口寻找相同纹理")
    right_ax.set_ylabel("灰度")
    right_ax.set_xlim(45, 125)
    right_ax.set_ylim(0, 270)
    right_ax.grid(alpha=0.18)
    right_ax.legend(loc="upper left")

    cost_ax.plot(SEARCH_CENTERS, SSD_COSTS, color="gray", linewidth=1.7)
    cost_ax.axvline(BEST_RIGHT_CENTER, color="tab:green", linestyle="--",
                    label="SSD 最低：最佳匹配")
    current_cost_marker, = cost_ax.plot([], [], "o", color="tab:red", markersize=8,
                                        label="当前尝试位置")
    cost_ax.set_xlim(SEARCH_CENTERS[0], SEARCH_CENTERS[-1])
    cost_ax.set_ylim(bottom=0)
    cost_ax.set_xlabel("右图候选窗口中心 uR（像素）")
    cost_ax.set_ylabel("SSD：平方差之和")
    cost_ax.set_title("两段纹理越相似，SSD 越低")
    cost_ax.grid(alpha=0.25)
    cost_ax.legend(loc="upper left")

    status = fig.text(0.5, 0.115, "", ha="center")
    slider = Slider(
        fig.add_axes([0.27, 0.045, 0.58, 0.03]),
        "右图候选位置 uR", int(SEARCH_CENTERS[0]), int(SEARCH_CENTERS[-1]),
        valinit=65, valstep=1, valfmt="%d px"
    )

    def update(candidate_center):
        candidate_center = int(candidate_center)
        candidate_span.set_x(candidate_center - WINDOW_RADIUS)
        candidate_span.set_width(2 * WINDOW_RADIUS)
        candidate_line.set_xdata([candidate_center, candidate_center])
        index = int(np.where(SEARCH_CENTERS == candidate_center)[0][0])
        current_cost = SSD_COSTS[index]
        current_cost_marker.set_data([candidate_center], [current_cost])
        current_disparity = LEFT_CENTER - candidate_center
        status.set_text(
            f"当前：uL={LEFT_CENTER}，uR={candidate_center}，d={current_disparity} px，"
            f"SSD={current_cost:.1f}    |    "
            f"最佳：uR={BEST_RIGHT_CENTER}，d={LEFT_CENTER - BEST_RIGHT_CENTER} px"
        )
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(slider.val)
    return fig, slider


if __name__ == "__main__":
    figure, candidate_slider = build_figure()
    plt.show()
