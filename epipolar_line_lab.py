"""L04：一条左相机射线上的候选三维点，在右图中形成一条极线。"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FX = FY = 600.0
CX, CY = 320.0, 240.0
BASELINE = 0.20

# 所有候选点都在左相机的同一条观察射线上，所以 X/Z、Y/Z 保持不变。
DEPTHS = np.array([0.8, 1.2, 2.0, 3.0, 5.0])
X_OVER_Z = 0.12
Y_OVER_Z = -0.05
POINTS = np.column_stack([
    X_OVER_Z * DEPTHS,
    Y_OVER_Z * DEPTHS,
    DEPTHS,
])


def project(points_camera):
    u = FX * points_camera[:, 0] / points_camera[:, 2] + CX
    v = FY * points_camera[:, 1] / points_camera[:, 2] + CY
    return np.column_stack([u, v])


LEFT_PIXELS = project(POINTS)
RIGHT_POINTS = POINTS - np.array([BASELINE, 0.0, 0.0])
RIGHT_PIXELS = project(RIGHT_POINTS)


def build_figure():
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(DEPTHS)))
    fig, (space_ax, left_ax, right_ax) = plt.subplots(
        1, 3, figsize=(12, 5), gridspec_kw={"width_ratios": [1.25, 1, 1]}
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.84, bottom=0.16, wspace=0.30)
    fig.suptitle("一个左图像素 → 一条空间射线 → 右图中的一条极线", y=0.95)

    # 左：俯视空间几何。
    space_ax.scatter([0], [0], marker="^", s=110, color="black", label="左相机")
    space_ax.scatter([BASELINE], [0], marker="^", s=110, color="tab:red", label="右相机")
    far_point = POINTS[-1]
    space_ax.plot([0, far_point[0]], [0, far_point[2]], "--", color="gray", linewidth=2,
                  label="左像素对应的射线")
    space_ax.scatter(POINTS[:, 0], POINTS[:, 2], s=70, c=colors,
                     edgecolors="black", linewidths=0.5)
    for point, depth in zip(POINTS, DEPTHS):
        space_ax.text(point[0] + 0.025, point[2], f"Z={depth:.1f}m", va="center", fontsize=9)
    space_ax.set_title("空间俯视图")
    space_ax.set_xlabel("X（米）")
    space_ax.set_ylabel("Z：向前深度（米）")
    space_ax.set_xlim(-0.08, 0.78)
    space_ax.set_ylim(-0.2, 5.5)
    space_ax.grid(alpha=0.25)
    space_ax.legend(loc="upper left", fontsize=8)

    # 中：候选点在左图完全重合。
    left_ax.set_title("左图：五个候选点重合")
    left_ax.set_xlim(180, 500)
    left_ax.set_ylim(360, 100)
    left_ax.set_aspect("equal")
    left_ax.axhline(LEFT_PIXELS[0, 1], color="gray", linestyle="--", linewidth=1)
    left_ax.scatter(LEFT_PIXELS[:, 0], LEFT_PIXELS[:, 1], s=170, c=colors,
                    edgecolors="black", linewidths=0.8)
    left_ax.annotate("同一个左像素", xy=LEFT_PIXELS[0], xytext=(250, 150),
                     arrowprops={"arrowstyle": "->"}, ha="center")
    left_ax.set_xlabel("u（像素）")
    left_ax.set_ylabel("v（像素）")
    left_ax.grid(alpha=0.18)

    # 右：同一条射线上的所有候选点落在一条水平极线上。
    epipolar_v = RIGHT_PIXELS[0, 1]
    right_ax.set_title("右图：候选位置排列在极线上")
    right_ax.set_xlim(180, 500)
    right_ax.set_ylim(360, 100)
    right_ax.set_aspect("equal")
    right_ax.axhline(epipolar_v, color="tab:red", linestyle="--", linewidth=2,
                     label="极线")
    right_ax.scatter(RIGHT_PIXELS[:, 0], RIGHT_PIXELS[:, 1], s=70, c=colors,
                     edgecolors="black", linewidths=0.5)
    for index, (pixel, depth) in enumerate(zip(RIGHT_PIXELS, DEPTHS)):
        label_y = pixel[1] - 18 if index % 2 == 0 else pixel[1] + 18
        vertical_alignment = "bottom" if index % 2 == 0 else "top"
        right_ax.text(pixel[0], label_y, f"{depth:.1f}m",
                      ha="center", va=vertical_alignment, fontsize=8)
    right_ax.set_xlabel("u（像素）")
    right_ax.set_ylabel("v（像素）")
    right_ax.grid(alpha=0.18)
    right_ax.legend(loc="lower right")

    disparities = LEFT_PIXELS[:, 0] - RIGHT_PIXELS[:, 0]
    fig.text(0.5, 0.06,
             "真正的右图匹配点会从这些候选位置中选出；选中后，视差即可确定深度。\n"
             f"示例视差（近→远）：{', '.join(f'{d:.0f}' for d in disparities)} px",
             ha="center")
    return fig


if __name__ == "__main__":
    figure = build_figure()
    plt.show()
