"""L04：观察右相机旋转造成的垂直视差，以及双目校正的作用。"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.collections import LineCollection

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

WIDTH, HEIGHT = 640, 480
FX = FY = 500.0
CX, CY = WIDTH / 2, HEIGHT / 2
BASELINE = 0.12

# 世界坐标系与左相机坐标系重合；每行是一个已知三维点 (X, Y, Z)，单位为米。
POINTS_WORLD = np.array([
    [-0.40, -0.18, 2.4],
    [-0.18,  0.16, 2.0],
    [ 0.00, -0.05, 1.7],
    [ 0.14,  0.08, 2.8],
    [ 0.38, -0.14, 3.4],
    [ 0.48,  0.20, 3.8],
], dtype=float)


def rotation_x(degrees):
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def project(points_camera):
    u = FX * points_camera[:, 0] / points_camera[:, 2] + CX
    v = FY * points_camera[:, 1] / points_camera[:, 2] + CY
    return np.column_stack([u, v])


def calculate_projections(right_rotation_degrees):
    left_points = POINTS_WORLD.copy()

    # 右相机中心比左相机向右 BASELINE 米。
    right_center_world = np.array([BASELINE, 0.0, 0.0])
    rotation_right_from_world = rotation_x(right_rotation_degrees)
    right_points = (rotation_right_from_world @
                    (POINTS_WORLD - right_center_world).T).T

    left_pixels = project(left_points)
    right_pixels_unrectified = project(right_points)

    # 已知右相机旋转后，把右相机射线转回与左相机平行的虚拟相机方向。
    right_points_rectified = (rotation_right_from_world.T @ right_points.T).T
    right_pixels_rectified = project(right_points_rectified)
    return left_pixels, right_pixels_unrectified, right_pixels_rectified


def prepare_pair_axis(axis, title):
    gap = 70
    right_offset = WIDTH + gap
    axis.set_xlim(-20, right_offset + WIDTH + 20)
    axis.set_ylim(HEIGHT + 25, -45)
    axis.set_aspect("equal")
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.add_patch(plt.Rectangle((0, 0), WIDTH, HEIGHT, fill=False,
                                 edgecolor="gray", linewidth=1.5))
    axis.add_patch(plt.Rectangle((right_offset, 0), WIDTH, HEIGHT, fill=False,
                                 edgecolor="gray", linewidth=1.5))
    axis.text(WIDTH / 2, -14, "左图", ha="center")
    axis.text(right_offset + WIDTH / 2, -14, "右图", ha="center")
    return right_offset


def build_figure():
    fig, (before_ax, after_ax) = plt.subplots(2, 1, figsize=(11, 8))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.90, bottom=0.18, hspace=0.25)
    fig.suptitle("右相机有轻微旋转：校正前后对应点的行坐标", y=0.97)

    before_offset = prepare_pair_axis(before_ax, "校正前：同名点不一定在同一行")
    after_offset = prepare_pair_axis(after_ax, "校正后：同名点连线恢复为水平")
    colors = plt.cm.tab10(np.arange(len(POINTS_WORLD)))

    empty_positions = np.zeros(len(POINTS_WORLD))
    left_before = before_ax.scatter(empty_positions, empty_positions, s=60, c=colors,
                                    edgecolors="black", linewidths=0.5)
    right_before = before_ax.scatter(empty_positions, empty_positions, s=60, c=colors,
                                     marker="x", linewidths=2)
    before_lines = LineCollection([], colors=colors, linewidths=1.7, alpha=0.85)
    before_ax.add_collection(before_lines)

    left_after = after_ax.scatter(empty_positions, empty_positions, s=60, c=colors,
                                  edgecolors="black", linewidths=0.5)
    right_after = after_ax.scatter(empty_positions, empty_positions, s=60, c=colors,
                                   marker="x", linewidths=2)
    after_lines = LineCollection([], colors=colors, linewidths=1.7, alpha=0.85)
    after_ax.add_collection(after_lines)

    status = fig.text(0.5, 0.115, "", ha="center")
    slider = Slider(fig.add_axes([0.29, 0.045, 0.54, 0.03]),
                    "右相机绕 x 轴旋转（度）", -8, 8,
                    valinit=5, valstep=0.5, valfmt="%+.1f°")

    def update(rotation_degrees):
        left, right_before_pixels, right_after_pixels = calculate_projections(rotation_degrees)
        left_before.set_offsets(left)
        right_before_display = right_before_pixels + np.array([before_offset, 0])
        right_before.set_offsets(right_before_display)
        before_lines.set_segments([
            [left_pixel, right_pixel]
            for left_pixel, right_pixel in zip(left, right_before_display)
        ])

        left_after.set_offsets(left)
        right_after_display = right_after_pixels + np.array([after_offset, 0])
        right_after.set_offsets(right_after_display)
        after_lines.set_segments([
            [left_pixel, right_pixel]
            for left_pixel, right_pixel in zip(left, right_after_display)
        ])

        before_vertical_error = right_before_pixels[:, 1] - left[:, 1]
        after_vertical_error = right_after_pixels[:, 1] - left[:, 1]
        status.set_text(
            f"校正前最大垂直差 {np.max(np.abs(before_vertical_error)):.1f} px；"
            f"校正后最大垂直差 {np.max(np.abs(after_vertical_error)):.2e} px"
        )
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(slider.val)
    return fig, slider


if __name__ == "__main__":
    figure, rotation_slider = build_figure()
    plt.show()
