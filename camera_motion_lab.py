"""L02：相机向右移动，静止点在图像中向左移动。"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import FancyArrowPatch
from projection_lab import project_x


def coordinates_and_projection(camera_X):
    # 物体固定在世界中的 X=0.30 m、Z=2.0 m；相机只沿 X 移动且不转动。
    X_camera = 0.30 - camera_X
    Z_camera = 2.0
    return X_camera, Z_camera, project_x(X_camera, Z_camera)


def build_figure():
    fig, (scene, image_ax) = plt.subplots(
        2, 1, figsize=(9, 7), gridspec_kw={"height_ratios": [3, 1.4]})
    fig.subplots_adjust(left=0.11, right=0.95, top=0.88, bottom=0.21, hspace=0.65)
    fig.suptitle("相机移动实验：物体不动，相机向右移动", y=0.97)

    scene.scatter([0.30], [2.0], s=90, color="tab:orange", label="静止物体")
    scene.annotate("物体固定：世界位置 (0.30, 2.00) 米", (0.30, 2.0),
                   xytext=(-6, 15), textcoords="offset points", ha="right")
    scene.scatter([0], [0], marker="^", s=85, color="gray", label="相机原始位置")
    scene.plot([0, 0.30], [0, 2.0], "--", color="gray", alpha=0.5)
    camera, = scene.plot([], [], "^", markersize=10, color="tab:blue", label="相机当前位置")
    ray, = scene.plot([], [], color="tab:blue", linewidth=2)
    forward_axis, = scene.plot([], [], ":", color="tab:blue", alpha=0.5)
    scene.set_xlim(-0.08, 0.44)
    scene.set_ylim(-0.25, 2.65)
    scene.set_xlabel("世界中的左右位置（米）")
    scene.set_ylabel("世界中的前方位置（米）")
    scene.grid(alpha=0.25)
    scene.legend(loc="upper left")

    half_width = 120
    pixels = np.zeros((21, 2 * half_width + 1), dtype=np.uint8)
    image = image_ax.imshow(pixels, cmap="gray", vmin=0, vmax=255,
                            interpolation="nearest", origin="upper", aspect="auto",
                            extent=(-120.5, 120.5, 20.5, -0.5))
    image_ax.axvline(0, color="tab:red", linestyle="--", linewidth=1)
    image_ax.scatter([90], [10], s=95, facecolors="none", edgecolors="gray", linewidths=2)
    dot, = image_ax.plot([], [], "o", color="tab:blue", markersize=7)
    arrow = FancyArrowPatch((90, 4), (60, 4), arrowstyle="->",
                            mutation_scale=14, color="tab:blue", linewidth=1.5)
    image_ax.add_patch(arrow)
    image_ax.set_yticks([])
    image_ax.set_xlabel("图像中相对中心的横向位置（像素；正数表示中心右侧）")
    image_ax.set_title("图像：灰圈是移动前的位置，蓝点是当前位置；红线是中心")

    status = fig.text(0.5, 0.105, "", ha="center")
    slider_ax = fig.add_axes([0.29, 0.035, 0.52, 0.03])
    slider = Slider(slider_ax, "相机向右移动（米）", 0.0, 0.20,
                    valinit=0.10, valstep=0.01, valfmt="%.2f")

    def update(camera_X):
        X_camera, Z_camera, x = coordinates_and_projection(camera_X)
        camera.set_data([camera_X], [0])
        ray.set_data([camera_X, 0.30], [0, 2.0])
        forward_axis.set_data([camera_X, camera_X], [0, 2.3])
        scene.set_title(f"俯视示意：相机右移 {camera_X:.2f} 米，物体保持不动（横纵比例不同）")
        pixels.fill(0)
        pixels[10, int(round(x)) + half_width] = 255
        image.set_data(pixels)
        dot.set_data([x], [10])
        arrow.set_positions((90, 4), (x, 4))
        arrow.set_visible(not np.isclose(x, 90))
        status.set_text(f"相对相机：右侧 {X_camera:.2f} 米，前方 {Z_camera:.2f} 米"
                        f"  →  图像位置：中心右侧 {x:.0f} 像素")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(slider.val)
    # 返回 slider 保留引用，窗口内的拖动才会持续有效。
    return fig, slider


if __name__ == "__main__":
    figure, camera_slider = build_figure()
    plt.show()
