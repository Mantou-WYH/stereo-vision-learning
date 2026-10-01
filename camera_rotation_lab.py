"""L02：相机原地右转，原来正前方的静止点投影到左侧。"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from projection_lab import project_x


def rotation_and_projection(angle_degrees):
    """只看 X-Z 俯视平面；正角度表示相机向右转。"""
    theta = np.deg2rad(angle_degrees)
    c, s = np.cos(theta), np.sin(theta)
    # R_wc 将相机坐标转为世界坐标，列向量是相机轴在世界中的方向。
    R_wc = np.array([[c, s], [-s, c]])
    # 物体不转，相机原点也不动；用反方向将世界坐标转入相机系。
    R_cw = R_wc.T
    point_world = np.array([0.0, 2.0])
    point_camera = R_cw @ point_world
    x = project_x(point_camera[0], point_camera[1])
    return R_wc, point_camera, x


def build_figure():
    fig, (scene, image_ax) = plt.subplots(
        2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [3.7, 1.3]})
    fig.subplots_adjust(left=0.12, right=0.94, top=0.88, bottom=0.21, hspace=0.65)
    fig.suptitle("相机旋转实验：相机原地转，物体不动", y=0.97)
    scene.scatter([0], [2], color="tab:orange", s=70, zorder=5)
    scene.annotate("静止点：世界坐标 (0, 2) 米", (0, 2),
                   xytext=(-12, 13), textcoords="offset points", ha="right")
    scene.scatter([0], [0], color="black", s=60, zorder=5)
    scene.annotate("相机位置固定", (0, 0), xytext=(-8, -15),
                   textcoords="offset points", ha="right")
    scene.plot([0, 0], [0, 2.4], "--", color="gray", alpha=0.5)
    scene.set_xlim(-1.3, 1.3)
    scene.set_ylim(-0.5, 2.7)
    scene.set_aspect("equal", adjustable="box")
    scene.set_xlabel("世界横向 X（米）")
    scene.set_ylabel("世界前方 Z（米）")
    scene.grid(alpha=0.2)
    forward_line, = scene.plot([], [], color="tab:blue", linewidth=2.5)
    right_line, = scene.plot([], [], color="tab:green", linewidth=2.5)
    component_line, = scene.plot([], [], ":", color="tab:purple", linewidth=2)
    forward_text = scene.text(0, 0, "相机正前方", color="tab:blue")
    right_text = scene.text(0, 0, "相机右侧", color="tab:green")

    half_width = 250
    pixels = np.zeros((21, 2 * half_width + 1), dtype=np.uint8)
    image = image_ax.imshow(pixels, cmap="gray", vmin=0, vmax=255,
                            interpolation="nearest", origin="upper", aspect="auto",
                            extent=(-250.5, 250.5, 20.5, -0.5))
    image_ax.axvline(0, color="tab:red", linestyle="--", linewidth=1)
    image_ax.scatter([0], [10], s=90, facecolors="none", edgecolors="gray", linewidths=2)
    dot, = image_ax.plot([], [], "o", color="tab:blue", markersize=7)
    image_ax.set_title("图像：灰圈是转动前的位置，蓝点是转动后的位置")
    image_ax.set_xlabel("相对图像中心的位置（像素；负数是左侧，正数是右侧）")
    image_ax.set_yticks([])
    status = fig.text(0.5, 0.10, "", ha="center")
    slider_ax = fig.add_axes([0.29, 0.035, 0.52, 0.03])
    slider = Slider(slider_ax, "相机右转角度（度）", -20, 20,
                    valinit=10, valstep=1, valfmt="%.0f")

    def update(angle):
        R_wc, point_camera, x = rotation_and_projection(angle)
        right_axis = R_wc[:, 0]
        forward_axis = R_wc[:, 1]
        forward_tip = 2.35 * forward_axis
        right_tip = 0.90 * right_axis
        forward_line.set_data([0, forward_tip[0]], [0, forward_tip[1]])
        right_line.set_data([0, right_tip[0]], [0, right_tip[1]])
        forward_text.set_position((forward_tip[0] + 0.06, forward_tip[1]))
        right_text.set_position((right_tip[0] + 0.04, right_tip[1]))
        # 紫色虚线连接点与相机正前方轴，帮助看出点在轴的哪一侧。
        foot = point_camera[1] * forward_axis
        component_line.set_data([foot[0], 0], [foot[1], 2])
        turn = f"向右转 {angle:.0f}°" if angle >= 0 else f"向左转 {-angle:.0f}°"
        scene.set_title(f"俯视：{turn}；蓝线是新的正前方（等比例）")
        pixels.fill(0)
        pixels[10, int(round(x)) + half_width] = 255
        image.set_data(pixels)
        dot.set_data([x], [10])
        side = f"左侧 {-x:.0f} 像素" if x < 0 else f"右侧 {x:.0f} 像素"
        if np.isclose(x, 0):
            side = "图像中心"
        status.set_text(f"新相机坐标：X={point_camera[0]:+.2f} 米，Z={point_camera[1]:.2f} 米"
                        f"  →  {side}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(slider.val)
    return fig, slider


if __name__ == "__main__":
    figure, rotation_slider = build_figure()
    plt.show()
