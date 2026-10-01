"""L02：相机平移加旋转；先换原点，再换朝向。仅研究 X-Z 平面。"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from projection_lab import project_x


def transform_and_project(camera_X, angle_degrees):
    point_world = np.array([0.30, 2.00])
    camera_world = np.array([camera_X, 0.0])
    theta = np.deg2rad(angle_degrees)
    c, s = np.cos(theta), np.sin(theta)
    R_wc = np.array([[c, s], [-s, c]])

    # 核心两步：减去相机的世界位置，再用逆旋转换到相机轴方向。
    delta_world = point_world - camera_world
    point_camera = R_wc.T @ delta_world
    x = project_x(point_camera[0], point_camera[1])
    return R_wc, delta_world, point_camera, x


def recover_world_point(R_wc, point_camera, camera_X):
    """把相机测得的点放回世界坐标：先转回世界方向，再加相机位置。"""
    camera_world = np.array([camera_X, 0.0])
    return R_wc @ point_camera + camera_world


def build_figure():
    fig, (scene, image_ax) = plt.subplots(
        2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [3.7, 1.3]})
    fig.subplots_adjust(left=0.12, right=0.94, top=0.88, bottom=0.27, hspace=0.65)
    fig.suptitle("相机位姿实验：先换原点，再换朝向", y=0.97)
    scene.scatter([0.30], [2.0], color="tab:orange", s=70, zorder=5)
    scene.annotate("物体固定：世界坐标 (0.30, 2.00) 米", (0.30, 2.0),
                   xytext=(-12, 13), textcoords="offset points", ha="right")
    scene.scatter([0], [0], marker="^", color="gray", s=65)
    camera, = scene.plot([], [], "^", color="tab:blue", markersize=10)
    recovered_marker, = scene.plot([], [], marker="*", linestyle="none",
                                   color="tab:purple", markersize=15, zorder=6)
    recovered_text = scene.text(0, 0, "", color="tab:purple")
    forward_line, = scene.plot([], [], color="tab:blue", linewidth=2.5)
    right_line, = scene.plot([], [], color="tab:green", linewidth=2.5)
    sight_line, = scene.plot([], [], "--", color="gray", alpha=0.6)
    forward_text = scene.text(0, 0, "新的正前方", color="tab:blue")
    scene.set_xlim(-1.0, 1.4)
    scene.set_ylim(-0.45, 2.8)
    scene.set_aspect("equal", adjustable="box")
    scene.set_xlabel("世界横向 X（米）")
    scene.set_ylabel("世界前方 Z（米）")
    scene.grid(alpha=0.2)

    half_width = 400
    pixels = np.zeros((21, 2 * half_width + 1), dtype=np.uint8)
    image = image_ax.imshow(pixels, cmap="gray", vmin=0, vmax=255,
                            interpolation="nearest", origin="upper", aspect="auto",
                            extent=(-400.5, 400.5, 20.5, -0.5))
    image_ax.axvline(0, color="tab:red", linestyle="--", linewidth=1)
    image_ax.scatter([90], [10], s=85, facecolors="none", edgecolors="gray", linewidths=2)
    dot, = image_ax.plot([], [], "o", color="tab:blue", markersize=7)
    image_ax.set_title("灰圈：原始图像位置；蓝点：改变相机位姿后的图像位置")
    image_ax.set_xlabel("相对图像中心的位置（像素；负数为左侧）")
    image_ax.set_yticks([])
    steps = fig.text(0.5, 0.15, "", ha="center")
    projection_status = fig.text(0.5, 0.12, "", ha="center")
    x_slider = Slider(fig.add_axes([0.29, 0.07, 0.52, 0.025]),
                      "相机右移（米）", 0, 0.20, valinit=0, valstep=0.01, valfmt="%.2f")
    rotation_slider = Slider(fig.add_axes([0.29, 0.03, 0.52, 0.025]),
                             "相机右转（度）", -20, 20, valinit=0, valstep=1, valfmt="%.0f")

    def update(_):
        camera_X, angle = x_slider.val, rotation_slider.val
        R_wc, delta, point_camera, x = transform_and_project(camera_X, angle)
        recovered_world = recover_world_point(R_wc, point_camera, camera_X)
        origin = np.array([camera_X, 0.0])
        forward_tip = origin + 2.35 * R_wc[:, 1]
        right_tip = origin + 0.85 * R_wc[:, 0]
        camera.set_data([camera_X], [0])
        recovered_marker.set_data([recovered_world[0]], [recovered_world[1]])
        recovered_text.set_position((recovered_world[0] + 0.08, recovered_world[1] - 0.25))
        recovered_text.set_text("紫星：逆变换恢复的世界位置")
        forward_line.set_data([camera_X, forward_tip[0]], [0, forward_tip[1]])
        right_line.set_data([camera_X, right_tip[0]], [0, right_tip[1]])
        sight_line.set_data([camera_X, 0.30], [0, 2.0])
        forward_text.set_position((forward_tip[0] + 0.05, forward_tip[1]))
        scene.set_title(f"俯视：相机右移 {camera_X:.2f} 米，右转 {angle:.0f}°（等比例）")
        pixels.fill(0)
        pixels[10, int(round(x)) + half_width] = 255
        image.set_data(pixels)
        dot.set_data([x], [10])
        steps.set_text(f"① 减相机位置：({delta[0]:+.2f}, {delta[1]:.2f}) 米"
                       f"    ② 转入相机朝向：({point_camera[0]:+.2f}, {point_camera[1]:.2f}) 米")
        side = f"中心左侧 {-x:.0f} 像素" if x < 0 else f"中心右侧 {x:.0f} 像素"
        if np.isclose(x, 0):
            side = "图像中心"
        round_trip_error = np.linalg.norm(recovered_world - np.array([0.30, 2.00]))
        projection_status.set_text(f"最后投影：{side}；逆变换往返误差：{round_trip_error:.2e} 米")
        fig.canvas.draw_idle()

    x_slider.on_changed(update)
    rotation_slider.on_changed(update)
    update(None)
    return fig, x_slider, rotation_slider


if __name__ == "__main__":
    figure, translation_slider, angle_slider = build_figure()
    plt.show()
