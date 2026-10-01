"""L03：观察径向畸变为什么在图像边缘和四角更明显。"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.collections import LineCollection

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


WIDTH, HEIGHT = 640, 480
FX = FY = 500.0
CX, CY = WIDTH / 2, HEIGHT / 2

# 一组规则的理想像素网格。先转成以主点为原点的归一化坐标。
U_VALUES = np.linspace(40, 600, 9)
V_VALUES = np.linspace(40, 440, 7)
UU, VV = np.meshgrid(U_VALUES, V_VALUES)
U_IDEAL = UU.ravel()
V_IDEAL = VV.ravel()


def distort_points(k1):
    """用一个径向畸变参数 k1 计算观测到的像素位置。"""
    x = (U_IDEAL - CX) / FX
    y = (V_IDEAL - CY) / FY
    radius_squared = x**2 + y**2
    scale = 1.0 + k1 * radius_squared
    u_distorted = FX * x * scale + CX
    v_distorted = FY * y * scale + CY
    return u_distorted, v_distorted, np.sqrt(radius_squared)


def build_figure():
    fig, (image_ax, residual_ax) = plt.subplots(
        2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [2.4, 1.2]}
    )
    fig.subplots_adjust(left=0.10, right=0.96, top=0.91, bottom=0.20, hspace=0.36)
    fig.suptitle("径向畸变：越远离主点，像素位置越容易被拉动", y=0.97)

    image_ax.set_xlim(0, WIDTH)
    image_ax.set_ylim(HEIGHT, 0)
    image_ax.set_aspect("equal")
    image_ax.set_xlabel("u（像素）")
    image_ax.set_ylabel("v（像素）")
    image_ax.grid(alpha=0.18)
    image_ax.scatter(U_IDEAL, V_IDEAL, s=48, facecolors="none", edgecolors="gray",
                     linewidths=1.5, label="理想针孔位置")
    image_ax.scatter([CX], [CY], marker="+", s=130, linewidths=2.5,
                     color="tab:red", label="主点")
    distorted_scatter = image_ax.scatter([], [], marker="x", s=42, linewidths=1.8,
                                         color="tab:blue", label="有畸变的位置")
    connectors = LineCollection([], colors="tab:blue", linewidths=1.0, alpha=0.60)
    image_ax.add_collection(connectors)
    image_ax.legend(loc="upper right")

    residual_scatter = residual_ax.scatter([], [], s=32, color="tab:blue")
    residual_ax.set_xlim(-0.02, 0.78)
    residual_ax.set_ylim(-2, 90)
    residual_ax.set_xlabel("离主点的归一化距离")
    residual_ax.set_ylabel("像素偏移量")
    residual_ax.grid(alpha=0.25)

    status = fig.text(0.5, 0.125, "", ha="center")
    slider = Slider(
        fig.add_axes([0.28, 0.055, 0.56, 0.03]),
        "径向畸变 k1", -0.50, 0.50, valinit=-0.35, valstep=0.01, valfmt="%+.2f"
    )

    def update(k1):
        u_distorted, v_distorted, radius = distort_points(k1)
        distorted_scatter.set_offsets(np.column_stack([u_distorted, v_distorted]))
        connectors.set_segments([
            [(u0, v0), (u1, v1)]
            for u0, v0, u1, v1 in zip(U_IDEAL, V_IDEAL, u_distorted, v_distorted)
        ])
        pixel_shift = np.hypot(u_distorted - U_IDEAL, v_distorted - V_IDEAL)
        residual_scatter.set_offsets(np.column_stack([radius, pixel_shift]))
        direction = "向主点内缩（桶形趋势）" if k1 < 0 else (
            "向图像外扩（枕形趋势）" if k1 > 0 else "无径向畸变"
        )
        image_ax.set_title(f"k1={k1:+.2f}：蓝叉相对灰圈{direction}")
        status.set_text(
            f"中心附近偏移 {pixel_shift[np.argmin(radius)]:.2f} px；"
            f"最远角点偏移 {pixel_shift[np.argmax(radius)]:.2f} px"
        )
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(slider.val)
    return fig, slider


if __name__ == "__main__":
    figure, distortion_slider = build_figure()
    plt.show()
