"""L03：固定三维点，观察错误焦距造成的系统性像素偏差。"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.collections import LineCollection

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

true_fx = 600.0
cx = 320.0
Z = 2.0
X = np.linspace(-0.60, 0.60, 7)
u_true = true_fx * X / Z + cx


def projected_with_focal_error(error_percent):
    used_fx = true_fx * (1.0 + error_percent / 100.0)
    u_wrong = used_fx * X / Z + cx
    return used_fx, u_wrong


def build_figure():
    fig, (image_ax, residual_ax) = plt.subplots(2, 1, figsize=(9, 7),
                                               gridspec_kw={"height_ratios": [2.2, 1.4]})
    fig.subplots_adjust(left=0.11, right=0.95, top=0.88, bottom=0.25, hspace=0.48)
    fig.suptitle("故意使用错误焦距：三维点不动，只改变标定参数", y=0.97)

    image_ax.set_xlim(0, 640)
    image_ax.set_ylim(-0.7, 0.7)
    image_ax.axvline(cx, color="tab:red", linestyle="--", linewidth=1, label="主点 cx")
    image_ax.scatter(u_true, np.zeros_like(u_true), s=90, facecolors="none",
                     edgecolors="gray", linewidths=2, label="真实焦距投影")
    wrong_scatter = image_ax.scatter([], [], marker="x", s=70, linewidths=2,
                                     color="tab:blue", label="错误焦距投影")
    connectors = LineCollection([], colors="tab:blue", linewidths=1.5, alpha=0.75)
    image_ax.add_collection(connectors)
    image_ax.set_yticks([])
    image_ax.set_xlabel("图像横坐标 u（像素）")
    image_ax.legend(loc="upper left")

    residual_line, = residual_ax.plot(X, np.zeros_like(X), "o-", color="tab:blue")
    residual_ax.axhline(0, color="gray", linewidth=1)
    residual_ax.set_xlim(-0.66, 0.66)
    residual_ax.set_ylim(-25, 25)
    residual_ax.set_xlabel("空间点的横向位置 X（米）")
    residual_ax.set_ylabel("像素偏差 u错误 − u真实（像素）")
    residual_ax.grid(alpha=0.25)

    status = fig.text(0.5, 0.14, "", ha="center")
    slider = Slider(fig.add_axes([0.29, 0.055, 0.52, 0.03]),
                    "使用的焦距误差（%）", -20, 20, valinit=10, valstep=1, valfmt="%+.0f")

    def update(error_percent):
        used_fx, u_wrong = projected_with_focal_error(error_percent)
        wrong_scatter.set_offsets(np.column_stack([u_wrong, np.zeros_like(u_wrong)]))
        connectors.set_segments([[(a, 0), (b, 0)] for a, b in zip(u_true, u_wrong)])
        residual = u_wrong - u_true
        residual_line.set_ydata(residual)
        max_abs = max(5.0, float(np.max(np.abs(residual))) * 1.35)
        residual_ax.set_ylim(-max_abs, max_abs)
        direction = "外扩，远离主点" if error_percent > 0 else "内缩，靠近主点"
        if np.isclose(error_percent, 0):
            direction = "与真实投影重合"
        image_ax.set_title(f"使用 fx={used_fx:.0f} px（真实值 {true_fx:.0f} px）：{direction}")
        status.set_text(f"最外侧点的最大像素偏差：{np.max(np.abs(residual)):.1f} px；"
                        "点的真实三维位置始终不变")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(slider.val)
    return fig, slider


if __name__ == "__main__":
    figure, focal_error_slider = build_figure()
    plt.show()
