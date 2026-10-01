"""L03：固定三维点和焦距，观察错误主点造成的整体像素平移。"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.collections import LineCollection
from calibration_focal_error_lab import true_fx, cx, Z, X, u_true


def projected_with_principal_error(cx_error_px):
    used_cx = cx + cx_error_px
    return used_cx, true_fx * X / Z + used_cx


def build_figure():
    fig, (image_ax, residual_ax) = plt.subplots(2, 1, figsize=(9, 7),
                                               gridspec_kw={"height_ratios": [2.2, 1.4]})
    fig.subplots_adjust(left=0.11, right=0.95, top=0.88, bottom=0.25, hspace=0.48)
    fig.suptitle("故意使用错误主点：三维点不动，焦距保持正确", y=0.97)
    image_ax.set_xlim(0, 640)
    image_ax.set_ylim(-0.7, 0.7)
    true_center = image_ax.axvline(cx, color="tab:red", linestyle="--", linewidth=1,
                                  label="真实主点")
    used_center = image_ax.axvline(cx, color="tab:purple", linestyle=":", linewidth=2,
                                  label="使用的主点")
    image_ax.scatter(u_true, np.zeros_like(u_true), s=90, facecolors="none",
                     edgecolors="gray", linewidths=2, label="正确投影")
    wrong_scatter = image_ax.scatter([], [], marker="x", s=70, linewidths=2,
                                     color="tab:blue", label="错误主点投影")
    connectors = LineCollection([], colors="tab:blue", linewidths=1.5, alpha=0.75)
    image_ax.add_collection(connectors)
    image_ax.set_yticks([])
    image_ax.set_xlabel("图像横坐标 u（像素）")
    image_ax.legend(loc="upper left")

    residual_line, = residual_ax.plot(X, np.zeros_like(X), "o-", color="tab:blue")
    residual_ax.axhline(0, color="gray", linewidth=1)
    residual_ax.set_xlim(-0.66, 0.66)
    residual_ax.set_ylim(-45, 45)
    residual_ax.set_xlabel("空间点的横向位置 X（米）")
    residual_ax.set_ylabel("像素偏差 u错误 − u真实（像素）")
    residual_ax.grid(alpha=0.25)

    status = fig.text(0.5, 0.14, "", ha="center")
    slider = Slider(fig.add_axes([0.29, 0.055, 0.52, 0.03]),
                    "主点 cx 误差（像素）", -40, 40, valinit=20, valstep=1, valfmt="%+.0f")

    def update(cx_error_px):
        used_cx, u_wrong = projected_with_principal_error(cx_error_px)
        used_center.set_xdata([used_cx, used_cx])
        wrong_scatter.set_offsets(np.column_stack([u_wrong, np.zeros_like(u_wrong)]))
        connectors.set_segments([[(a, 0), (b, 0)] for a, b in zip(u_true, u_wrong)])
        residual = u_wrong - u_true
        residual_line.set_ydata(residual)
        image_ax.set_title(f"真实 cx={cx:.0f} px，使用 cx={used_cx:.0f} px：所有点整体平移")
        status.set_text(f"七个点的像素偏差均为 {cx_error_px:+.0f} px；"
                        "误差不随点离主点的距离变化")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(slider.val)
    return fig, slider


if __name__ == "__main__":
    figure, principal_error_slider = build_figure()
    plt.show()
