"""L06单点交互：点击原左照片，查看一个估计三维点及回投位置。

教师提供读文件、交互与反投影实现；学生运行不等于独立实现。
沿用原classroom视差，不匹配、不读参考、不生成整幅点云。
"""
import argparse
from pathlib import Path

import numpy as np
from wood_depth_map_lab import load_inputs, to_depth_m

ROOT = Path(__file__).resolve().parent


def backproject(u, v, z, fx, fy, cx, cy):
    """核心几何：像素偏移 / 像素焦距 × 前向深度 = 米制偏移。"""
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z], dtype=np.float64)


def inspect_point(u, v, disparity, k, baseline_m):
    d = float(disparity[v, u])
    if not np.isfinite(d) or d <= 0:
        return None
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    z = to_depth_m(d, fx, baseline_m)
    xyz = backproject(u, v, z, fx, fy, cx, cy)
    x, y, z = xyz
    uv = np.array([fx*x/z + cx, fy*y/z + cy])
    return d, xyz, uv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-only", action="store_true")
    parser.add_argument("--point", choices=["A", "B"], default="A")
    args = parser.parse_args()
    import matplotlib
    if args.save_only:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    photo, disparity, _, baseline_m = load_inputs()
    calib = dict(line.split("=", 1) for line in
                 (ROOT / "data/eth3d/delivery_area_1s/calib.txt").read_text().splitlines()
                 if "=" in line)
    k = np.array([[float(v) for v in row.split()]
                  for row in calib["cam0"].strip("[]").split(";")])
    cx, cy = k[0, 2], k[1, 2]
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(14, 7.6))
    fig.suptitle("选一个像素 → 查看它的估计三维位置", fontsize=21, y=.96)
    fig.text(.5, .905, "左：原照片，单位像素    |    右：左相机坐标，单位米（一次只显示一个点）",
             ha="center", fontsize=13, color="#42516a")
    ax_photo = fig.add_axes([.06, .30, .45, .54])
    ax_space = fig.add_axes([.56, .27, .39, .58], projection="3d")
    ax_photo.imshow(photo)
    ax_photo.plot(cx, cy, "+", color="#38c7ff", ms=15, mew=2)
    ax_photo.axvline(cx, color="#38c7ff", ls="--", lw=1, alpha=.7)
    ax_photo.axhline(cy, color="#38c7ff", ls="--", lw=1, alpha=.7)
    ax_photo.set(xlabel="u 向右（像素）", ylabel="v 向下（像素）")
    for label, u, v in [("A：箱面", 450, 350), ("B：后方立柱", 530, 130)]:
        ax_photo.text(u+12, v-16, label, fontsize=11, color="white",
                      bbox=dict(fc="#172330", alpha=.85, boxstyle="round,pad=.25"))
    marker, = ax_photo.plot([], [], "o", ms=11, mfc="none", mec="#ff6e40", mew=2.5)
    status = fig.text(.07, .17, "", fontsize=13, color="#172330", linespacing=1.6)
    fig.text(.5, .035, "蓝色十字：标定光轴落点  |  橙色：当前选点  |  灰色射线：观察方向  |  无可用视差时不生成三维点",
             ha="center", fontsize=10, color="#42516a")

    def update(u, v):
        marker.set_data([u], [v])
        ax_photo.set_title(f"当前像素 ({u}, {v})；可直接点击照片", fontsize=13)
        ax_space.clear()
        # Matplotlib display axes: horizontal X, receding Z, vertical Y.
        # Invert vertical Y so positive values point downward as in the camera frame.
        ax_space.set(xlabel="X 向右（米）", ylabel="Z 向前（米）", zlabel="Y 向下（米）",
                     xlim=(-2, 3), ylim=(0, 10), zlim=(2, -2))
        ax_space.set_box_aspect((5, 10, 4))
        ax_space.view_init(elev=18, azim=-65)
        ax_space.scatter([0], [0], [0], color="#172330", s=35)
        ax_space.text(0, 0, 0, "  相机", fontsize=10)
        result = inspect_point(u, v, disparity, k, baseline_m)
        if result is None:
            message = f"像素 ({u}, {v})：原视差未知或不能换算有限正深度。\n没有三维点；未知不当作零米。"
            ax_space.set_title("当前点未知", fontsize=13)
        else:
            d, (x, y, z), uv = result
            ax_space.plot([0, x], [0, z], [0, y], color="#89949e", lw=1.5)
            ax_space.scatter([x], [z], [y], color="#ff6e40", s=65)
            ax_space.text(x, z, y, "  当前点", fontsize=10)
            # Keep comparison scales fixed; explicitly flag points outside this view.
            outside = not (-2 <= x <= 3 and 0 <= z <= 10 and -2 <= y <= 2)
            ax_space.set_title("超出当前固定显示范围：以读数为准" if outside else "固定坐标范围，可拖动查看方向",
                               fontsize=12)
            message = (f"像素 ({u}, {v})   估计视差 {d:.4f} 像素\n"
                       f"X = {x:+.6f} 米    Y = {y:+.6f} 米    Z = {z:.6f} 米\n"
                       f"回投 = ({uv[0]:.3f}, {uv[1]:.3f}) 像素；仅检查计算自洽")
        status.set_text(message)
        print(message, flush=True)
        fig.canvas.draw_idle()

    def onclick(event):
        if event.inaxes is not ax_photo or event.button != 1:
            return
        toolbar = getattr(fig.canvas.manager, "toolbar", None)
        if toolbar is not None and toolbar.mode:
            return
        if event.xdata is None or event.ydata is None:
            return
        u, v = int(np.floor(event.xdata+.5)), int(np.floor(event.ydata+.5))
        if 0 <= u < disparity.shape[1] and 0 <= v < disparity.shape[0]:
            update(u, v)

    fig.canvas.mpl_connect("button_press_event", onclick)
    button_a = Button(fig.add_axes([.64, .105, .12, .055]), "选 A：箱面")
    button_b = Button(fig.add_axes([.79, .105, .15, .055]), "选 B：后方立柱")
    button_a.on_clicked(lambda event: update(450, 350))
    button_b.on_clicked(lambda event: update(530, 130))
    update(*(dict(A=(450, 350), B=(530, 130))[args.point]))
    if args.save_only:
        out = ROOT / "notes/eth3d_validation/point_backprojection_lab"
        out.mkdir(exist_ok=True)
        fig.savefig(out / f"teacher_preview_{args.point}.png", dpi=120)
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
