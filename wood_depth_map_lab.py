"""L06：把已有木箱视差换成整幅前向深度，未知保留未知。

核心换算表达式由学生提交；教师负责筛选、写回、加载与绘图。
课堂关注图像位置、米制深度和未知区域，不做连续基础语法填空。
不重新匹配，不读取参考答案，不改原照片或预测。
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "data/eth3d/delivery_area_1s"
RESULTS = ROOT / "notes/eth3d_validation"


def to_depth_m(d_px, focal_px, baseline_m):
    """学生已提交的表达式；输入应是已选出的有限正视差。"""
    return (focal_px * baseline_m) / d_px


def select_valid(d_px):
    """本例已解码视差：有限且大于0的位置可以换算。"""
    finite = np.isfinite(d_px)  # 每个位置是否是有限数，NaN/无穷为False。
    positive = d_px > 0        # 每个位置的视差是否大于0。

    return finite & positive


def make_depth_map(d_px, valid, focal_px, baseline_m):
    """按原位置写回米数；缺少可用视差的位置仍为NaN。"""
    depth_m = np.full(d_px.shape, np.nan, dtype=np.float64)
    depth_m[valid] = to_depth_m(d_px[valid], focal_px, baseline_m)
    return depth_m


def load_inputs():
    """教师辅助：保留原生分辨率；已保存视差无需再次除以16。"""
    photo = np.array(Image.open(SCENE / "im0.png"))
    with np.load(RESULTS / "predictions/delivery_area_1s.npz") as saved:
        d_px = saved["classroom"].astype(np.float64)
    calibration = dict(line.split("=", 1) for line in
                       (SCENE / "calib.txt").read_text().splitlines() if "=" in line)
    assert float(calibration["doffs"]) == 0
    focal_px = float(calibration["cam0"].lstrip("[").split()[0])
    baseline_m = float(calibration["baseline"]) / 1000
    assert photo.shape[:2] == d_px.shape == (435, 711)
    return photo, d_px, focal_px, baseline_m


def show_result(photo, d_px, depth_m, save_only=False):
    """教师辅助：三图同位置、各自色条；灰色表示未知。"""
    import matplotlib
    if save_only:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(17.4, 6.7), facecolor="white")
    fig.suptitle("已有木箱视差 → 整幅前向深度", fontsize=23, y=.96)
    fig.text(.5, .875, "同一位置：外观 → 估计视差（像素）→ 前向深度（米）",
             ha="center", fontsize=15, color="#425164")
    axes = [fig.add_axes([x, .29, .277, .48]) for x in (.047, .371, .695)]
    cmap_d = plt.get_cmap("viridis").copy()
    cmap_z = plt.get_cmap("viridis_r").copy()
    for cmap in (cmap_d, cmap_z):
        cmap.set_bad("#b8b8b8")
    axes[0].imshow(photo)
    im_d = axes[1].imshow(np.ma.masked_invalid(d_px), cmap=cmap_d,
                          vmin=0, vmax=24, interpolation="nearest")
    im_z = axes[2].imshow(np.ma.masked_invalid(depth_m), cmap=cmap_z,
                          vmin=0, vmax=24, interpolation="nearest")
    titles = ["已有左照片", "已有程序估计：视差", "本次换算：前向深度"]
    labels = ["A：箱面上的点", f"A：{d_px[350, 450]:.4f} 像素",
              f"A：{depth_m[350, 450]:.2f} 米"]
    background_labels = ["B：后方立柱", f"B：{d_px[130, 530]:.2f} 像素",
                         f"B：{depth_m[130, 530]:.2f} 米"]
    for ax, title, label, label_b in zip(axes, titles, labels, background_labels):
        ax.set_title(title, fontsize=14, pad=12)
        ax.set_xlim(-.5, 710.5)
        ax.set_ylim(434.5, -.5)
        ax.set_xticks([0, 200, 400, 600])
        ax.set_yticks([0, 100, 200, 300, 400])
        ax.set_xlabel("横向位置 u（像素）", fontsize=11)
        ax.set_ylabel("纵向位置 v（像素）", fontsize=11)
        ax.plot(450, 350, "o", ms=9, mfc="none", mec="#ff623b", mew=2)
        ax.annotate(label, xy=(450, 350), xytext=(290, 270),
                    color="white", fontsize=12,
                    bbox=dict(boxstyle="round,pad=.35", facecolor="#172330", alpha=.9),
                    arrowprops=dict(arrowstyle="->", color="#ff623b", linewidth=1.8))
        ax.plot(530, 130, "o", ms=9, mfc="none", mec="#53d8ff", mew=2)
        ax.annotate(label_b, xy=(530, 130), xytext=(300, 60),
                    color="white", fontsize=12,
                    bbox=dict(boxstyle="round,pad=.35", facecolor="#172330", alpha=.9),
                    arrowprops=dict(arrowstyle="->", color="#53d8ff", linewidth=1.8))
    for x, im, label in ((.371, im_d, "视差（像素）：紫色小 → 黄色大"),
                         (.695, im_z, "前向深度（米）：黄色近 → 紫色远")):
        cax = fig.add_axes([x, .16, .277, .025])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 4, 8, 12, 16, 20, 24])
        cb.set_label(label, fontsize=11)
    fig.text(.047, .18, "同一位置 (u, v)\nA = (450, 350)；B = (530, 130)", fontsize=12,
             color="#243548", linespacing=1.6)
    fig.text(.5, .035, "中、右图灰色 = 未知（NaN）   |   参考答案未参与生成   |   本图保留已有匹配误差",
             ha="center", fontsize=11, color="#465160")
    output = RESULTS / "wood_depth_map_classroom.png"
    fig.savefig(output, dpi=130)
    print(f"本次图像：{output.relative_to(ROOT)}")
    if save_only:
        plt.close(fig)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-only", action="store_true")
    args = parser.parse_args()
    photo, d_px, focal_px, baseline_m = load_inputs()
    valid = select_valid(d_px)
    depth_m = make_depth_map(d_px, valid, focal_px, baseline_m)
    assert valid.shape == depth_m.shape == d_px.shape
    assert valid.dtype == np.bool_
    assert np.array_equal(np.isfinite(depth_m), valid)
    assert np.all(depth_m[valid] > 0)
    np.save(RESULTS / "wood_depth_map_classroom.npy", depth_m)
    print(f"A点前向深度：{depth_m[350, 450]:.6f} 米")
    print(f"B点前向深度：{depth_m[130, 530]:.6f} 米")
    print(f"深度数组形状：{depth_m.shape}（行，列）；单位：米")
    print(f"有深度的位置：{valid.sum()}；未知位置：{(~valid).sum()}")
    show_result(photo, d_px, depth_m, save_only=args.save_only)


if __name__ == "__main__":
    main()
