"""L06教师辅助图：读取既有视差并换算；不匹配、不读参考、不计学生完成。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SCENE = ROOT / "data/eth3d/delivery_area_1s"


def main():
    photo = np.array(Image.open(SCENE / "im0.png"))
    with np.load(OUT / "predictions/delivery_area_1s.npz") as saved:
        disparity = saved["classroom"].astype(np.float64)
    calibration = dict(line.split("=", 1) for line in
                       (SCENE / "calib.txt").read_text().splitlines() if "=" in line)
    focal_px = float(calibration["cam0"].lstrip("[").split()[0])
    baseline_m = float(calibration["baseline"]) / 1000
    assert float(calibration["doffs"]) == 0
    assert photo.shape[:2] == disparity.shape == (435, 711)

    # 保存前已按接口排除无效哨兵，并换为像素单位；这里不能再除以16。
    valid = np.isfinite(disparity) & (disparity > 0)
    depth = np.full(disparity.shape, np.nan, dtype=np.float64)
    depth[valid] = focal_px * baseline_m / disparity[valid]
    assert np.array_equal(np.isfinite(depth), valid)
    assert np.all(depth[valid] > 0)
    assert abs(depth[350, 450] - 2.126536) < 0.000001

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(17.4, 6.7), facecolor="white")
    fig.suptitle("一个点约 2.13 米，怎样变成一张深度图", fontsize=23, y=.96)
    fig.text(.5, .875, "同一张左照片的位置不变：每个可用位置的视差，分别换成前向米数",
             ha="center", fontsize=15, color="#425164")
    axes = [fig.add_axes([x, .29, .277, .48]) for x in (.047, .371, .695)]
    cmap_d = plt.get_cmap("viridis").copy()
    cmap_z = plt.get_cmap("viridis_r").copy()
    for cmap in (cmap_d, cmap_z):
        cmap.set_bad("#b8b8b8")
    axes[0].imshow(photo)
    im_d = axes[1].imshow(np.ma.masked_invalid(disparity), cmap=cmap_d,
                          vmin=0, vmax=24, interpolation="nearest")
    im_z = axes[2].imshow(np.ma.masked_invalid(depth), cmap=cmap_z,
                          vmin=0, vmax=24, interpolation="nearest")
    titles = ["已有左照片：木箱与后方结构", "已有程序估计：视差", "换算后的深度：教师预览"]
    labels = ["A：箱面上的点", "A：20.0625 像素", "A：约 2.13 米"]
    for ax, title, label in zip(axes, titles, labels):
        ax.set_title(title, fontsize=14, pad=12)
        ax.set_xlim(-.5, 710.5)
        ax.set_ylim(434.5, -.5)
        ax.set_xticks([0, 200, 400, 600])
        ax.set_yticks([0, 100, 200, 300, 400])
        ax.tick_params(labelsize=10)
        ax.set_xlabel("横向位置 u（像素）", fontsize=11)
        ax.set_ylabel("纵向位置 v（像素）", fontsize=11)
        ax.plot(450, 350, "o", ms=9, mfc="none", mec="#ff623b", mew=2)
        ax.annotate(label, xy=(450, 350), xytext=(290, 270),
                    color="white", fontsize=12,
                    bbox=dict(boxstyle="round,pad=.35", facecolor="#172330", alpha=.9),
                    arrowprops=dict(arrowstyle="->", color="#ff623b", linewidth=1.8))
    for x, im, label in ((.371, im_d, "视差（像素）：紫色小 → 黄色大"),
                         (.695, im_z, "前向深度（米）：黄色近 → 紫色远")):
        cax = fig.add_axes([x, .16, .277, .025])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 4, 8, 12, 16, 20, 24])
        cb.set_label(label, fontsize=11)
        cb.ax.tick_params(labelsize=10)
    fig.text(.047, .18, "A 在三幅图中均为\n(u, v) = (450, 350)", fontsize=13,
             color="#243548", linespacing=1.6)
    fig.text(.5, .035, "中、右图灰色 = 未知（NaN）   |   深度由程序视差换算，参考答案未参与   |   本图为教师备课，学生实现与运行待完成",
             ha="center", fontsize=11, color="#465160")
    output = OUT / "wood_depth_map_teacher_preview.png"
    fig.savefig(output, dpi=130)
    plt.close(fig)
    print(f"教师预览：{output.relative_to(ROOT)}")
    print(f"A 深度：{depth[350, 450]:.6f} m")
    print(f"可换算：{valid.sum()}；未知：{(~valid).sum()}；形状：{depth.shape}")
    print(f"数值范围：{depth[valid].min():.6f}—{depth[valid].max():.6f} m")


if __name__ == "__main__":
    main()
