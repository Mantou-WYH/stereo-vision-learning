"""L05：同一对照片，只改变候选视差范围。

展示已经真实运行并独立评分的两组SGBM结果，不修改算法、不按参考调参。
--save-only 用于教师保存并检查图，其余情况下显示窗口。
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from wood_point_depth_lab import read_reference


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/eth3d/delivery_area_1s"
OUTPUT = ROOT / "notes/eth3d_validation"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-only", action="store_true")
    args = parser.parse_args()
    left = np.asarray(Image.open(DATA / "im0.png"))
    with np.load(OUTPUT / "predictions/delivery_area_1s.npz") as saved:
        wide, restricted = saved["classroom"], saved["aloe_range"]
    reference = read_reference(DATA / "disp0GT.pfm")
    mask = np.asarray(Image.open(DATA / "mask0nocc.png"))
    assert left.shape[:2] == wide.shape == restricted.shape == reference.shape
    # 按物理场景选择背景立柱和前方箱面，教学展示不代替全图评价。
    points = [("B", 530, 130, "后方立柱表面"), ("A", 450, 350, "前方木箱表面")]
    for name, u, v, description in points:
        assert np.isfinite(reference[v, u]) and mask[v, u] == 255
        print(f"{name} {description} (u={u}, v={v})：参考 {reference[v,u]:.4f} px；"
              f"0—127估计 {wide[v,u]:.4f} px；16—111估计 {restricted[v,u]:.4f} px")
    assert abs(float(reference[130, 530]) - 6) < .1
    assert wide[130, 530] == 6 and restricted[130, 530] == 16
    assert wide[350, 450] == restricted[350, 450] == 20.0625

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#b8b8b8")
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.4))
    fig.subplots_adjust(left=.055, right=.915, top=.81, bottom=.29, wspace=.18)
    axes[0].imshow(left)
    axes[0].set_title("同一场景的左相机照片\n实际匹配同时输入右相机照片", fontsize=12)
    for ax, values, title in zip(axes[1:], [wide, restricted],
                                ["允许视差 0—127", "允许视差 16—111"]):
        im = ax.imshow(values, cmap=cmap, vmin=0, vmax=32, interpolation="nearest")
        ax.set_title(title + "\n灰色：没有有效估计", fontsize=12)
    for ax in axes:
        ax.set_xlabel("横向位置 u（像素）")
        ax.set_ylabel("纵向位置 v（像素）")
        for name, u, v, _ in points:
            ax.plot(u, v, "o", ms=9, markerfacecolor="none", markeredgecolor="white", mew=1.8)
            ax.annotate(name, (u, v), xytext=(u-35, v-30), fontsize=12,
                        weight="bold", color="white",
                        bbox=dict(facecolor="#222222", edgecolor="none", alpha=.85, pad=2),
                        arrowprops=dict(arrowstyle="-", color="white", linewidth=1.2))
    bar = fig.add_axes([.935, .34, .012, .42])
    fig.colorbar(im, cax=bar, extend="max", label="视差（像素）；超过32使用顶端颜色")
    fig.suptitle("L05：如果正确答案不在搜索范围里，程序会怎样？", fontsize=17, y=.96)
    fig.text(.5, .195, "先看 B：参考 5.92 像素  ｜  0—127 得到 6.00  ｜  16—111 得到 16.00（错误）",
             ha="center", fontsize=13, weight="bold", color="#9b3c23")
    fig.text(.5, .13, "再看 A：参考 20.12 像素  ｜  两组都得到 20.06（此点的候选仍在范围内）",
             ha="center", fontsize=12, color="#333333")
    fig.text(.5, .067, "固定照片、分辨率与其余匹配参数；只改变候选范围。参考只用于事后核对，不进入匹配。",
             ha="center", fontsize=10, color="#555555")
    fig.text(.5, .025, "这是已保存的真实运行结果。一个点的表现不代表全图；两组都可能有错误或未知。",
             ha="center", fontsize=10, color="#555555")
    fig.savefig(OUTPUT / "search_range_lesson.png", dpi=130)
    if args.save_only:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
