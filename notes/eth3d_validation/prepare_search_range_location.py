"""教师示意：0—127约束匹配时的候选视差；只叠加标注，不匹配或改照片。"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "data/eth3d/delivery_area_1s"
OUT = Path(__file__).resolve().parent


def main():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.3))
    fig.subplots_adjust(left=.065, right=.975, bottom=.30, top=.80, wspace=.18)
    fig.suptitle("0～127 限制的是：寻找对应点时，尝试的横向位移", fontsize=21, y=.97)
    fig.text(.5, .885, "左右照片 →【匹配：在这里使用候选视差范围】→ 已有视差图 → 深度换算",
             ha="center", fontsize=15, color="#12675e")
    for ax, name in zip(axes, ("im0.png", "im1.png")):
        ax.imshow(Image.open(SCENE / name))
        ax.set_xlim(-.5, 710.5)
        ax.set_ylim(434.5, -.5)
        ax.set_xticks([0, 100, 200, 300, 400, 500, 600, 700])
        ax.set_yticks([0, 100, 200, 300, 400])
        ax.set_xlabel("横向位置 u（像素）", fontsize=12)
        ax.set_ylabel("纵向位置 v（像素）", fontsize=12)
    axes[0].set_title("左照片：这次固定箱面 A 点", fontsize=15, pad=12)
    axes[0].plot(450, 350, "o", ms=12, mfc="none", mec="#ff6e40", mew=2.5)
    axes[0].annotate("A：u左 = 450，v = 350", (450, 350), (160, 270),
                     fontsize=13, color="white",
                     bbox=dict(boxstyle="round,pad=.4", fc="#172330", alpha=.94),
                     arrowprops=dict(arrowstyle="->", color="#ff6e40", lw=2))
    axes[1].set_title("右照片：在黄线范围内尝试对应位置", fontsize=15, pad=12)
    axes[1].plot([323, 450], [350, 350], color="#ffdf3d", linewidth=6, solid_capstyle="butt")
    for u, label, xytext, color in [
        (323, "试 d = 127\n右图 u = 323", (110, 190), "#ffdf3d"),
        (430, "试 d = 20\n右图 u = 430", (335, 75), "#53d8ff"),
        (450, "试 d = 0\n右图 u = 450", (520, 200), "#ff9c80"),
    ]:
        axes[1].plot(u, 350, "o", ms=7, mfc=color, mec="#172330")
        axes[1].annotate(label, (u, 350), xytext, fontsize=12, color="white",
                         bbox=dict(boxstyle="round,pad=.4", fc="#172330", alpha=.94),
                         arrowprops=dict(arrowstyle="->", color=color, lw=1.8))
    fig.text(.5, .185, "固定左图 A 后，每尝试一个视差 d，就去右图 u右 = 450 − d 的位置比较附近图像块。",
             ha="center", fontsize=15)
    fig.text(.5, .112, "所以 d = 0～127，对这个 A 点对应的是右图 u = 450～323；三个标记只示意候选位置。",
             ha="center", fontsize=13, color="#465160")
    fig.text(.5, .05, "黄线标候选中心的范围；照片、相机和物体都未移动。本图不重跑匹配，不改变已有视差或深度。",
             ha="center", fontsize=11, color="#465160")
    path = OUT / "search_range_at_A.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
