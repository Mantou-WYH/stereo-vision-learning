"""真实木箱上的一个点：沿用已验证视差，不改 SGBM 或重新调参。

直接运行显示图；--save-only 只保存教学图，供教师检查。
照片、预测、标定均保持原始 711×435 分辨率。
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "data/eth3d/delivery_area_1s"
RESULTS = ROOT / "notes/eth3d_validation"


def read_reference(path):
    """读取本数据集的单通道 PFM；只用于核对，不参与预测。"""
    with path.open("rb") as stream:
        assert stream.readline().strip() == b"Pf"
        width, height = map(int, stream.readline().split())
        scale = float(stream.readline())
        assert abs(scale) == 1.0
        values = np.frombuffer(stream.read(), dtype="<f4" if scale < 0 else ">f4")
    return np.flipud(values.reshape(height, width))


def depth_from_disparity(disparity_px, focal_px, baseline_m):
    """适用于本例 doffs=0：返回前向深度 Z（米），不是斜向直线距离。"""
    if not np.isfinite(disparity_px) or disparity_px <= 0:
        raise ValueError("本例只将有限正视差换为有限正深度，不把缺失值填成 0。")
    return focal_px * baseline_m / disparity_px


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-only", action="store_true")
    args = parser.parse_args()

    # 固定项：同一对照片、同一份已验证的 SGBM 结果、木箱面上的 A 点。
    u_left, v = 450, 350
    left = np.array(Image.open(SCENE / "im0.png"))
    right = np.array(Image.open(SCENE / "im1.png"))
    with np.load(RESULTS / "predictions/delivery_area_1s.npz") as saved:
        prediction = saved["classroom"]
    assert left.shape[:2] == right.shape[:2] == prediction.shape == (435, 711)

    # 配套 calib.txt 遵循 Middlebury v3：baseline 以毫米记录。
    calibration = dict(line.split("=", 1) for line in
                       (SCENE / "calib.txt").read_text().splitlines() if "=" in line)
    assert float(calibration["doffs"]) == 0
    focal_px = float(calibration["cam0"].lstrip("[").split()[0])
    baseline_m = float(calibration["baseline"]) / 1000

    disparity_px = float(prediction[v, u_left])
    u_right = u_left - disparity_px
    estimated_z = depth_from_disparity(disparity_px, focal_px, baseline_m)

    # 预测完成后才读标准答案，不能拿参考答案冒充程序估计。
    reference_d = float(read_reference(SCENE / "disp0GT.pfm")[v, u_left])
    reference_z = depth_from_disparity(reference_d, focal_px, baseline_m)
    assert np.array(Image.open(SCENE / "mask0nocc.png"))[v, u_left] == 255
    print(f"A 点左图坐标 (u, v) = ({u_left}, {v})")
    print(f"程序估计视差：{disparity_px:.4f} 像素；对应右图 u = {u_right:.4f}")
    print(f"焦距：{focal_px} 像素；基线：{baseline_m} 米")
    print(f"程序估计前向深度：{estimated_z:.6f} 米")
    print(f"参考视差：{reference_d:.6f} 像素；换算前向深度：{reference_z:.6f} 米")
    print("注意：这里只核对一个箱面点，不代表整张图都达到相同精度。")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    fig.subplots_adjust(left=.055, right=.985, top=.80, bottom=.25, wspace=.16)
    fig.suptitle("这次只测箱面上的 A 点：从图像位置差，换算到前向深度", fontsize=17, y=.965)
    for ax, photo, u, title in zip(
            axes, [left, right], [u_left, u_right],
            [f"左相机照片：A 在 u={u_left}, v={v}",
             f"右相机照片：程序选在 u≈{u_right:.2f}, v={v}"]):
        ax.imshow(photo, cmap="gray", vmin=0, vmax=255)
        ax.axhline(v, color="#f9d65c", alpha=.75, linewidth=1, linestyle="--")
        ax.plot(u, v, "o", markersize=12, markerfacecolor="none",
                markeredgecolor="#ff6e40", markeredgewidth=2.3)
        ax.annotate("A：木箱表面的一点", xy=(u, v), xytext=(u-165, v-110),
                    color="white", fontsize=12,
                    bbox=dict(boxstyle="round,pad=.3", facecolor="#222222", alpha=.87),
                    arrowprops=dict(arrowstyle="->", color="#ff6e40", linewidth=1.8))
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel("横向像素位置 u")
        ax.set_ylabel("纵向像素位置 v")

    fig.text(.5, .15, f"程序估计：视差 {disparity_px:.4f} 像素  →  A 点前向深度 {estimated_z:.2f} 米",
             ha="center", fontsize=16, weight="bold", color="#17635b")
    fig.text(.5, .087, f"独立核对：数据集参考视差换算得 {reference_z:.2f} 米（参考不参与匹配）",
             ha="center", fontsize=12, color="#444444")
    fig.text(.5, .03, "匹配使用原始分辨率；橙圈指对应位置，黄虚线指同一行。这里只检查 A，不替整幅图保证精度。",
             ha="center", fontsize=10, color="#555555")
    output = RESULTS / "wood_point_depth.png"
    fig.savefig(output, dpi=140)
    print(f"教学图：{output.relative_to(ROOT)}")
    if args.save_only:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
