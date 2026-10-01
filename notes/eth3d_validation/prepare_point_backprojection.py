"""教师备图：同一个 A 像素对应的左相机坐标；不读取参考或重新匹配。"""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def main():
    scene = ROOT / "data/eth3d/delivery_area_1s"
    calib = dict(line.split("=", 1) for line in
                 (scene / "calib.txt").read_text().splitlines() if "=" in line)
    k = np.array([[float(v) for v in row.split()]
                  for row in calib["cam0"].strip("[]").split(";")])
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    assert float(calib["doffs"]) == 0
    baseline_m = float(calib["baseline"]) / 1000
    u, v = 450, 350
    with np.load(OUT / "predictions/delivery_area_1s.npz") as saved:
        disparity = float(saved["classroom"][v, u])
    assert np.isfinite(disparity) and disparity > 0
    z = fx * baseline_m / disparity
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    reprojection = [fx * x / z + cx, fy * y / z + cy]
    assert np.allclose(reprojection, [u, v], atol=1e-10, rtol=0)
    result = dict(status="teacher_preparation_only", pixel_uv=[u, v],
                  principal_point_px=[cx, cy], focal_px=[fx, fy],
                  disparity_px=disparity, xyz_m=[x, y, z],
                  reprojected_uv=reprojection,
                  note="Reprojection checks algebraic consistency, not accuracy.")
    (OUT / "wood_point_backprojection_check.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, (photo, space) = plt.subplots(1, 2, figsize=(14, 7.2),
                                     gridspec_kw={"width_ratios": [1.12, 1]})
    fig.subplots_adjust(left=.055, right=.97, top=.80, bottom=.21, wspace=.17)
    fig.suptitle("同一个 A：从照片中的位置，得到相机前的三维位置", fontsize=21, y=.96)
    fig.text(.5, .885, "沿用原照片、原估计视差与标定；这次只换坐标", ha="center", fontsize=14, color="#42516a")
    photo.imshow(Image.open(scene / "im0.png"))
    photo.set_title("左照片：坐标单位为像素", fontsize=15, pad=12)
    photo.set_xlabel("u 向右（像素）")
    photo.set_ylabel("v 向下（像素）")
    photo.plot(cx, cy, "+", color="#57dcff", ms=16, mew=2.5)
    photo.plot(u, v, "o", color="#ff7348", ms=10, mfc="none", mew=2.5)
    photo.plot([cx, u, u], [cy, cy, v], "--", color="#ffdd65", lw=2)
    box = dict(boxstyle="round,pad=.35", fc="#172330", alpha=.94)
    photo.annotate("光轴在照片上的位置\n标定给出 (376.739, 222.302)",
                   (cx, cy), (75, 85), color="white", fontsize=11, bbox=box,
                   arrowprops=dict(arrowstyle="->", color="#57dcff", lw=1.7))
    photo.annotate("A：箱面上的点\n(450, 350)", (u, v), (480, 300),
                   color="white", fontsize=11, bbox=box,
                   arrowprops=dict(arrowstyle="->", color="#ff7348", lw=1.7))
    photo.set_xlim(-.5, 710.5)
    photo.set_ylim(434.5, -.5)

    # Oblique schematic with deliberately expanded lateral dimensions.
    # Coordinates below are page positions, never substituted into geometry.
    origin = np.array([1.05, 1.65])
    forward = np.array([1.35, 2.4])
    right = np.array([2.05, 0.0])
    down = np.array([0.0, -1.15])
    foot = origin + forward
    corner = foot + right
    a = corner + down
    for end, label, pos in [
        (origin + [3.65, 0], "X：向右", (4.1, 1.22)),
        (origin + [0, -1.35], "Y：向下", (.35, .03)),
        (origin + forward * 1.32, "Z：向前", (2.05, 5.02)),
    ]:
        space.annotate("", end, origin, arrowprops=dict(arrowstyle="->", color="#68788c", lw=1.8))
        space.text(*pos, label, fontsize=12, color="#42516a")
    space.plot(*origin, "ko", ms=7)
    space.text(.05, 1.90, "左相机光心\n(0, 0, 0)", fontsize=12)
    space.plot([origin[0], a[0]], [origin[1], a[1]], color="#ff7348", lw=1.7, alpha=.65)
    space.plot([origin[0], foot[0]], [origin[1], foot[1]], color="#32a7c4", lw=3)
    space.plot([foot[0], corner[0]], [foot[1], corner[1]], "--", color="#32a7c4", lw=2)
    space.plot([corner[0], a[0]], [corner[1], a[1]], "--", color="#32a7c4", lw=2)
    space.plot(*a, "o", color="#ff7348", ms=10)
    space.text(a[0]+.13, a[1]-.12, "A", fontsize=17, color="#d14e28")
    space.text(1.58, 2.85, f"向前\n{z:.3f} 米", fontsize=12, ha="right", color="#11687b")
    space.text(3.35, 4.20, f"向右 {x:.3f} 米", fontsize=12, ha="center", color="#11687b")
    space.text(4.58, 3.48, f"向下\n{y:.3f} 米", fontsize=12, color="#11687b")
    space.set(xlim=(-.15, 5.75), ylim=(-.15, 5.4))
    space.set_aspect("equal")
    space.axis("off")
    space.set_title("相机坐标：单位为米\n方向示意，横向偏移放大显示", fontsize=14, pad=5)
    fig.text(.5, .13, f"A 的估计位置：(X, Y, Z) ≈ ({x:.3f}, {y:.3f}, {z:.3f}) 米",
             ha="center", fontsize=18, color="#172330")
    fig.text(.5, .055, "蓝色十字来自标定；橙色 A 的深度来自原 SGBM 估计。这里没有使用参考答案。",
             ha="center", fontsize=12, color="#42516a")
    fig.savefig(OUT / "wood_point_backprojection.png", dpi=150)
    plt.close(fig)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
