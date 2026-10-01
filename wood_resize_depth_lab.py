"""L06第5步：同一A点、同一对应关系，检查像素尺度与米制深度。

把左右照片每个像素复制为2x2块；不重新匹配，不改变原数据。
新视差来自原对应坐标的尺度变换，不冒充新分辨率下的匹配结果。
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from wood_depth_map_lab import load_inputs, to_depth_m
from wood_point_backprojection_lab import backproject

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "notes/eth3d_validation/resize_depth_lab"


def prepare():
    photo, disparity, focal, baseline = load_inputs()
    scene = ROOT / "data/eth3d/delivery_area_1s"
    calib = dict(line.split("=", 1) for line in
                 (scene / "calib.txt").read_text().splitlines() if "=" in line)
    k = np.array([[float(v) for v in row.split()]
                  for row in calib["cam0"].strip("[]").split(";")])
    assert float(calib["doffs"]) == 0
    scale = 2
    # 原像素中心u变成新2x2像素块中心：u'=2*(u+0.5)-0.5。
    transform = np.array([[scale, 0, (scale-1)/2],
                          [0, scale, (scale-1)/2], [0, 0, 1.]])
    k_scaled = transform @ k
    u, v = 450., 350.
    d = float(disparity[int(v), int(u)])
    left_uv = np.array([u, v, 1.])
    right_uv = np.array([u-d, v, 1.])
    new_left = transform @ left_uv
    new_right = transform @ right_uv
    d_scaled = float(new_left[0]-new_right[0])

    # 本轮核心：f和d都是像素单位，必须对应同一图像尺度。
    original_z = to_depth_m(d, focal, baseline)
    scaled_z = to_depth_m(d_scaled, k_scaled[0, 0], baseline)
    wrong_z = to_depth_m(d_scaled, focal, baseline)  # 故意漏改f，其他条件固定。
    original_xyz = backproject(u, v, original_z, k[0, 0], k[1, 1], k[0, 2], k[1, 2])
    scaled_xyz = backproject(new_left[0], new_left[1], scaled_z,
                             k_scaled[0, 0], k_scaled[1, 1], k_scaled[0, 2], k_scaled[1, 2])
    assert d_scaled == scale*d
    assert np.allclose(original_xyz, scaled_xyz, atol=1e-12, rtol=0)
    assert np.isclose(wrong_z, original_z/scale, atol=1e-12, rtol=0)
    OUT.mkdir(exist_ok=True)
    enlarged = np.repeat(np.repeat(photo, scale, axis=0), scale, axis=1)
    right = np.asarray(Image.open(scene / "im1.png"))
    enlarged_right = np.repeat(np.repeat(right, scale, axis=0), scale, axis=1)
    Image.fromarray(enlarged).save(OUT / "left_2x.png")
    Image.fromarray(enlarged_right).save(OUT / "right_2x.png")
    report = dict(scale=scale, original_size_wh=[photo.shape[1], photo.shape[0]],
                  scaled_size_wh=[enlarged.shape[1], enlarged.shape[0]],
                  focal_px=[focal, float(k_scaled[0, 0])],
                  disparity_px=[d, d_scaled], baseline_m=baseline,
                  original_z_m=original_z, correct_scaled_z_m=scaled_z,
                  wrong_old_focal_z_m=wrong_z,
                  original_left_uv=left_uv[:2].tolist(), scaled_left_uv=new_left[:2].tolist(),
                  original_right_uv=right_uv[:2].tolist(), scaled_right_uv=new_right[:2].tolist(),
                  original_K=k.tolist(), scaled_K=k_scaled.tolist(),
                  original_xyz_m=original_xyz.tolist(), scaled_xyz_m=scaled_xyz.tolist(),
                  note="Same estimated correspondence transformed analytically; no new matching or reference input.",
                  coordinate_convention="Pixel centres: u_new=s*(u_old+0.5)-0.5; same for v.")
    (OUT / "checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return photo, enlarged, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-only", action="store_true")
    args = parser.parse_args()
    photo, enlarged, report = prepare()
    import matplotlib
    if args.save_only:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 8))
    fig.subplots_adjust(left=.065, right=.965, bottom=.42, top=.80, wspace=.20)
    fig.suptitle("照片像素变多，木箱的米制深度应当不变", fontsize=22, y=.96)
    fig.text(.5, .885, "同一木箱A点、同一估计对应关系、同一相机基线；左右图宽高均放大2倍",
             ha="center", fontsize=13, color="#42516a")
    for ax, img, title, uv in zip(axes, [photo, enlarged],
                                ["原左照片：711 × 435", "放大后的左照片：1422 × 870"],
                                [report["original_left_uv"], report["scaled_left_uv"]]):
        ax.imshow(img)
        ax.plot(*uv, "o", ms=11, mfc="none", mec="#ff623b", mew=2.5)
        ax.set(title=title, xlabel="u（当前图像的像素）", ylabel="v（当前图像的像素）")
        ax.text(.035, .09, "橙圈：同一个箱面A点", transform=ax.transAxes, color="white", fontsize=11,
                bbox=dict(fc="#172330", alpha=.9, boxstyle="round,pad=.3"))
    ax_table = fig.add_axes([.075, .16, .85, .19])
    ax_table.axis("off")
    table = ax_table.table(
        cellText=[
            ["原图", f"{report['focal_px'][0]:.3f}", f"{report['disparity_px'][0]:.4f}", f"{report['original_z_m']:.6f}"],
            ["放大2倍：f与d一起更新", f"{report['focal_px'][1]:.3f}", f"{report['disparity_px'][1]:.4f}", f"{report['correct_scaled_z_m']:.6f}"],
            ["故意出错：d更新，f沿用旧值", f"{report['focal_px'][0]:.3f}", f"{report['disparity_px'][1]:.4f}", f"{report['wrong_old_focal_z_m']:.6f}"],
        ], colLabels=["条件", "焦距 f（像素）", "视差 d（像素）", "计算深度 Z（米）"],
        colWidths=[.40, .20, .20, .20], cellLoc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#c8d3de")
        cell.set_facecolor({0: "#e9eef4", 2: "#e7f4ef", 3: "#fff0e9"}.get(row, "white"))
    fig.text(.5, .09, "Z = fB / d；基线 B 始终为 0.0599701 米。照片并排采用相同显示尺寸，请看各自坐标刻度。",
             ha="center", fontsize=11, color="#42516a")
    fig.text(.5, .035, "新视差由原对应坐标放大得到，未重新匹配；复制像素不会增加真实拍摄细节。",
             ha="center", fontsize=11, color="#42516a")
    fig.savefig(OUT / "comparison.png", dpi=140)
    print(f"原图A深度：{report['original_z_m']:.6f} 米")
    print(f"放大2倍并同步更新：{report['correct_scaled_z_m']:.6f} 米")
    print(f"故意漏改焦距：{report['wrong_old_focal_z_m']:.6f} 米")
    print("正确更新后XYZ与原值一致；这仅验证几何尺度自洽。")
    if args.save_only:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
