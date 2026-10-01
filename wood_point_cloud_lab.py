"""L06：已有整幅前向深度 -> 左相机坐标中的单帧点云。

沿用原classroom深度，不重新匹配、不读取参考、不补洞、不去噪。
教师提供批量整理、保存和绘图；学生运行不等于独立实现。
"""
import argparse
import json
from pathlib import Path

import numpy as np
from wood_depth_map_lab import load_inputs, select_valid, make_depth_map
from wood_point_backprojection_lab import backproject

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "notes/eth3d_validation/point_cloud_lab"


def build_cloud():
    photo, disparity, focal_px, baseline_m = load_inputs()
    depth_path = ROOT / "notes/eth3d_validation/wood_depth_map_classroom.npy"
    if not depth_path.exists():
        raise FileNotFoundError("请先运行已有 wood_depth_map_lab.py，保存原整幅深度。")
    depth = np.load(depth_path)
    expected = make_depth_map(disparity, select_valid(disparity), focal_px, baseline_m)
    if depth.shape != disparity.shape or not np.allclose(depth, expected, equal_nan=True,
                                                       rtol=1e-12, atol=1e-12):
        raise ValueError("保存的深度与原classroom基线不一致；请检查来源，未自动替换。")
    calib = dict(line.split("=", 1) for line in
                 (ROOT / "data/eth3d/delivery_area_1s/calib.txt").read_text().splitlines()
                 if "=" in line)
    k = np.array([[float(value) for value in row.split()]
                  for row in calib["cam0"].strip("[]").split(";")])
    valid = np.isfinite(depth) & (depth > 0)
    v, u = np.nonzero(valid)
    z = depth[v, u]
    # 与单点完全相同的几何；u、v、z现在各自是一列有效位置的数据。
    xyz = backproject(u, v, z, k[0, 0], k[1, 1], k[0, 2], k[1, 2]).T
    uv = np.column_stack((u, v))
    rgb = photo[v, u]
    if rgb.ndim == 1:
        rgb = np.repeat(rgb[:, None], 3, axis=1)
    rgb = rgb[:, :3].astype(np.uint8)
    reprojected = np.column_stack((k[0, 0]*xyz[:, 0]/xyz[:, 2]+k[0, 2],
                                  k[1, 1]*xyz[:, 1]/xyz[:, 2]+k[1, 2]))
    max_error = float(np.max(np.abs(reprojected-uv)))
    assert np.isfinite(xyz).all() and np.all(xyz[:, 2] > 0)
    assert len(xyz) == int(valid.sum())
    assert max_error < 1e-9
    return photo, xyz, uv, rgb, valid, max_error


def show_cloud(photo, xyz, uv, rgb, valid, save_only=False, full=False):
    import matplotlib
    if save_only:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(15, 8))
    fig.suptitle("从每个有效像素，到相机前的一片表面", fontsize=22, y=.965)
    fig.text(.5, .905, "沿用原估计深度  |  每个点的外观取自左照片同一像素  |  未知位置不生成点",
             ha="center", fontsize=13, color="#42516a")
    left = fig.add_axes([.05, .33, .40, .48])
    space = fig.add_axes([.49, .22, .47, .64], projection="3d", computed_zorder=False)
    left.imshow(photo)
    left.set(xlabel="u（像素）", ylabel="v（像素）", title="原左照片：同一木箱与后方立柱")
    anchors = []
    for label, (u, v), color, offset in [
        ("A：箱面", (450, 350), "#ff623b", (-150, -45)),
        ("B：后方立柱", (530, 130), "#00a3de", (-170, -70)),
    ]:
        left.plot(u, v, "o", ms=10, mfc="none", mec=color, mew=2.5)
        left.annotate(label, (u, v), (u+offset[0], v+offset[1]), fontsize=11,
                      color="white", bbox=dict(fc="#172330", alpha=.9, boxstyle="round,pad=.3"),
                      arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
        index = np.flatnonzero((uv[:, 0] == u) & (uv[:, 1] == v))
        assert len(index) == 1
        anchors.append((label, xyz[index[0]], color))
    fig.text(.055, .22, f"整幅生成 {len(xyz):,} 个点\n保留 {valid.size-valid.sum():,} 个未知位置",
             fontsize=15, color="#172330", linespacing=1.5)
    fig.text(.055, .105, "X 向右，Y 向下，Z 向前，单位均为米。\n拖动右图可改变观察方向；两枚按钮恢复预设视图。",
             fontsize=11, color="#42516a", linespacing=1.6)
    info = fig.text(.5, .04, "", ha="center", fontsize=10, color="#42516a")
    # 仅显示按原图每隔3像素取样的点；保存文件始终保留全部有效位置。
    sampled = (uv[:, 0] % 3 == 0) & (uv[:, 1] % 3 == 0)

    def redraw(show_full):
        space.clear()
        if show_full:
            low = np.minimum(xyz.min(axis=0), 0)
            high = np.maximum(xyz.max(axis=0), 0)
            margin = np.maximum((high-low)*.035, .1)
            low, high = low-margin, high+margin
            in_view = np.ones(len(xyz), dtype=bool)
            title = "全部范围：包含原估计中的远点与异常形状"
        else:
            low, high = np.array([-2., -2., 0.]), np.array([3., 2., 10.])
            in_view = ((xyz >= low) & (xyz <= high)).all(axis=1)
            title = "观察木箱与立柱：只调整显示范围"
        shown = sampled & in_view
        points = xyz[shown]
        # 显示轴顺序为X、Z、Y，Y轴反向，保持正Y向下。
        space.scatter(points[:, 0], points[:, 2], points[:, 1],
                      c=rgb[shown]/255., s=2, linewidths=0, depthshade=False, zorder=1)
        for label, (x, y, z), color in anchors:
            space.scatter([x], [z], [y], c=color, s=70, edgecolors="white",
                          linewidths=1, depthshade=False, zorder=10)
            space.text(x, z, y, "  "+label[0], color=color, fontsize=14,
                       bbox=dict(fc="white", alpha=.8, ec="none", pad=1), zorder=12)
        space.scatter([0], [0], [0], color="#172330", s=35)
        space.text(0, 0, 0, "  左相机", fontsize=10)
        space.set(xlabel="X 向右（米）", ylabel="Z 向前（米）", zlabel="Y 向下（米）",
                  xlim=(low[0], high[0]), ylim=(low[2], high[2]),
                  zlim=(high[1], low[1]), title=title)
        space.set_box_aspect((high-low)[[0, 2, 1]])
        space.view_init(elev=16, azim=-62)
        info.set_text(f"保存全部 {len(xyz):,} 点；显示每隔3像素取样的 {shown.sum():,} 点（另标A/B）；"
                      f"{len(xyz)-in_view.sum():,} 点在当前观察范围外。无去噪或补洞。")
        fig.canvas.draw_idle()

    near_button = Button(fig.add_axes([.57, .12, .16, .055]), "木箱与立柱")
    full_button = Button(fig.add_axes([.77, .12, .16, .055]), "全部范围")
    near_button.on_clicked(lambda event: redraw(False))
    full_button.on_clicked(lambda event: redraw(True))
    redraw(full)
    if save_only:
        fig.savefig(OUT / ("teacher_preview_full.png" if full else "teacher_preview_near.png"), dpi=130)
        plt.close(fig)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-only", action="store_true")
    parser.add_argument("--full-view", action="store_true")
    args = parser.parse_args()
    photo, xyz, uv, rgb, valid, error = build_cloud()
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / "wood_cloud_classroom.npz", xyz_m=xyz, pixel_uv=uv,
                        rgb_u8=rgb, valid_mask=valid)
    report = dict(points=len(xyz), unknown_pixels=int(valid.size-valid.sum()),
                  camera_axes="X right, Y down, Z forward", units="metres",
                  source_depth="notes/eth3d_validation/wood_depth_map_classroom.npy",
                  source_prediction="classroom (original)",
                  maximum_reprojection_error_px=error,
                  xyz_min_m=xyz.min(axis=0).tolist(), xyz_max_m=xyz.max(axis=0).tolist(),
                  note="All valid points saved. Display subsampling and view bounds only; no accuracy claim.")
    (OUT / "checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"生成点数：{len(xyz):,}；未知位置：{valid.size-valid.sum():,}")
    print(f"最大回投坐标差：{error:.3e} 像素（仅检查计算自洽）")
    print("全部点已保存：notes/eth3d_validation/point_cloud_lab/wood_cloud_classroom.npz")
    show_cloud(photo, xyz, uv, rgb, valid, args.save_only, args.full_view)


if __name__ == "__main__":
    main()
