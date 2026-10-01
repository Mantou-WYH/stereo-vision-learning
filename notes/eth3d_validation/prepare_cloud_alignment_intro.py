"""建图导入：同一木箱点集在两个相机坐标系中的表达。

第二组由已知设定位姿换算，不是第二次实拍或新的深度估计。
只示意坐标统一；不模拟遮挡，不增加表面，不证明融合精度。
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "cloud_alignment_intro"


def main():
    with np.load(ROOT / "notes/eth3d_validation/point_cloud_lab/wood_cloud_classroom.npz") as data:
        xyz, uv = data["xyz_m"], data["pixel_uv"]
    selected = ((uv[:, 0] % 6 == 0) & (uv[:, 1] % 6 == 0) &
                (xyz[:, 0] > -1.5) & (xyz[:, 0] < 2.5) &
                (xyz[:, 1] > -1.6) & (xyz[:, 1] < 1.3) & (xyz[:, 2] < 9))
    first = xyz[selected]
    angle = np.deg2rad(8.)
    c, s = np.cos(angle), np.sin(angle)
    # R_wc把第二个相机的局部方向转到第一相机/地图方向。
    r_wc = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    t_wc = np.array([1., 0., 0.])
    second_local = (first - t_wc) @ r_wc
    aligned = second_local @ r_wc.T + t_wc
    error = float(np.linalg.norm(aligned-first, axis=1).max())
    assert np.allclose(r_wc.T@r_wc, np.eye(3), atol=1e-12)
    assert error < 1e-12
    OUT.mkdir(exist_ok=True)
    (OUT / "checks.json").write_text(json.dumps(dict(
        kind="simulated coordinate transform of the same saved point set",
        points=int(len(first)), R_wc=r_wc.tolist(), t_wc_m=t_wc.tolist(),
        yaw_degrees=8, maximum_recovery_difference_m=error,
        note="No second capture, new surface, pose estimation, visibility model or fusion."),
        ensure_ascii=False, indent=2), encoding="utf-8")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(14.5, 7.5))
    fig.suptitle("建图的第一步：让各帧点云使用同一套坐标", fontsize=22, y=.95)
    fig.text(.5, .875, "蓝色：原木箱点集   橙色：同一组点在下一拍摄位置的坐标表达（模拟）",
             ha="center", fontsize=13, color="#42516a")
    axes = [fig.add_axes([left, .23, .42, .59], projection="3d", computed_zorder=False)
            for left in [.055, .525]]
    for ax, second, title in zip(axes, [second_local, aligned],
                                 ["直接叠加：坐标原点和朝向混用", "按已知相机位置与朝向变换后"]):
        ax.scatter(first[:, 0], first[:, 2], first[:, 1], c="#007bb8", s=3,
                   alpha=.75, linewidths=0, depthshade=False, zorder=1)
        ax.scatter(second[:, 0], second[:, 2], second[:, 1], c="#ff861a", s=3,
                   alpha=.55, linewidths=0, depthshade=False, zorder=2)
        ax.set(xlabel="X（米）", ylabel="Z（米）", zlabel="Y 向下（米）",
               xlim=(-4, 3), ylim=(0, 10), zlim=(1.5, -1.8), title=title)
        ax.set_box_aspect((7, 10, 3.3))
        ax.view_init(elev=26, azim=-72)
    fig.text(.275, .175, "同一表面被画成两份，出现错位", ha="center", fontsize=14, color="#42516a")
    fig.text(.745, .175, "恢复到共同坐标，两个点集重合", ha="center", fontsize=14, color="#42516a")
    fig.text(.5, .10, "教师设定：整套双目相机右移1米、右转8°；物体不动。每个时刻以左相机为坐标原点。",
             ha="center", fontsize=11, color="#42516a")
    fig.text(.5, .045, "这里只演示坐标统一；第二组不是新实拍，重合不代表已实现真实多帧融合或补出缺失表面。",
             ha="center", fontsize=11, color="#42516a")
    fig.savefig(OUT / "comparison.png", dpi=140)
    plt.close(fig)
    print(f"Simulated points: {len(first)}; maximum recovery difference: {error:.3e} m")


if __name__ == "__main__":
    main()
