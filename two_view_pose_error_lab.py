"""L08：深度固定，只把第2帧用于拼接的相机位置记错。

复用两视角合成观测，错误只加在地图Z方向的位置分量上。
此为人为误差对照，无新匹配、位姿估计、去噪或融合。
"""
import argparse
import json
from pathlib import Path
import numpy as np
from two_view_mapping_lab import build_views, LOW

OUT = Path(__file__).resolve().parent / "notes/synthetic_mapping/pose_error"


def place_second(view, error_m):
    """观测和旋转固定，只改用于拼接的相机位置。"""
    recorded_position = view["camera"] + np.array([0., 0., error_m])
    return view["camera_points"] @ view["R_wc"].T + recorded_position


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-only", action="store_true")
    args = parser.parse_args()
    first, second = build_views()
    first_points = first["world_points"]
    correct = place_second(second, 0.)
    wrong = place_second(second, .15)
    second_front = second["face"][second["valid"]] == 4
    assert np.allclose(correct, second["world_points"], atol=1e-12, rtol=0)
    assert np.allclose(wrong-correct, [0., 0., .15], atol=1e-12, rtol=0)
    assert np.allclose(wrong[second_front, 2], LOW[2]+.15, atol=1e-12, rtol=0)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "comparison.npz", first_world=first_points,
                        second_correct_world=correct, second_wrong_world=wrong,
                        second_camera_points=second["camera_points"],
                        second_depth=second["depth"], R_wc=second["R_wc"],
                        true_t_wc=second["camera"], error_t_wc=np.array([0., 0., .15]),
                        second_front_mask=second_front)
    report = dict(kind="teacher-set pose translation error with unchanged ideal observations",
                  pose_error_world_m=[0., 0., .15], true_camera_world_m=second["camera"].tolist(),
                  wrong_camera_world_m=(second["camera"]+[0., 0., .15]).tolist(),
                  front_reference_z_m=float(LOW[2]), front_wrong_z_m=float(LOW[2]+.15),
                  second_front_mean_abs_plane_residual_m=float(np.abs(wrong[second_front, 2]-LOW[2]).mean()),
                  changed="Only second camera translation used for placement; depth and rotation fixed.",
                  note="Single pair demonstration, not a drift sequence or pose estimator evaluation.")
    (OUT / "checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    import matplotlib
    if args.save_only:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(14, 9.5))
    fig.suptitle("深度都正确，地图也可能出现重影", fontsize=22, y=.97)
    fig.text(.5, .913, "固定两帧观测和相机朝向；只把第2帧的相机位置沿地图Z方向记错",
             ha="center", fontsize=13, color="#42516a")
    fig.text(.5, .87, "本图颜色表示帧来源：蓝色 = 第1帧，橙色 = 第2帧（与上一图的面颜色含义不同）",
             ha="center", fontsize=11, color="#42516a")
    cloud_axes = [fig.add_axes([left, .39, .40, .43], projection="3d") for left in [.065, .55]]
    slice_axes = [fig.add_axes([left, .22, .32, .10]) for left in [.11, .60]]
    status = fig.text(.5, .158, "", ha="center", fontsize=12, color="#42516a")

    def draw(ax, panel, points, title, preserve=False):
        elev, azim = (ax.elev, ax.azim) if preserve else (22, -53)
        ax.clear()
        for p, color in [(first_points[::3], "#008bc7"), (points[::3], "#ff841c")]:
            ax.scatter(p[:, 0], p[:, 2], p[:, 1], c=color, s=3, alpha=.65,
                       linewidths=0, depthshade=False)
        ax.set(xlabel="地图X（米）", ylabel="地图Z（米）", zlabel="Y向下（米）",
               xlim=(-1, 1), ylim=(2.9, 5.1), zlim=(.8, -.8), title=title)
        ax.set_box_aspect((2, 2.2, 1.6))
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([3, 4, 5]); ax.set_zticks([-.6, 0, .6])
        ax.view_init(elev=elev, azim=azim)
        # 只俯视两帧实际观测的正面采样，单独显出法向错位。
        panel.clear()
        p1, p2 = first_points[::8], points[second_front][::8]
        panel.scatter(p1[:, 0], p1[:, 2], s=7, c="#008bc7", alpha=.7)
        panel.scatter(p2[:, 0], p2[:, 2], s=7, c="#ff841c", alpha=.7)
        panel.set(xlim=(-1, 1), ylim=(3.02, 3.48), xlabel="地图X（米）", ylabel="地图Z（米）")
        panel.set_xticks([-1, 0, 1]); panel.set_yticks([3.1, 3.25, 3.4])
        panel.set_title("只看正面位置：俯视", fontsize=10, pad=6)
        panel.grid(alpha=.2)

    draw(cloud_axes[0], slice_axes[0], correct, "位置记录正确：两帧正面落在一起")
    def update(error_m):
        moved = place_second(second, error_m)
        draw(cloud_axes[1], slice_axes[1], moved,
             f"第2帧位置多记 {error_m*100:.0f} 厘米", preserve=True)
        residual = np.abs(moved[second_front, 2]-LOW[2]).mean()
        status.set_text(f"正面在地图中的Z：第1帧 3.10 米，第2帧 {LOW[2]+error_m:.2f} 米；"
                        f"两层相差 {residual*100:.0f} 厘米。")
        fig.canvas.draw_idle()
    cloud_axes[1].view_init(elev=22, azim=-53)
    slider = Slider(fig.add_axes([.24, .103, .37, .022]), "位置多记（米）", 0., .30,
                    valinit=.15, valstep=.01, valfmt="%.2f")
    slider.on_changed(update)
    reset = Button(fig.add_axes([.67, .079, .12, .052]), "恢复正确位置")
    preset = Button(fig.add_axes([.82, .079, .12, .052]), "记错15厘米")
    reset.on_clicked(lambda event: slider.set_val(0.))
    preset.on_clicked(lambda event: slider.set_val(.15))
    fig.text(.5, .035, "合成理想深度 + 人为位姿误差；相机实际没再移动，错误仅发生在放入地图时的位置记录。",
             ha="center", fontsize=10, color="#42516a")
    update(.15)
    fig.savefig(OUT / "comparison.png", dpi=130)
    print("深度和朝向保持不变；第2帧地图Z位置故意多记0.15米。")
    print(f"第二帧正面到真实平面的平均距离：{report['second_front_mean_abs_plane_residual_m']:.6f} 米")
    if args.save_only:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
