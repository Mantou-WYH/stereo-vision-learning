"""L01：固定相机间距和焦距，观察视差随深度变化；不加测量误差。"""
import numpy as np
import matplotlib.pyplot as plt
from projection_lab import project_x, focal_px, baseline_m


def disparity_at_depth(Z):
    # 固定物体横向位置，只改变前方深度。复用学生写过的投影函数。
    X = 0.10
    left = project_x(X, Z)
    right = project_x(X - baseline_m, Z)
    return left - right


if __name__ == "__main__":
    depths = np.linspace(1.0, 8.0, 200)
    disparities = disparity_at_depth(depths)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(depths, disparities, color="tab:blue", linewidth=2)
    for Z in [1.0, 2.0, 4.0]:
        d = disparity_at_depth(Z)
        ax.scatter([Z], [d], s=65, color="tab:orange", zorder=3)
        ax.annotate(f"{Z:g} 米，{d:g} 像素", (Z, d), xytext=(12, 10),
                    textcoords="offset points")
        print(f"前方 {Z:g} 米 → 视差 {d:g} 像素")
    ax.set_title(f"无误差实验：物体距离与视差\n"
                 f"焦距固定 {focal_px:g} 像素，相机间距固定 {baseline_m:g} 米")
    ax.set_xlabel("物体前方深度 Z（米）")
    ax.set_ylabel("左右位置差 d（像素）")
    ax.set_xlim(0.5, 8.5)
    ax.set_ylim(0, 72)
    ax.grid(alpha=0.3)
    plt.show()
