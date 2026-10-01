import numpy as np
import matplotlib.pyplot as plt

# 空间坐标 X、Z 都用米；焦距用像素，投影结果也用像素。
focal_px = 600.0
baseline_m = 0.10
disparity_error_px = -1.0  # 教师调整：视差偏小 1 像素；正值偏大，0 为无误差。

# 使用中文标签；若系统没有雅黑，Matplotlib 会尝试后面的字体。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def project_x(X, Z):
    """返回相对图像中心的横向位置，不是数组列号。"""
    return focal_px*(X/Z)


def show_projection(points, projected, projected_right, recovered):
    """画出计算得到的位置，以及用于定位的左右投影。"""
    fig = plt.figure(figsize=(10, 7), constrained_layout=True)
    fig.suptitle(f"双目定位实验：视差测量偏差 {disparity_error_px:+.1f} 像素")
    layout = fig.add_gridspec(2, 2, height_ratios=[3, 1.3])
    scene = fig.add_subplot(layout[0, :])
    scene.scatter([0, baseline_m], [0, 0], color="black", marker="^", s=65)
    scene.annotate("左相机（原点）", (0, 0), xytext=(-8, -22),
                   textcoords="offset points", ha="right")
    scene.annotate("右相机", (baseline_m, 0), xytext=(8, -22),
                   textcoords="offset points", ha="left")
    scene.plot([0, baseline_m], [0, 0], color="black", linewidth=1)
    colors = ["tab:blue", "tab:orange"]

    # 空心圆仅作真值对照；彩色叉号的位置来自恢复计算，不直接使用 points。
    scene.scatter(points[:, 0], points[:, 1], s=160, facecolors="none",
                  edgecolors="gray", linewidths=1.5, label="设定位置（用于对照）")
    for i, (X_est, Z_est) in enumerate(recovered):
        color = colors[i % len(colors)]
        scene.plot([0, X_est], [0, Z_est], "--", color=color, alpha=0.55)
        scene.plot([baseline_m, X_est], [0, Z_est], ":", color=color, alpha=0.75)
        scene.scatter([X_est], [Z_est], marker="x", s=110, linewidths=2.5,
                      color=color, label=f"P{i + 1} 计算出的定位")
        horizontal = f"右侧 {X_est:.2f} 米" if X_est >= 0 else f"左侧 {-X_est:.2f} 米"
        depth_error_cm = (Z_est - points[i, 1]) * 100
        scene.annotate(f"P{i + 1} 定位结果\n{horizontal}，前方 {Z_est:.2f} 米"
                       f"\n深度误差：{depth_error_cm:+.1f} 厘米",
                       (X_est, Z_est), xytext=(18, 24), textcoords="offset points",
                       va="bottom")

    # 每台相机画两行，分别显示两个候选点，避免重合时遮住另一个点。
    # 这是投影位置示意，不是同时拍摄两个实体点的真实照片。
    max_offset = max(abs(x) for x in projected + projected_right)
    half_width = max(60, int(np.ceil(max_offset)) + 20)
    row_centers = np.arange(len(points)) * 21 + 10
    for camera_index, (positions, title) in enumerate([
        (projected, "左相机看到的位置"),
        (projected_right, "右图测量位置（已加入模拟误差）"),
    ]):
        image = np.zeros((21 * len(points), 2 * half_width + 1), dtype=np.uint8)
        for row, x in zip(row_centers, positions):
            image[row, int(round(x)) + half_width] = 255
        ax = fig.add_subplot(layout[1, camera_index])
        ax.imshow(image, cmap="gray", vmin=0, vmax=255,
                  interpolation="nearest", origin="upper",
                  extent=(-half_width - 0.5, half_width + 0.5,
                          image.shape[0] - 0.5, -0.5), aspect="auto")
        ax.axvline(0, color="tab:red", linestyle="--", linewidth=1)
        for i, (row, x) in enumerate(zip(row_centers, positions)):
            ax.scatter([x], [row], s=36, color=colors[i % len(colors)])
        ax.set_title(title)
        ax.set_xlabel("相对图像中心的位置（像素；红线是中心）")
        ax.set_yticks(row_centers, [f"P{i + 1}" for i in range(len(points))])

    all_X = np.concatenate((points[:, 0], recovered[:, 0], [0, baseline_m]))
    all_Z = np.concatenate((points[:, 1], recovered[:, 1], [0]))
    x_margin = max(0.12, float(np.ptp(all_X)) * 0.5)
    z_margin = max(0.8, float(np.ptp(all_Z)) * 0.32)
    scene.set_xlim(float(all_X.min()) - x_margin,
                   float(all_X.max()) + x_margin * 2.3)
    scene.set_ylim(-z_margin, float(all_Z.max()) + z_margin)
    scene.set_xlabel("相对左相机的左右位置 X（米；正数表示右侧）")
    scene.set_ylabel("相对左相机的前方深度 Z（米）")
    scene.set_title("俯视定位图：圆圈是真实设定位置，叉号是测量后算出的定位（横纵比例不同）")
    scene.grid(alpha=0.3)
    scene.legend()
    plt.show()


if __name__ == "__main__":
    # 每一行是一个候选空间点的 (X, Z)。分别投影，不模拟遮挡。
    points = np.array([[0.10, 2.0], [0.30, 4.0]])
    projected = []
    projected_right = []
    recovered = []
    for i, (X, Z) in enumerate(points, start=1):
        x = project_x(X, Z)
        X_right = X - baseline_m
        x_right_true = project_x(X_right, Z)
        # 左图测量保持不变，右图测量位置减去设定的视差偏差。
        # 因为视差=左图位置-右图位置，最终视差偏差等于 disparity_error_px。
        x_right = x_right_true - disparity_error_px
        projected.append(x)
        projected_right.append(x_right)
        disparity = x - x_right
        if not np.isfinite(disparity) or disparity <= 0:
            raise ValueError("当前理想双目模型要求正且有限的视差，请检查计算")
        depth = focal_px * baseline_m / disparity

        # 先从视差求深度，再用左图位置反推出横向位置。
        # 定位计算只使用左右投影、焦距及基线，不读取原始 X、Z。
        X_est = x * depth / focal_px
        recovered.append([X_est, depth])
        horizontal = f"右侧 {X_est:.2f} 米" if X_est >= 0 else f"左侧 {-X_est:.2f} 米"
        print(f"P{i} 定位结果：在左相机{horizontal}，前方 {depth:.2f} 米")
        print(f"    真实深度 {Z:.2f} 米；深度误差 {(depth - Z) * 100:+.1f} 厘米")
    show_projection(points, projected, projected_right, np.array(recovered))
