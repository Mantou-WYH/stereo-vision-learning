"""L02：用已知三维点模拟一帧双目点云，检查三维坐标的正逆变换。

这里的点不是由单张单目图像直接得到；后续课程会由同一时刻的左右图像
匹配和三角测量生成三维点。本实验只隔离验证坐标变换。
"""
from itertools import product
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def cube_points(center=(0.0, 0.0, 3.0), side=1.0):
    offsets = np.array(list(product([-0.5, 0.5], repeat=3))) * side
    return offsets + np.asarray(center)


def camera_pose():
    """相机在世界中的位置与朝向：向右转 20 度。"""
    theta = np.deg2rad(20.0)
    c, s = np.cos(theta), np.sin(theta)
    R_wc = np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])
    t_wc = np.array([0.8, 0.0, 0.3])
    return R_wc, t_wc


def world_to_camera(points_world, R_wc, t_wc):
    """每行是一个三维点；先减相机位置，再使用逆旋转。"""
    return (R_wc.T @ (points_world - t_wc).T).T


def camera_to_world(points_camera, R_wc, t_wc):
    """先转回世界方向，再加相机位置。"""
    return (R_wc @ points_camera.T).T + t_wc


def make_T_wc(R_wc, t_wc):
    """组装相机坐标到世界坐标的 4×4 齐次变换。"""
    T_wc = np.eye(4)
    T_wc[:3, :3] = R_wc
    T_wc[:3, 3] = t_wc
    return T_wc


def transform_homogeneous(points, T):
    """用一个 4×4 矩阵批量变换每行一个的三维点。"""
    points_h = np.column_stack([points, np.ones(len(points))])
    transformed_h = (T @ points_h.T).T
    return transformed_h[:, :3]


def cube_edges(points):
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            # 角点生成顺序对应三位二进制；只翻转一位就是一条边。
            # 这样判断不依赖点变换后的坐标方向。
            if (i ^ j) in (1, 2, 4):
                yield points[[i, j]]


def draw_cube(ax, points, color, linestyle="-", label=None):
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               color=color, s=30, label=label)
    for edge in cube_edges(points):
        ax.plot(edge[:, 0], edge[:, 1], edge[:, 2],
                color=color, linestyle=linestyle, linewidth=1.6)


def set_axes(ax, title, points):
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = (lower + upper) / 2
    span = max(float(np.max(upper - lower)), 1.0) * 1.35
    ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
    ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
    ax.set_zlim(center[2] - span / 2, center[2] + span / 2)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X：右（米）")
    ax.set_ylabel("Y：下（米）")
    ax.set_zlabel("Z：前（米）")
    ax.set_title(title)
    ax.view_init(elev=20, azim=-65)


if __name__ == "__main__":
    world = cube_points()
    R_wc, t_wc = camera_pose()
    # 旧写法用于对照：旋转、平移分开计算。
    camera_separate = world_to_camera(world, R_wc, t_wc)

    # 新写法：把旋转和平移装进齐次矩阵。
    T_wc = make_T_wc(R_wc, t_wc)
    T_cw = np.linalg.inv(T_wc)
    camera = transform_homogeneous(world, T_cw)
    recovered = transform_homogeneous(camera, T_wc)
    error = np.linalg.norm(recovered - world, axis=1)
    representation_difference = np.linalg.norm(camera - camera_separate, axis=1)
    inverse_check = np.max(np.abs(T_wc @ T_cw - np.eye(4)))

    fig = plt.figure(figsize=(13, 5), constrained_layout=True)
    fig.suptitle("4×4 齐次变换：世界 → 相机 → 世界")
    ax_world = fig.add_subplot(131, projection="3d")
    ax_camera = fig.add_subplot(132, projection="3d")
    ax_recovered = fig.add_subplot(133, projection="3d")

    draw_cube(ax_world, world, "tab:orange", label="世界中的立方体")
    ax_world.scatter(*t_wc, marker="^", color="black", s=70, label="相机世界位置")
    set_axes(ax_world, "1  世界坐标", np.vstack([world, t_wc]))
    ax_world.legend(loc="upper left")

    draw_cube(ax_camera, camera, "tab:blue", label="相机坐标中的立方体")
    ax_camera.scatter([0], [0], [0], marker="^", color="black", s=70)
    set_axes(ax_camera, "2  转入相机坐标", np.vstack([camera, np.zeros(3)]))
    ax_camera.legend(loc="upper left")

    draw_cube(ax_recovered, world, "gray", linestyle="--", label="原世界位置")
    ax_recovered.scatter(recovered[:, 0], recovered[:, 1], recovered[:, 2],
                         marker="*", color="tab:purple", s=75, label="逆变换恢复")
    set_axes(ax_recovered, "3  逆变换放回世界", np.vstack([world, recovered]))
    ax_recovered.legend(loc="upper left")

    print("每个角点的往返误差（米）：")
    print(error)
    print(f"最大往返误差：{error.max():.3e} 米")
    print("\nT_wc（相机 → 世界）：")
    print(T_wc)
    print("\nT_cw（世界 → 相机）：")
    print(T_cw)
    print(f"\n4×4写法与分开计算的最大差异：{representation_difference.max():.3e} 米")
    print(f"T_wc @ T_cw 与单位矩阵的最大差异：{inverse_check:.3e}")
    fig.text(0.5, 0.015,
             f"4×4与分开计算的差异：{representation_difference.max():.1e} 米；"
             f"往返误差：{error.max():.1e} 米；紫星应与灰点重合",
             ha="center")
    plt.show()
