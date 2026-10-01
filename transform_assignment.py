"""L02 最终实现：学生填写批量齐次点变换的核心两行。"""
import numpy as np
from cube_transform_lab import cube_points, camera_pose, make_T_wc


def transform_points(points, T):
    """points 每行一个 (X,Y,Z)，T 是4×4变换；返回每行一个三维点。"""
    points_h = np.column_stack([points, np.ones(len(points))])

    points_h = np.column_stack([points, np.ones(len(points))])

    transformed_h = (T @ points_h.T).T
    return transformed_h[:, :3]


if __name__ == "__main__":
    points_world = cube_points()
    R_wc, t_wc = camera_pose()
    T_wc = make_T_wc(R_wc, t_wc)
    T_cw = np.linalg.inv(T_wc)

    points_camera = transform_points(points_world, T_cw)
    points_recovered = transform_points(points_camera, T_wc)
    errors = np.linalg.norm(points_recovered - points_world, axis=1)

    print("输入点集 shape：", points_world.shape)
    print("相机点集 shape：", points_camera.shape)
    print("恢复点集 shape：", points_recovered.shape)
    print(f"最大往返误差：{errors.max():.3e} 米")

    assert points_camera.shape == (8, 3)
    assert points_recovered.shape == (8, 3)
    assert errors.max() < 1e-9
    print("L02 批量变换检查通过")
