"""L08：两个模拟视角看到不同表面，用已知位姿拼接。

合成箱体、针孔相机、理想深度与位姿均由教师设定。
按每条射线的最近交点生成可见深度；不复用同一组点伪造第二观测。
无实拍、双目匹配、位姿估计、噪声或TSDF融合；只研究可见性与坐标统一。
"""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "notes/synthetic_mapping/two_views"
WIDTH, HEIGHT = 240, 180
FX = FY = 190.
CX, CY = (WIDTH-1)/2, (HEIGHT-1)/2
LOW = np.array([-.8, -.65, 3.1])
HIGH = np.array([.8, .65, 4.7])
FACE_NAMES = ["left", "right", "top", "bottom", "front", "back"]
PALETTE = np.array([[.6, .4, .8], [1., .53, .12], [.4, .8, .5],
                    [.8, .7, .4], [.08, .57, .82], [.7, .4, .5]])


def look_at(camera, target):
    forward = target-camera
    forward /= np.linalg.norm(forward)
    right = np.cross(np.array([0., 1., 0.]), forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.column_stack([right, down, forward])


def render(camera, rotation):
    v, u = np.indices((HEIGHT, WIDTH))
    camera_rays = np.stack([(u-CX)/FX, (v-CY)/FY, np.ones_like(u)], axis=-1)
    rays = camera_rays @ rotation.T
    lower = np.full_like(rays, -np.inf)
    upper = np.full_like(rays, np.inf)
    for axis in range(3):
        moving = np.abs(rays[..., axis]) > 1e-12
        a = np.zeros((HEIGHT, WIDTH))
        b = np.zeros_like(a)
        np.divide(LOW[axis]-camera[axis], rays[..., axis], out=a, where=moving)
        np.divide(HIGH[axis]-camera[axis], rays[..., axis], out=b, where=moving)
        lower[..., axis][moving] = np.minimum(a, b)[moving]
        upper[..., axis][moving] = np.maximum(a, b)[moving]
        if not LOW[axis] <= camera[axis] <= HIGH[axis]:
            lower[..., axis][~moving] = np.inf
            upper[..., axis][~moving] = -np.inf
    entering, leaving = lower.max(axis=-1), upper.min(axis=-1)
    valid = np.isfinite(entering) & (entering > 0) & (leaving >= entering)
    depth = np.full((HEIGHT, WIDTH), np.nan)
    depth[valid] = entering[valid]  # 相机射线Z分量为1，因此参数就是前向Z。
    face = np.full((HEIGHT, WIDTH), -1, dtype=np.int8)
    axis = lower.argmax(axis=-1)
    entry_direction = np.take_along_axis(rays, axis[..., None], axis=-1)[..., 0]
    face[valid] = (2*axis+(entry_direction < 0))[valid]
    picture = np.full((HEIGHT, WIDTH, 3), .91)
    picture[valid] = PALETTE[face[valid]]
    # 前面学过的反投影：只处理本帧实际可见的像素。
    z = depth[valid]
    camera_points = np.column_stack([(u[valid]-CX)*z/FX, (v[valid]-CY)*z/FY, z])
    world_points = camera_points @ rotation.T + camera
    # 独立沿世界射线求交的坐标，用于核对反投影/变换链。
    ray_hits = camera + entering[valid, None]*rays[valid]
    assert np.allclose(world_points, ray_hits, atol=1e-12, rtol=0)
    assert np.all(world_points >= LOW-1e-12) and np.all(world_points <= HIGH+1e-12)
    return dict(depth=depth, valid=valid, face=face, picture=picture,
                camera_points=camera_points, world_points=world_points,
                colors=picture[valid], camera=camera, R_wc=rotation)


def build_views():
    camera1 = np.array([0., 0., 0.])
    camera2 = np.array([2.5, 0., 1.])
    target = np.array([0., 0., 3.9])
    first = render(camera1, np.eye(3))
    second = render(camera2, look_at(camera2, target))
    assert set(first["face"][first["valid"]]) == {4}
    assert set(second["face"][second["valid"]]) == {1, 4}
    return first, second


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-only", action="store_true")
    args = parser.parse_args()
    first, second = build_views()
    OUT.mkdir(parents=True, exist_ok=True)
    report = dict(kind="synthetic pinhole observations with ideal depth and known poses",
                  intrinsics=dict(fx=FX, fy=FY, cx=CX, cy=CY),
                  box_min_m=LOW.tolist(), box_max_m=HIGH.tolist(),
                  note="Concatenation only; front observed twice; no fusion or real-scene accuracy claim.", views=[])
    for number, data in enumerate([first, second], start=1):
        np.savez_compressed(OUT / f"view_{number}.npz", **data)
        counts = {FACE_NAMES[i]: int((data["face"] == i).sum()) for i in range(6)
                  if (data["face"] == i).any()}
        report["views"].append(dict(camera_world_m=data["camera"].tolist(),
                                   R_wc=data["R_wc"].tolist(), visible_surface_samples=counts))
        print(f"第{number}帧：{counts}")
    merged = np.concatenate([first["world_points"], second["world_points"]])
    np.savez_compressed(OUT / "combined.npz", xyz_m=merged,
                        rgb=np.concatenate([first["colors"], second["colors"]]))
    (OUT / "checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    import matplotlib
    if args.save_only:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("第二个视角，给地图带来了什么？", fontsize=22, y=.965)
    fig.text(.5, .915, "合成场景：蓝色正面、橙色右侧面。颜色表示箱子的表面，单位为米。",
             ha="center", fontsize=13, color="#42516a")
    for left, data, title in [(.08, first, "第1帧：站在正前方，只看见蓝色正面"),
                              (.57, second, "第2帧：移到右侧，看见正面和右侧面")]:
        ax = fig.add_axes([left, .62, .34, .24])
        ax.imshow(data["picture"], interpolation="nearest")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("图像 u（像素）")
        ax.set_ylabel("图像 v（像素）")
    cloud_axes = [fig.add_axes([left, .15, .40, .36], projection="3d") for left in [.065, .55]]
    for ax, datasets, title in [(cloud_axes[0], [first], "地图中只有第1帧：一片正面"),
                                 (cloud_axes[1], [first, second], "加入第2帧：新增橙色侧面")]:
        for data in datasets:
            p, colors = data["world_points"][::2], data["colors"][::2]
            ax.scatter(p[:, 0], p[:, 2], p[:, 1], c=colors, s=3, linewidths=0, depthshade=False)
        ax.set(xlabel="地图 X（米）", ylabel="地图 Z（米）", zlabel="Y 向下（米）",
               xlim=(-1, 1), ylim=(2.9, 4.9), zlim=(.8, -.8), title=title)
        ax.set_box_aspect((2, 2, 1.6))
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([3, 4, 4.8])
        ax.set_zticks([-.6, 0, .6])
        ax.view_init(elev=22, azim=-53)
    def reset(_):
        for ax in cloud_axes:
            ax.view_init(elev=22, azim=-53)
        fig.canvas.draw_idle()
    reset_button = Button(fig.add_axes([.42, .075, .16, .045]), "恢复观察角度")
    reset_button.on_clicked(reset)
    fig.text(.5, .028, "灰色像素未观测到箱体；未看见的背面保持空缺。深度和位姿由模拟器给定；本次只拼接、不去重融合。",
             ha="center", fontsize=10, color="#42516a")
    fig.savefig(OUT / "comparison.png", dpi=130)
    if args.save_only:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
