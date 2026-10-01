"""L09：模拟真实三维视角变化，串接图像跟踪、第一帧深度与PnP。"""
import argparse
import json
from pathlib import Path
import numpy as np
from two_view_mapping_lab import render, look_at, LOW, HIGH, WIDTH, HEIGHT, FX, FY, CX, CY

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'notes/synthetic_motion/tracked_pnp'


def textured_view(camera, rotation, textures, cv2):
    """纹理固定在箱体表面，每个视角独立求射线最近交点。"""
    view = render(camera, rotation)
    xyz = np.zeros((HEIGHT, WIDTH, 3))
    xyz[view['valid']] = view['world_points']
    image = view['picture'].copy()
    for face in range(6):
        mask = view['face'] == face
        if not mask.any():
            continue
        axes = [a for a in range(3) if a != face//2]
        tx = ((xyz[..., axes[0]]-LOW[axes[0]])/(HIGH[axes[0]]-LOW[axes[0]])*255).astype(np.float32)
        ty = ((xyz[..., axes[1]]-LOW[axes[1]])/(HIGH[axes[1]]-LOW[axes[1]])*255).astype(np.float32)
        values = cv2.remap(textures[face], tx, ty, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        image[mask] *= (.3+.7*values[mask, None])
    rgb = np.round(image*255).astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return view, rgb, gray


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save-only', action='store_true')
    args = parser.parse_args()
    import cv2
    rng = np.random.default_rng(20261001)
    textures = []
    for _ in range(6):
        texture = cv2.GaussianBlur(rng.random((256, 256)).astype(np.float32), (0, 0), 1.)
        texture = (texture-texture.min())/(texture.max()-texture.min())
        textures.append(texture)
    first_camera = np.array([1.4, 0., 0.])
    first_R_wc = look_at(first_camera, np.array([0., 0., 3.9]))
    # 这两项只用于生成观察与事后评估，不输入跟踪器或PnP。
    truth_camera_in_first = np.array([.12, .02, .05])
    angle = np.deg2rad(-1.)
    truth_R_wc_in_first = np.array([[np.cos(angle), 0., np.sin(angle)],
                                  [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]])
    second_camera = first_camera + first_R_wc @ truth_camera_in_first
    second_R_wc = first_R_wc @ truth_R_wc_in_first
    view0, rgb0, gray0 = textured_view(first_camera, first_R_wc, textures, cv2)
    view1, rgb1, gray1 = textured_view(second_camera, second_R_wc, textures, cv2)
    # 第一帧深度可用区域，避开外轮廓窗口；不使用第二帧真值筛选。
    mask = cv2.erode(view0['valid'].astype(np.uint8)*255, np.ones((5, 5), np.uint8))
    p0 = cv2.goodFeaturesToTrack(gray0, maxCorners=150, qualityLevel=.01, minDistance=6,
                                 mask=mask, blockSize=3, useHarrisDetector=False)
    if p0 is None:
        raise RuntimeError('没有检测到可用特征')
    p1, status, local_error = cv2.calcOpticalFlowPyrLK(
        gray0, gray1, p0, None, winSize=(15, 15), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, .01))
    if p1 is None:
        raise RuntimeError('跟踪未返回位置')
    uv0, uv1 = p0.reshape(-1, 2), p1.reshape(-1, 2)
    pixels = np.floor(uv0+.5).astype(int)
    z = view0['depth'][pixels[:, 1], pixels[:, 0]]
    usable = (status.ravel() == 1) & np.isfinite(z) & (z > 0) & np.all(np.isfinite(uv1), axis=1)
    usable &= (uv1[:, 0] >= 0) & (uv1[:, 0] < WIDTH) & (uv1[:, 1] >= 0) & (uv1[:, 1] < HEIGHT)
    indices = np.flatnonzero(usable)
    xyz = np.column_stack([(uv0[usable, 0]-CX)*z[usable]/FX,
                           (uv0[usable, 1]-CY)*z[usable]/FY, z[usable]])
    K = np.array([[FX, 0., CX], [0., FY, CY], [0., 0., 1.]])
    cv2.setRNGSeed(20261001)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        xyz, uv1[usable], K, np.zeros(5), iterationsCount=100,
        reprojectionError=2., confidence=.99, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inliers is None:
        raise RuntimeError('PnP未恢复位姿')
    accepted = inliers.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(xyz[accepted], uv1[usable][accepted], K, np.zeros(5), rvec, tvec)
    R_cw, _ = cv2.Rodrigues(rvec)
    R_wc = R_cw.T
    camera_in_first = (-R_wc @ tvec).ravel()
    predicted, _ = cv2.projectPoints(xyz, rvec, tvec, K, np.zeros(5))
    predicted = predicted.reshape(-1, 2)
    assert np.all(((xyz-camera_in_first) @ R_wc)[:, 2] > 0)
    reprojection_errors = np.linalg.norm(predicted-uv1[usable], axis=1)
    # 求解完成后才评价真实位姿、跟踪位置与几何可见性。
    xyz_true_second = (xyz-truth_camera_in_first) @ truth_R_wc_in_first
    truth_uv1 = np.column_stack([FX*xyz_true_second[:, 0]/xyz_true_second[:, 2]+CX,
                                FY*xyz_true_second[:, 1]/xyz_true_second[:, 2]+CY])
    tracking_errors = np.linalg.norm(uv1[usable]-truth_uv1, axis=1)
    rotation_error = np.rad2deg(np.arccos(np.clip((np.trace(R_wc.T @ truth_R_wc_in_first)-1)/2, -1, 1)))
    faces = view0['face'][pixels[usable, 1], pixels[usable, 0]]
    yaw = float(np.rad2deg(np.arctan2(R_wc[0, 2], R_wc[2, 2])))
    OUT.mkdir(parents=True, exist_ok=True)
    for number, rgb in enumerate([rgb0, rgb1]):
        from PIL import Image
        Image.fromarray(rgb).save(OUT/f'frame_{number}.png')
    np.savez_compressed(OUT/'inputs.npz', first_depth_m=view0['depth'], first_uv_px=uv0,
                        estimated_second_uv_px=uv1, tracking_status=status.ravel(),
                        usable_original_indices=indices, points_in_first_camera_m=xyz, K=K)
    np.savez_compressed(OUT/'estimated_pose.npz', camera_position_in_first_m=camera_in_first,
                        R_wc_in_first=R_wc, rvec_cw=rvec, tvec_cw=tvec,
                        accepted_input_rows=accepted, predicted_second_uv_px=predicted)
    report = dict(kind='Image tracking to PnP on two independently rendered 3D box views',
                  opencv_version=cv2.__version__, detected_points=len(uv0), usable_pairs=len(xyz),
                  ransac_supporting_pairs=len(accepted),
                  first_surface_counts={str(i):int((faces == i).sum()) for i in np.unique(faces)},
                  truth_camera_position_in_first_m=truth_camera_in_first.tolist(),
                  estimated_camera_position_in_first_m=camera_in_first.tolist(),
                  position_error_m=float(np.linalg.norm(camera_in_first-truth_camera_in_first)),
                  truth_yaw_in_first_deg=-1., estimated_yaw_in_first_deg=yaw,
                  full_rotation_error_deg=float(rotation_error),
                  mean_inlier_reprojection_error_px=float(reprojection_errors[accepted].mean()),
                  median_tracking_position_error_px=float(np.median(tracking_errors)),
                  max_tracking_position_error_px=float(tracking_errors.max()),
                  depth_source='Ideal first-frame forward depth from renderer, not SGBM or ground-truth input to tracking',
                  evidence='Teacher preparation and run; student run/independent implementation pending',
                  limits='Small synthetic motion, ideal known depth and fixed textured box; no real temporal sequence or stereo matching')
    (OUT/'checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    import matplotlib
    if args.save_only:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(12, 8.5))
    fig.suptitle('从图像对应，算出第二次拍摄的相机位置', fontsize=19, y=.96)
    fig.text(.5, .895, '同一有纹理箱体，两次独立三维渲染；第一帧深度由模拟器提供。',
             ha='center', fontsize=12)
    axes = [fig.add_axes([.065, .51, .38, .31]), fig.add_axes([.555, .51, .38, .31])]
    selected = accepted[np.linspace(0, len(accepted)-1, min(12, len(accepted)), dtype=int)]
    for i, (ax, rgb, title) in enumerate(zip(axes, [rgb0, rgb1], ['第一次拍摄', '第二次拍摄：位置与朝向都改变'])):
        ax.imshow(rgb)
        uv = [uv0[usable], uv1[usable]][i][selected]
        ax.scatter(uv[:, 0], uv[:, 1], s=32, facecolors='none', edgecolors='#00dc87', linewidths=1.3)
        ax.set(title=title, xlabel='u（像素）', ylabel='v（像素）')
    for row in selected:
        fig.add_artist(ConnectionPatch(xyA=uv0[usable][row], xyB=uv1[usable][row],
                       coordsA='data', coordsB='data', axesA=axes[0], axesB=axes[1],
                       color='#00ac72', alpha=.5, lw=.65))
    ax = fig.add_axes([.10, .14, .30, .24])
    ax.scatter(0, 0, c='#172330', s=60, label='第一相机')
    ax.scatter(*truth_camera_in_first[[0, 2]], facecolors='none', edgecolors='#e87623', s=170,
               linewidths=2, label='设定位置')
    ax.scatter(*camera_in_first[[0, 2]], marker='x', c='#00966a', s=80, linewidths=2, label='PnP估计')
    ax.plot([0, camera_in_first[0]], [0, camera_in_first[2]], color='#00966a', lw=1)
    ax.set(xlim=(-.03, .16), ylim=(-.025, .105), xlabel='X 向右（米）', ylabel='Z 向前（米）', title='俯视：以第一相机为原点')
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, loc='upper left')
    fig.text(.48, .38, f'检测 {len(uv0)} 点 → 可用对应 {len(xyz)} 对 → RANSAC支持 {len(accepted)} 对', fontsize=12, va='top')
    fig.text(.48, .33, '第二相机位置，单位米：\n'
             '设定：(0.120, 0.020, 0.050)\n'
             f'估计：({camera_in_first[0]:.3f}, {camera_in_first[1]:.3f}, {camera_in_first[2]:.3f})',
             fontsize=12, linespacing=1.6, va='top')
    fig.text(.48, .195, f'向左转：设定1.000°，估计{-yaw:.3f}°\n位置残差：{report["position_error_m"]*1000:.2f} 毫米',
             fontsize=12, linespacing=1.6, va='top')
    fig.text(.5, .035, '蓝/橙为箱面外观，绿线为展示的跟踪对应；位姿真值只用于生成和事后核对，未输入PnP。',
             ha='center', fontsize=10, color='#42516a')
    fig.savefig(OUT/'comparison.png', dpi=150)
    if args.save_only:
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    main()
