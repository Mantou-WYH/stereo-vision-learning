"""L09：用位置变化已知的测试图，观察Lucas–Kanade跟踪输出。"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'notes/feature_tracking/wood_lk_translation'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save-only', action='store_true')
    args = parser.parse_args()
    import cv2
    first = np.asarray(Image.open(ROOT/'data/eth3d/delivery_area_1s/im0.png').convert('L'))
    uv0 = np.load(ROOT/'notes/feature_tracking/wood_detection/detected_uv_px.npy').astype(np.float32)
    shift = np.array([-8., 3.], dtype=np.float32)
    warp = np.array([[1., 0., shift[0]], [0., 1., shift[1]]], dtype=np.float32)
    second = cv2.warpAffine(first, warp, (first.shape[1], first.shape[0]),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # 不提供已知位移、不提供初始猜测；算法只接收两图和第一图特征位置。
    next_points, status, patch_error = cv2.calcOpticalFlowPyrLK(
        first, second, uv0.reshape(-1, 1, 2), None, winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
    if next_points is None:
        raise RuntimeError('未返回跟踪位置')
    uv1 = next_points.reshape(-1, 2)
    h, w = first.shape
    usable = (status.ravel() == 1) & np.all(np.isfinite(uv1), axis=1)
    usable &= (uv1[:, 0] >= 0) & (uv1[:, 0] < w) & (uv1[:, 1] >= 0) & (uv1[:, 1] < h)
    # 已知变化只在求解结束后评价；status=1本身不证明身份正确。
    expected = uv0 + shift
    errors = np.linalg.norm(uv1-expected, axis=1)
    example = 50
    assert usable[example]
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(second).save(OUT/'generated_second.png')
    np.savez_compressed(OUT/'correspondences.npz', first_uv_px=uv0, estimated_second_uv_px=uv1,
                        status=status.ravel(), usable=usable, patch_error=patch_error.ravel(),
                        expected_second_uv_px_for_evaluation=expected, position_error_px=errors)
    report = dict(method='Pyramidal Lucas-Kanade / cv2.calcOpticalFlowPyrLK',
                  opencv_version=cv2.__version__, input_points=len(uv0), returned_status_1=int(status.sum()),
                  usable_count=int(usable.sum()), generated_shift_uv_px=shift.tolist(),
                  parameters=dict(winSize=[21, 21], maxLevel=3, max_iterations=30, epsilon=.01),
                  median_position_error_px=float(np.median(errors[usable])),
                  max_position_error_px=float(errors[usable].max()),
                  example=dict(zero_based_index=example, first_uv_px=uv0[example].tolist(),
                               estimated_second_uv_px=uv1[example].tolist()),
                  evidence='Teacher preparation and execution; student understanding/run pending',
                  limits='Generated 2D translation test; no actual temporal sequence, camera motion or PnP estimate')
    (OUT/'checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    import matplotlib
    if args.save_only:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.subplots_adjust(left=.07, right=.96, top=.83, bottom=.15, hspace=.55, wspace=.22)
    fig.suptitle('从已选特征点，到另一张图中的对应位置', fontsize=19, y=.96)
    fig.text(.5, .90, '左：原木箱照片；右：从原图生成的测试图。位置变化已知，未交给跟踪器。',
             ha='center', fontsize=11)
    colors = ['#25d9c7', '#ff8b3d']
    photos, points = [first, second], [uv0, uv1]
    p = uv0[example]
    for i in range(2):
        axes[0, i].imshow(photos[i], cmap='gray', vmin=0, vmax=255)
        axes[0, i].scatter(points[i][usable, 0], points[i][usable, 1], s=16, facecolors='none',
                           edgecolors=colors[i], linewidths=.7)
        axes[0, i].scatter(*points[i][example], s=95, facecolors='none', edgecolors=colors[i], linewidths=2)
        axes[0, i].annotate('P', points[i][example], xytext=(7, -7), textcoords='offset points',
                            color=colors[i], fontsize=12, weight='bold')
        axes[0, i].set(title=['原图中的100个特征位置', '跟踪器返回的100个位置'][i],
                       xlabel='u（像素）', ylabel='v（像素）')
        ax = axes[1, i]
        ax.imshow(photos[i], cmap='gray', vmin=0, vmax=255)
        ax.set_xlim(p[0]-25, p[0]+25)
        ax.set_ylim(p[1]+25, p[1]-25)
        ax.scatter(*points[i][example], s=110, facecolors='none', edgecolors=colors[i], linewidths=2)
        if i:
            ax.scatter(*p, s=60, marker='+', color='#25d9c7', linewidths=1.5)
            ax.annotate('', xy=uv1[example], xytext=p,
                        arrowprops=dict(arrowstyle='->', color='#ff8b3d', lw=1.7))
        ax.set(title=['放大P附近：原位置', '相同坐标范围：P的新位置'][i],
               xlabel='u（原图像素刻度）', ylabel='v（原图像素刻度）')
    fig.text(.5, .075,
             f'P：({p[0]:.0f}, {p[1]:.0f}) → ({uv1[example, 0]:.3f}, {uv1[example, 1]:.3f}) 像素',
             ha='center', fontsize=13)
    fig.text(.5, .025, '青色：原图位置；橙色：跟踪估计。测试图整体向左8、向下3像素，未模拟真实三维相机运动。',
             ha='center', fontsize=10, color='#42516a')
    fig.savefig(OUT/'comparison.png', dpi=150)
    if args.save_only:
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    main()
