"""L09：沿用单错配输入，对比普通PnP与RANSAC-PnP。"""
import argparse
import json
from pathlib import Path
import numpy as np
from two_view_mapping_lab import build_views

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'notes/synthetic_motion/pnp_wrong_match'
OUT = ROOT/'notes/synthetic_motion/pnp_ransac'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save-only', action='store_true')
    args = parser.parse_args()
    import cv2
    with np.load(SOURCE/'input_comparison.npz') as data:
        points = data['world_points_m'].copy()
        observed = data['wrong_observed_uv_px'].copy()
        K = data['K'].copy()
        # 教师已知身份只用于事后评价，不交给求解器。
        correct_for_evaluation = data['known_correct_subset_for_evaluation'].copy()
    with np.load(SOURCE/'one_wrong_match.npz') as data:
        ordinary_predicted = data['predicted_uv_px'].copy()
        ordinary_camera = data['camera_position_m'].copy()
        assert np.array_equal(data['observed_uv_px'], observed)
        assert np.array_equal(data['world_points_m'], points)
        assert np.array_equal(data['K'], K)
    cv2.setRNGSeed(20261001)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        points, observed, K, np.zeros(5), iterationsCount=100,
        reprojectionError=2., confidence=.99, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inliers is None:
        raise RuntimeError('RANSAC未找到可用位姿')
    accepted = inliers.ravel()
    # 仅使用算法返回的支持对应细化位姿。
    rvec, tvec = cv2.solvePnPRefineLM(
        points[accepted], observed[accepted], K, np.zeros(5), rvec, tvec)
    R_cw, _ = cv2.Rodrigues(rvec)
    R_wc = R_cw.T
    camera = (-R_wc @ tvec).ravel()
    predicted, _ = cv2.projectPoints(points, rvec, tvec, K, np.zeros(5))
    predicted = predicted.reshape(-1, 2)
    assert np.all(((points-camera) @ R_wc)[:, 2] > 0)
    view = build_views()[1]  # 求解结束后取人为真值评价与绘图。
    errors = [np.linalg.norm(p-observed, axis=1) for p in [ordinary_predicted, predicted]]
    means = [float(e[correct_for_evaluation].mean()) for e in errors]
    excluded = np.setdiff1d(np.arange(len(points)), accepted)
    OUT.mkdir(parents=True, exist_ok=True)
    report = dict(kind='Teacher-run RANSAC-PnP on unchanged saved wrong-match input',
                  opencv_version=cv2.__version__, seed=20261001,
                  parameters=dict(iterationsCount=100, reprojectionError_px=2., confidence=.99,
                                  flags='SOLVEPNP_ITERATIVE', refinement='solvePnPRefineLM'),
                  accepted_zero_based_indices=accepted.tolist(), excluded_zero_based_indices=excluded.tolist(),
                  mean_same_correct_11_reprojection_error_px=dict(ordinary=means[0], ransac=means[1]),
                  camera_position_m=camera.tolist(),
                  camera_position_error_m=dict(ordinary=float(np.linalg.norm(ordinary_camera-view['camera'])),
                                               ransac=float(np.linalg.norm(camera-view['camera']))),
                  all_12_reprojection_error_px=errors[1].tolist(),
                  evidence='Teacher validation only; no student execution or independent implementation.',
                  limits='One planar synthetic configuration; rejected identities are not guaranteed in general.')
    (OUT/'checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez_compressed(OUT/'result.npz', world_points_m=points, observed_uv_px=observed, K=K,
                        predicted_uv_px=predicted, inlier_indices=accepted,
                        camera_position_m=camera, R_wc=R_wc, rvec_cw=rvec, tvec_cw=tvec)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    import matplotlib
    if args.save_only:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    fig.subplots_adjust(left=.07, right=.96, top=.76, bottom=.28, wspace=.25)
    fig.suptitle('同一条错配：先寻找共同支持，再求位姿', fontsize=19, y=.95)
    fig.text(.5, .87, '同一箱面、同一12对输入、同一标定；求解器没有收到错误点的身份。',
             ha='center', fontsize=11)
    for i, (ax, pp, title) in enumerate(zip(axes, [ordinary_predicted, predicted],
            ['普通PnP：12对一起拟合', 'RANSAC-PnP：11对支持最终拟合'])):
        ax.imshow(view['picture'], interpolation='nearest', alpha=.7)
        ax.scatter(observed[:, 0], observed[:, 1], c='#162435', s=20, zorder=3, label='输入图像位置')
        ax.scatter(pp[:, 0], pp[:, 1], s=90, facecolors='none', edgecolors='#009774',
                   linewidths=1.7, label='求解位姿的投影', zorder=4)
        for j, (o, pred) in enumerate(zip(observed, pp)):
            if i == 1 and j in excluded:
                continue
            ax.plot([o[0], pred[0]], [o[1], pred[1]], color='#009774', lw=1)
        ax.annotate('A', pp[5], xytext=(-14, -12), textcoords='offset points', fontsize=12)
        ax.annotate('B', observed[10], xytext=(8, 6), textcoords='offset points', fontsize=12)
        ax.scatter(*observed[5], s=180, facecolors='none', edgecolors='#dc6c2a', linewidths=2, zorder=5)
        if i == 1:
            ax.annotate('A被错配到B的输入\n未参加最后拟合', observed[5], xytext=(128, 140),
                        fontsize=9, color='#ad4d19',
                        arrowprops=dict(arrowstyle='->', color='#dc6c2a'))
        ax.set(title=title, xlabel='u（像素，向右）', ylabel='v（像素，向下）')
        ax.legend(loc='lower right', fontsize=9)
        fig.text(.285 if i == 0 else .76, .19,
                 f'同一11个正确对应的平均偏差：{means[i]:.2f} 像素', ha='center', fontsize=12)
    fig.text(.5, .11, '黑点是输入，绿圈是估计投影；橙圈位置同时收到正确的B和错误的A两条对应。',
             ha='center', fontsize=10)
    fig.text(.5, .035, '教师模拟验证：图片与地图均为人为场景；输入没有删除，RANSAC输出了不参与拟合的对应索引。',
             ha='center', fontsize=10, color='#42516a')
    fig.savefig(OUT/'comparison.png', dpi=150)
    if args.save_only:
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    main()
