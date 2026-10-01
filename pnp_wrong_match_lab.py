"""L09：12个模拟对应中只错配1个，观察普通PnP对其他点的影响。"""
import argparse
import json
from pathlib import Path
import numpy as np
from pnp_pose_lab import estimate_pose
from two_view_mapping_lab import build_views, FX, FY, CX, CY

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'notes/synthetic_motion/pnp_wrong_match'
A_INDEX, B_INDEX = 5, 10


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save-only', action='store_true')
    args = parser.parse_args()
    view = build_views()[1]
    points = np.array([[x, y, 3.1] for y in [-.35, 0., .35]
                       for x in [-.5, -.15, .2, .55]])
    p = (points-view['camera']) @ view['R_wc']
    ideal = np.column_stack([FX*p[:, 0]/p[:, 2]+CX, FY*p[:, 1]/p[:, 2]+CY])
    pixels = np.floor(ideal+.5).astype(int)
    assert np.all(view['face'][pixels[:, 1], pixels[:, 0]] == 4)
    K = np.array([[FX, 0., CX], [0., FY, CY], [0., 0., 1.]])
    rng = np.random.default_rng(20261001)
    correct_observed = ideal + rng.normal(0., .2, ideal.shape)
    wrong_observed = correct_observed.copy()
    wrong_observed[A_INDEX] = correct_observed[B_INDEX]
    correct_subset = np.arange(len(points)) != A_INDEX
    assert np.array_equal(wrong_observed[correct_subset], correct_observed[correct_subset])
    OUT.mkdir(parents=True, exist_ok=True)
    predictions, metrics = [], []
    for name, observed in [('correct', correct_observed), ('one_wrong_match', wrong_observed)]:
        C, R, rvec, tvec, predicted = estimate_pose(points, observed, K)
        assert np.all(((points-C) @ R)[:, 2] > 0)
        predictions.append(predicted)
        # 已知错误身份只用于事后评价，没有提供给普通PnP求解器。
        good_error = np.linalg.norm(predicted[correct_subset]-correct_observed[correct_subset], axis=1)
        all_error = np.linalg.norm(predicted-observed, axis=1)
        metrics.append(dict(condition=name, camera_position_m=C.tolist(),
                            camera_position_error_m=float(np.linalg.norm(C-view['camera'])),
                            mean_other_11_reprojection_error_px=float(good_error.mean()),
                            all_12_reprojection_error_px=all_error.tolist()))
        np.savez_compressed(OUT/f'{name}.npz', world_points_m=points, K=K,
                            observed_uv_px=observed, predicted_uv_px=predicted,
                            camera_position_m=C, R_wc=R, rvec_cw=rvec, tvec_cw=tvec)
    assert metrics[1]['mean_other_11_reprojection_error_px'] > metrics[0]['mean_other_11_reprojection_error_px']
    report = dict(kind='Teacher-run ordinary PnP one-correspondence identity error.',
                  points=12, seed=20261001, pixel_noise_std=.2,
                  change='Only A 3D point is paired with B observed 2D location; all other inputs identical.',
                  A_zero_based_index=A_INDEX, B_zero_based_index=B_INDEX, metrics=metrics,
                  evidence='Teacher preparation; student understanding and execution pending.',
                  limits='One planar synthetic configuration; no RANSAC, feature tracking or general accuracy claim.')
    (OUT/'checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez_compressed(OUT/'input_comparison.npz', world_points_m=points,
                        correct_observed_uv_px=correct_observed, wrong_observed_uv_px=wrong_observed,
                        known_correct_subset_for_evaluation=correct_subset, K=K)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    import matplotlib
    if args.save_only:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    fig.subplots_adjust(left=.07, right=.96, top=.76, bottom=.28, wspace=.25)
    fig.suptitle('一个点身份配错，也会牵动其他正确点的位姿拟合', fontsize=19, y=.95)
    fig.text(.5, .87, '同一箱面选12个标记；只把地图A对应到照片B的位置，其余11对完全不变。',
             ha='center', fontsize=11)
    for i, (ax, observed, predicted, title) in enumerate(zip(axes,
            [correct_observed, wrong_observed], predictions,
            ['12个对应身份全部正确', '仅A的对应身份错误'])):
        ax.imshow(view['picture'], interpolation='nearest', alpha=.7)
        ax.scatter(observed[:, 0], observed[:, 1], c='#162435', s=20, zorder=3,
                   label='输入图像位置')
        ax.scatter(predicted[:, 0], predicted[:, 1], s=90, facecolors='none',
                   edgecolors='#009774', linewidths=1.7, label='普通PnP的投影', zorder=4)
        for o, pred in zip(observed, predicted):
            ax.plot([o[0], pred[0]], [o[1], pred[1]], color='#009774', lw=1)
        for label, index in [('A', A_INDEX), ('B', B_INDEX)]:
            ax.annotate(label, correct_observed[index], xytext=(-12, -10),
                        textcoords='offset points', color='#182334', fontsize=12)
        if i:
            a, b = correct_observed[A_INDEX], wrong_observed[A_INDEX]
            ax.annotate('', xy=b, xytext=a,
                        arrowprops=dict(arrowstyle='->', color='#dc6c2a', lw=2))
            ax.scatter(*b, s=160, facecolors='none', edgecolors='#dc6c2a', linewidths=2, zorder=5)
        ax.set(title=title, xlabel='u（像素）', ylabel='v（像素）')
        ax.legend(loc='lower right', fontsize=9)
        fig.text(.285 if i == 0 else .76, .19,
                 f"其余11对的平均偏差：{metrics[i]['mean_other_11_reprojection_error_px']:.2f} 像素",
                 ha='center', fontsize=13)
    fig.text(.5, .11, '黑点是输入位置，绿圈是求解投影；右图橙色箭头表示人为把A错认成B。',
             ha='center', fontsize=11)
    fig.text(.5, .035, '地图点、相机标定和求解方法不变；错误身份未告知求解器，教师只在事后评价中知道它。',
             ha='center', fontsize=10, color='#42516a')
    fig.savefig(OUT/'comparison.png', dpi=150)
    if args.save_only:
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    main()
