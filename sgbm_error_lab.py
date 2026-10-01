"""固定估计结果，区分误差与缺失；非官方基准评估。"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from real_sgbm_lab import load_pair, compute_disparity
from sgbm_reference_lab import load_reference


def evaluate(estimate, reference):
    known = np.isfinite(reference)
    answered = np.isfinite(estimate)
    common = known & answered
    missing = known & ~answered
    if not known.any() or not common.any():
        raise ValueError('没有足够的可评估位置')
    error = np.full(reference.shape, np.nan)
    error[common] = np.abs(estimate[common] - reference[common])
    stats = dict(reference_count=int(known.sum()), answered_count=int(common.sum()),
                 missing_count=int(missing.sum()), coverage=float(common.sum()/known.sum()),
                 mae=float(error[common].mean()),
                 bad2=float(np.mean(error[common] > 2)))
    assert stats['answered_count'] + stats['missing_count'] == stats['reference_count']
    return error, known, common, missing, stats


def main(save_path=None):
    left, right = load_pair()
    estimate = compute_disparity(left, right)
    reference = load_reference(estimate.shape)
    error, known, common, missing, stats = evaluate(estimate, reference)
    print('范围：整张缩小后的左图中，参考答案有效的位置；未排除左侧搜索边带。')
    print('参考图没有答案的位置不评价。参考采用最近邻降采样，非官方榜单协议。')
    print(f"有参考答案：{stats['reference_count']}；程序作答：{stats['answered_count']}；未作答：{stats['missing_count']}")
    print(f"作答比例：{stats['coverage']:.2%}（不是正确率）")
    print(f"双方都有数值的位置，平均绝对误差：{stats['mae']:.3f} 像素")
    print(f"双方都有数值的位置，误差超过2像素的比例：{stats['bad2']:.2%}")
    print('误差单位：缩小后图像像素。未作答位置不当成零误差。参数未调整。')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 2, figsize=(12, 7.5), layout='constrained')
    # 状态0/1/2不是视差，只表示三类位置。
    status = np.zeros(reference.shape, dtype=np.uint8)
    status[common] = 1
    status[missing] = 2
    colors = ['#b8b8b8', '#548bab', '#ed9d40']
    axes[0].imshow(status, cmap=ListedColormap(colors), vmin=0, vmax=2, interpolation='nearest')
    axes[0].set_title(f"有没有作答？有参考答案处的作答比例 {stats['coverage']:.1%}", fontsize=11)
    axes[0].legend(handles=[Patch(facecolor=c, label=t) for c,t in zip(colors,
                      ['灰：参考未知，不评价', '蓝：程序给了答案（不保证对）', '橙：参考已知，程序未作答'])],
                   loc='upper center', bbox_to_anchor=(.5,-.16), fontsize=10)
    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad('#b8b8b8')
    plotted = axes[1].imshow(error, cmap=cmap, vmin=0, vmax=10, interpolation='nearest')
    axes[1].set_title(f"给出的答案错多少？平均 {stats['mae']:.2f} 像素", fontsize=11)
    fig.colorbar(plotted, ax=axes[1], shrink=.75, extend='max',
                 label='绝对误差（像素；超过10以最亮色显示）')
    axes[1].set_xlabel('横向位置 u\n灰色：至少一方无数值，未计算误差')
    axes[0].set_xlabel('横向位置 u')
    for ax in axes:
        ax.set_ylabel('纵向位置 v')
    fig.suptitle('仍是同一盆芦荟：左图检查缺失，右图检查数值错误', fontsize=14)
    fig.get_layout_engine().set(h_pad=0.15, w_pad=0.12)
    if save_path:
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
    else:
        plt.show()
    return stats


if __name__ == '__main__':
    main()
