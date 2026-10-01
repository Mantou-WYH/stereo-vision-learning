"""枚举三个像素的八种组合，直观核对SSD与变化惩罚；不是完整SGBM。"""
from itertools import product
import numpy as np
import matplotlib.pyplot as plt


SSD = np.array([[2, 8], [5, 4], [8, 1]])
DISPARITIES = np.array([6, 18])


def choose(penalty):
    candidates = []
    for choices in product(range(2), repeat=3):
        matching = sum(int(SSD[i, c]) for i, c in enumerate(choices))
        changes = sum(choices[i] != choices[i-1] for i in range(1, 3))
        total = matching + changes * penalty
        candidates.append((total, choices, matching, changes))
    # 枚举只用于这个极小例子的透明验算，大图不能这样穷举。
    return min(candidates, key=lambda item: item[0])


def main(save_path=None):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), layout='constrained')
    for ax, penalty in zip(axes, [0, 3, 8]):
        total, choices, matching, changes = choose(penalty)
        selected = DISPARITIES[list(choices)]
        assert total == matching + penalty * changes
        assert selected.tolist() == ([6, 18, 18] if penalty < 8 else [18, 18, 18])
        ax.plot(range(3), selected, color='#ed8731', linewidth=3, zorder=2)
        for i in range(3):
            for c, d in enumerate(DISPARITIES):
                picked = choices[i] == c
                ax.scatter(i, d, s=280, color='#ed8731' if picked else '#d4d9df', zorder=3)
                ax.text(i, d+1.5, f'SSD={SSD[i,c]}', ha='center', fontsize=11)
        ax.set_title(f'视差每变化一次，加 {penalty} 分', fontsize=12)
        ax.set_xticks([0, 1, 2], ['A', 'B', 'C'])
        ax.set_yticks([6, 18])
        ax.set_ylabel('候选视差（像素）')
        ax.set_xlim(-.45, 2.45)
        ax.set_ylim(2, 23)
        ax.grid(alpha=.2)
        ax.set_xlabel(f'选中：{selected.tolist()}\nSSD之和 {matching} + 惩罚 {changes * penalty} = 总分 {total}', fontsize=11)
        print(f'惩罚{penalty}：选择{selected.tolist()}，SSD之和{matching}，变化{changes}次，总分{total}')
    fig.suptitle('同一组SSD，橙色是选中的组合；只改变视差变化的惩罚', fontsize=14)
    fig.text(.5, .02, '教学假设分数，没有真实距离标签：总分最低不等于一定符合真实场景。', ha='center', fontsize=10)
    fig.get_layout_engine().set(rect=(0, .09, 1, .92))
    if save_path:
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    main()
