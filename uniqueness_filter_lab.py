"""L05：复用学生的 SSD 函数，观察唯一性阈值如何过滤匹配。"""
import sys

import matplotlib.pyplot as plt
import numpy as np

from ssd_assignment import compute_ssd


def main():
    # 三组独立的教学候选，不是从一条真实图像行切出的相邻窗口。
    left_patch = np.array([10, 30, 50], dtype=float)
    candidates = [
        np.array([[50, 10, 20], [20, 25, 45], [13, 27, 54], [12, 28, 52]], dtype=float),
        np.array([[14, 32, 50], [14, 32, 51], [20, 25, 45], [50, 10, 20]], dtype=float),
        np.array([[10, 30, 50], [20, 25, 45], [10, 30, 50], [50, 10, 20]], dtype=float),
    ]
    names = ["A：最优较突出", "B：候选接近", "C：完全重复"]
    gaps = []
    for name, right_candidates in zip(names, candidates):
        costs = []
        for right_patch in right_candidates:
                costs.append(float(compute_ssd(left_patch, right_patch)))

        sorted_costs = np.sort(costs)
        gap = sorted_costs[1] - sorted_costs[0]
        gaps.append(gap)
        print(f"{name}：SSD={costs}；最优与次优的差距={gap:g}")

    gaps = np.array(gaps)
    for threshold in [10, 30]:
        print(f"\n阈值={threshold}（要求差距严格大于阈值）")
        accepted_count = 0
        for name, gap in zip(names, gaps):
            # 本轮核心：只在最优候选领先足够多时接受匹配。
            if gap > threshold:
                accepted_count += 1
                print(f"  {name}：通过唯一性检查")
            else:
                print(f"  {name}：拒绝，深度未知")
        print(f"  接受数量：{accepted_count}/3")

    print("\n这是简化的绝对差距检查。阈值依赖灰度尺度、窗口大小和噪声。")
    print("通过检查不等于匹配正确；此实验没有测量正确率。")
    if "--check" in sys.argv:
        assert np.allclose(gaps, [22, 1, 0])
        assert np.count_nonzero(gaps > 10) == 1
        assert np.count_nonzero(gaps > 30) == 0
        print("教师数值检查通过")
        return

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.bar(names, gaps, width=0.45, color="#3575aa")
    for index, gap in enumerate(gaps):
        ax.text(index, gap + 0.7, f"差距 {gap:g}", ha="center")
    ax.axhline(10, color="#b36b00", linestyle="--", label="阈值10：仅A通过")
    ax.axhline(30, color="#95436c", linestyle=":", label="阈值30：全部拒绝")
    ax.set_ylim(0, 38)
    ax.set_ylabel("次优SSD − 最优SSD（灰度值平方）")
    ax.set_title("柱顶超过阈值线，才通过唯一性检查")
    ax.legend(loc="upper right")
    ax.set_xlabel("被拒绝的匹配：深度未知；通过检查：仍不保证正确")
    plt.show()


if __name__ == "__main__":
    main()
