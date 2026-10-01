"""L05 核心练习：遍历多个右窗口，选择 SSD 最小的匹配。"""
import numpy as np

from ssd_assignment import compute_ssd


left_patch = np.array([10, 30, 50], dtype=float)
u_left = 105

# 四个右图候选中心，以及各自对应的灰度窗口。
candidate_u_right = np.array([90, 89, 88, 87])
right_candidates = np.array([
    [50, 10, 20],
    [20, 25, 45],
    [13, 27, 54],
    [12, 28, 52],
], dtype=float)

costs = []

for right_patch in right_candidates:
    ssd = compute_ssd(right_patch,left_patch)
    costs.append(ssd)
    # TODO 1：调用 compute_ssd，计算 left_patch 与 right_patch 的 SSD。
    # TODO 2：用 costs.append(...) 把这次的 SSD 保存进列表。 

costs = np.array(costs)
best_index = np.argmin(costs)
best_u_right = candidate_u_right[best_index]
disparity = u_left - best_u_right

print("所有候选的 SSD：", costs)
print("最佳候选索引：", best_index)
print("最佳右图位置：", best_u_right)
print("估计视差：", disparity)

assert np.allclose(costs, [2900.0, 150.0, 34.0, 12.0])
assert best_u_right == 87
assert disparity == 18
print("多候选搜索检查通过")
