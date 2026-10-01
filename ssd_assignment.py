"""L05 核心练习：实现两个等长灰度窗口的 SSD（平方差之和）。"""
import numpy as np


def compute_ssd(left_patch, right_patch):
    """返回两个灰度窗口的平方差之和。"""
    if left_patch.shape != right_patch.shape:
        raise ValueError("两个窗口的 shape 必须相同")

    difference_arr = (left_patch - right_patch)**2
    difference = difference_arr.sum()
    return difference


if __name__ == "__main__":
    # 使用浮点数，避免 uint8 相减时出现回绕问题。
    left = np.array([10, 30, 50], dtype=float)
    candidate_good = np.array([12, 28, 52], dtype=float)
    candidate_bad = np.array([50, 10, 20], dtype=float)

    good_ssd = compute_ssd(left, candidate_good)
    bad_ssd = compute_ssd(left, candidate_bad)

    print("相似窗口 SSD：", good_ssd)
    print("不相似窗口 SSD：", bad_ssd)

    assert np.isclose(good_ssd, 12.0)
    assert np.isclose(bad_ssd, 2900.0)
    assert good_ssd < bad_ssd
    print("SSD 核心计算检查通过")
