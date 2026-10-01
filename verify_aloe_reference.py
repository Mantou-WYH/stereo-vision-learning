"""教师核验：来源一致性与现有课堂参考读取，不修改数据或参数。"""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np

from real_sgbm_lab import load_pair, compute_disparity
from sgbm_reference_lab import load_reference
from sgbm_error_lab import evaluate


def main():
    folder = Path(__file__).resolve().parent / "data/opencv-aloe"
    files = {}
    for name in ("aloeL.jpg", "aloeR.jpg", "aloeGT.png"):
        local = (folder / name).read_bytes()
        url = f"https://raw.githubusercontent.com/opencv/opencv/4.12.0/samples/data/{name}"
        with urlopen(url, timeout=30) as response:
            official = response.read()
        assert local == official, f"官方文件不一致：{name}"
        decoded = cv2.imdecode(np.frombuffer(local, np.uint8), cv2.IMREAD_UNCHANGED)
        files[name] = dict(sha256=hashlib.sha256(local).hexdigest(),
                           shape=list(decoded.shape), official_equal=True)

    assert (folder / "aloeGT.png").read_bytes() == (folder / "aloeGT-official-2015.png").read_bytes()
    gt = cv2.imdecode(np.fromfile(folder / "aloeGT.png", np.uint8), cv2.IMREAD_UNCHANGED)
    left, right = load_pair()
    reference = load_reference(left.shape[:2])
    # 独立检查现有半尺寸最近邻协议：取偶数行列，视差乘0.5。
    expected = gt[::2, ::2].astype(np.float32) * 0.5
    expected[gt[::2, ::2] == 0] = np.nan
    assert np.array_equal(reference, expected, equal_nan=True)
    estimate = compute_disparity(left, right)
    _, _, _, _, stats = evaluate(estimate, reference)
    print(json.dumps(dict(files=files, first_official_release_equal=True,
                          half_size_reader_verified=True,
                          original_gt_unknown=int((gt == 0).sum()),
                          metrics=stats), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
