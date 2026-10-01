"""运行未修改的OpenCV官方示例，记录原始视差并与课堂设置比较。

适配仅限中文路径读取、无窗口运行和跳过猜测标定的PLY写盘。
官方main中的pyrDown、匹配器创建、compute和Q重投影实际执行。
"""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from real_sgbm_lab import load_pair, compute_disparity
from sgbm_reference_lab import load_reference
from sgbm_error_lab import evaluate

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "third_party/opencv_4_12_0/stereo_match.py"
OUT = ROOT / "notes/official_stereo_comparison"
SOURCE_HASH = "0f2e5c4f27d4007e4fd959fe0b5d7a78a6296a079197dd213119b07b3532307a"
BASE_PARAMS = dict(minDisparity=0, numDisparities=128, blockSize=5,
                   P1=200, P2=800, disp12MaxDiff=1, preFilterCap=31,
                   uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
                   mode=cv2.STEREO_SGBM_MODE_SGBM)


def decoded(raw, minimum):
    result = raw.astype(np.float32) / 16.0
    result[raw <= (minimum - 1) * 16] = np.nan
    return result


class RecordingMatcher:
    def __init__(self, record, kwargs):
        self.record = record
        self.matcher = cv2.StereoSGBM_create(**kwargs)
        self.record["params"] = {name: int(getattr(self.matcher, "get" + name[0].upper() + name[1:])())
                                 for name in BASE_PARAMS}

    def compute(self, left, right):
        self.record["inputs"] = (left.copy(), right.copy())
        result = self.matcher.compute(left, right)
        self.record["raw"] = result.copy()
        return result


class SampleCvAdapter:
    def __init__(self, record):
        self.record = record
        self.samples = SimpleNamespace(findFile=lambda name: str(ROOT / "data/opencv-aloe" / name))

    def __getattr__(self, name):
        return getattr(cv2, name)

    def imread(self, path):
        return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)

    def StereoSGBM_create(self, **kwargs):
        return RecordingMatcher(self.record, kwargs)

    def imshow(self, name, data):
        # 保存原示例显示数值用于回归检查，不弹出阻塞窗口。
        self.record.setdefault("displayed", {})[name] = data.copy()

    def waitKey(self, *args):
        return -1


def run_official_main():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_HASH
    spec = importlib.util.spec_from_file_location("official_stereo_match", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    record = {}
    module.cv = SampleCvAdapter(record)

    def omit_ply(name, vertices, colors):
        record["ply_skipped"] = dict(name=name, vertex_count=len(vertices),
                                     reason="Official Q uses guessed focal length; not metric calibration")

    module.write_ply = omit_ply
    module.main()  # 真正执行官方原始main，不只手抄其参数。
    minimum = record["params"]["minDisparity"]
    count = record["params"]["numDisparities"]
    np.testing.assert_array_equal(record["displayed"]["disparity"],
                                  (record["raw"].astype(np.float32) / 16 - minimum) / count)
    return record, decoded(record["raw"], minimum)


def calculate(images, params, color=False):
    pair = images if color else [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in images]
    raw = cv2.StereoSGBM_create(**params).compute(*pair)
    return decoded(raw, params["minDisparity"])


def stats_on(disp, truth, mask=None):
    ref = truth.copy()
    if mask is not None:
        ref[~mask] = np.nan
    *_, stats = evaluate(disp, ref)
    valid = np.isfinite(ref) & np.isfinite(disp)
    correct = valid & (np.abs(disp - ref) <= 2)
    stats["within_2px_fraction_of_all_reference"] = float(correct.sum() / np.isfinite(ref).sum())
    return stats


def plot_comparison(truth, classroom, official, results, left):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#b8b8b8")
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), layout="constrained")
    axes[0].imshow(cv2.cvtColor(left, cv2.COLOR_BGR2RGB))
    axes[0].set_title("同一张左照片：位置参照")
    for ax, data, title in zip(axes[1:], [truth, classroom, official],
                               ["已发布的标准答案", "原课堂设置", "OpenCV官方示例原设置"]):
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=128, interpolation="nearest")
        ax.set_title(title)
    axes[1].set_xlabel("灰色：标准答案未知")
    for ax, key in zip(axes[2:], ["classroom", "official_original"]):
        stats = results[key]
        ax.set_xlabel(f"作答 {stats['coverage']:.1%}；作答处平均误差 {stats['mae']:.2f} px\n灰色：程序没有答案")
    for ax in axes:
        ax.set_ylabel("纵向位置 v")
    fig.colorbar(im, ax=list(axes[1:]), shrink=.8, label="视差 / 半尺寸图像素（不是米）")
    fig.suptitle("同一对芦荟照片、同一标准答案；官方设置是否更好？", fontsize=16)
    fig.savefig(OUT / "comparison.png", dpi=125)
    plt.close(fig)

    # 固定几何区域：中间竖直叶片及其背景；三个面板坐标相同。
    xmin, xmax, ymin, ymax = 250, 420, 40, 330
    fig, axes = plt.subplots(1, 3, figsize=(10, 7), layout="constrained")
    for ax, data, title in zip(axes, [truth, classroom, official],
                               ["标准答案", "原课堂设置", "官方示例原设置"]):
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=128, interpolation="nearest")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymax, ymin)
        ax.set_title(title)
        ax.set_xlabel("左照片横向位置 u")
    axes[0].set_ylabel("左照片纵向位置 v")
    fig.colorbar(im, ax=list(axes), shrink=.7, label="视差 / 像素")
    fig.suptitle("放大同一片竖直叶子：关注轮廓位置和灰色缺失，不只看是否平滑", fontsize=12)
    fig.savefig(OUT / "leaf_zoom.png", dpi=125)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    record, official = run_official_main()
    area_pair = load_pair()
    pyr_pair = record["inputs"]
    truth = load_reference(official.shape)
    classroom = compute_disparity(*area_pair)
    assert np.array_equal(classroom, calculate(area_pair, BASE_PARAMS), equal_nan=True)
    assert np.array_equal(official, calculate(pyr_pair, record["params"], color=True), equal_nan=True)

    variants = {"classroom": classroom, "official_original": official,
                "official_params_on_area": calculate(area_pair, record["params"], color=True),
                "classroom_params_on_pyr": calculate(pyr_pair, BASE_PARAMS)}
    # 单项替换只用于定位，不从GT选出一个所谓最优设置。
    changes = {
        "classroom_only_official_range": dict(minDisparity=16, numDisparities=96),
        "classroom_only_block16": dict(blockSize=16),
        "classroom_only_official_penalties": dict(P1=216, P2=864),
        "classroom_only_prefilter0": dict(preFilterCap=0),
        "classroom_only_speckle32": dict(speckleRange=32),
    }
    for name, patch in changes.items():
        variants[name] = calculate(area_pair, BASE_PARAMS | patch)
    variants["classroom_only_color"] = calculate(area_pair, BASE_PARAMS, color=True)
    results = {name: stats_on(data, truth) for name, data in variants.items()}
    common = np.isfinite(truth) & np.isfinite(classroom) & np.isfinite(official)
    common_results = {name: stats_on(variants[name], truth, common)
                      for name in ("classroom", "official_original")}
    common_area = np.isfinite(truth) & np.isfinite(classroom) & np.isfinite(variants["official_params_on_area"])
    same_area_results = {name: stats_on(variants[name], truth, common_area)
                         for name in ("classroom", "official_params_on_area")}
    unique_groups = {}
    for name, data, other in [("classroom_only_answered", classroom, official),
                              ("official_only_answered", official, classroom)]:
        selected = np.isfinite(truth) & np.isfinite(data) & ~np.isfinite(other)
        unique_groups[name] = stats_on(data, truth, selected)

    # 附加相同可搜索ROI口径，不能用它替换全图覆盖率。
    height, width = truth.shape
    rect = (0, 0, width, height)
    rois, masks = {}, []
    for name, params in [("classroom", BASE_PARAMS), ("official_original", record["params"])]:
        x, y, w, h = cv2.getValidDisparityROI(rect, rect, params["minDisparity"],
                                            params["numDisparities"], params["blockSize"])
        rois[name] = [x, y, w, h]
        mask = np.zeros(truth.shape, bool)
        mask[y:y+h, x:x+w] = True
        masks.append(mask)
    shared_roi = masks[0] & masks[1]
    roi_results = {name: stats_on(variants[name], truth, shared_roi)
                   for name in ("classroom", "official_original")}
    output = dict(opencv=cv2.__version__, official_source_sha256=SOURCE_HASH,
                  official_params=record["params"], classroom_params=BASE_PARAMS,
                  official_invalid_raw=15*16, classroom_invalid_raw=-16,
                  ply=record["ply_skipped"], results=results, common_answered=common_results,
                  same_area_common_answered=same_area_results, unique_answered_groups=unique_groups,
                  valid_rois=rois, shared_roi_results=roi_results,
                  gt_outside_official_range=int((np.isfinite(truth) & ((truth < 16) | (truth >= 112))).sum()),
                  protocol="Full half-size GT; NN [::2,::2]*0.5; 0 unknown; not an official benchmark")
    (OUT / "results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / "disparities.npz", truth=truth, **variants)
    plot_comparison(truth, classroom, official, results, area_pair[0])
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
