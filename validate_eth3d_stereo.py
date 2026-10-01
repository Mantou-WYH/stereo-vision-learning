"""教师跨场景验证。方案先固定；GT仅在估计生成后用于评价。

官方C++示例配置段经预先显式选择，通过同一OpenCV Python接口复现，
不是C++原程序的编译执行，也不声称其CLI默认参数可直接运行。
评分实际调用ETH3D官方Windows可执行程序，并与独立实现交叉核对。
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from real_sgbm_lab import compute_disparity
from compare_official_stereo import BASE_PARAMS, decoded

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/eth3d"
OUT = ROOT / "notes/eth3d_validation"
EVALUATOR = DATA / "ETH3DTwoViewEvaluation/ETH3DTwoViewEvaluation.exe"
METHODS = ("classroom", "aloe_range", "opencv_cpp_sgbm", "opencv_cpp_hh")


def read_pfm(path):
    with Path(path).open("rb") as stream:
        assert stream.readline().strip() == b"Pf"
        width, height = map(int, stream.readline().split())
        scale = float(stream.readline())
        assert abs(scale) == 1.0
        array = np.frombuffer(stream.read(), dtype="<f4" if scale < 0 else ">f4")
    assert array.size == width * height
    return np.flipud(array.reshape(height, width)).astype(np.float32).copy()


def write_pfm(path, data):
    array = np.array(data, dtype=np.float32, copy=True)
    array[~np.isfinite(array)] = np.inf
    h, w = array.shape
    with Path(path).open("wb") as stream:
        stream.write(f"Pf\n{w} {h}\n-1\n".encode("ascii"))
        stream.write(np.flipud(array).astype("<f4").tobytes())


def read_image(path):
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    return image


def params_for(method, width, channels):
    if method == "classroom":
        return BASE_PARAMS.copy()
    if method == "aloe_range":
        return BASE_PARAMS | dict(minDisparity=16, numDisparities=96)
    return dict(minDisparity=0, numDisparities=((width // 8) + 15) & -16,
                blockSize=3, P1=8*channels*9, P2=32*channels*9,
                preFilterCap=63, uniquenessRatio=10, speckleWindowSize=100,
                speckleRange=32, disp12MaxDiff=1,
                mode=cv2.STEREO_SGBM_MODE_HH if method.endswith("hh") else cv2.STEREO_SGBM_MODE_SGBM)


def estimate(left, right, method):
    if method in ("classroom", "aloe_range") and left.ndim == 3:
        left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    channels = 1 if left.ndim == 2 else left.shape[2]
    params = params_for(method, left.shape[1], channels)
    matcher = cv2.StereoSGBM_create(**params)
    raw = matcher.compute(left, right)
    return decoded(raw, params["minDisparity"]), params


def metrics(prediction, truth, domain):
    known = np.isfinite(truth) & domain
    answered = known & np.isfinite(prediction) & (np.abs(prediction) <= truth.shape[1])
    errors = np.abs(prediction[answered].astype(np.float64) - truth[answered])
    assert known.any()
    return dict(reference_count=int(known.sum()), answered_count=int(answered.sum()),
                coverage=float(answered.sum()/known.sum()),
                avgerr=float(errors.mean()) if len(errors) else float("inf"),
                bad_1_0=float((errors > 1).mean()) if len(errors) else 1.0,
                bad_2_0=float((errors > 2).mean()) if len(errors) else 1.0,
                correct2_fraction=float((errors <= 2).sum()/known.sum()))


def official_metrics(scene, prediction_path):
    args = [str(EVALUATOR.relative_to(ROOT)), str(prediction_path.relative_to(ROOT)),
            str((DATA / scene / "disp0GT.pfm").relative_to(ROOT)),
            str((DATA / scene / "mask0nocc.png").relative_to(ROOT))]
    proc = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    sections, section = {}, None
    for line in proc.stdout.splitlines():
        if line == "Non-occluded:":
            section = "nocc"
            sections[section] = {}
        elif line == "All:":
            section = "all"
            sections[section] = {}
        elif section and ": " in line:
            key, value = line.split(": ", 1)
            sections[section][key] = float(value)
    assert set(sections) == {"all", "nocc"}
    return sections, proc.stdout


def unit_checks():
    # 官方有限GT、无效预测、严格>阈值与分母的独立小数组检查。
    truth = np.array([[1, 2, 3, 4, np.inf], [1, 1, 1, 1, 1]], np.float32)
    pred = np.array([[1, np.inf, 5, 1, 2], [1, 1, 1, 1, 1]], np.float32)
    m = metrics(pred, truth, np.ones(truth.shape, bool))
    assert m["reference_count"] == 9 and m["answered_count"] == 8
    assert m["bad_2_0"] == 1/8 and m["correct2_fraction"] == 7/9
    path = OUT / "pfm_roundtrip_check.pfm"
    write_pfm(path, pred)
    np.testing.assert_array_equal(read_pfm(path), pred)
    # 已知向左位移12px，检验视差正号、1/16尺度与中央区恢复。
    rng = np.random.default_rng(20260926)
    left = rng.integers(0, 256, (72, 320), dtype=np.uint8)
    right = rng.integers(0, 256, left.shape, dtype=np.uint8)
    right[:, :-12] = left[:, 12:]
    d, _ = estimate(left, right, "classroom")
    central = d[8:-8, 150:-20]
    assert np.isfinite(central).mean() > .95
    assert np.nanmedian(np.abs(central - 12)) < .1
    # 灰度原图与课堂入口BGR再转灰度相同，验证没有误改旧配置。
    bgr = [cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) for im in (left, right)]
    assert np.array_equal(d, compute_disparity(*bgr), equal_nan=True)


def main(limit=None):
    OUT.mkdir(parents=True, exist_ok=True)
    predictions_dir = OUT / "predictions"
    logs = OUT / "official_scores"
    predictions_dir.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    unit_checks()
    scenes = sorted(p.parent.name for p in DATA.glob("*/im0.png"))
    assert len(scenes) == 27
    if limit:
        scenes = scenes[:limit]
    records = []
    max_disagreement = {key: 0.0 for key in ("coverage", "avgerr", "bad_1_0", "bad_2_0")}
    for scene in scenes:
        folder = DATA / scene
        left, right = [read_image(folder / f"im{i}.png") for i in (0, 1)]
        assert left.shape == right.shape
        estimates, parameters, runtimes = {}, {}, {}
        # 必须先产生所有估计，再读取真值用于评价。
        for method in METHODS:
            start = time.perf_counter()
            estimates[method], parameters[method] = estimate(left, right, method)
            runtimes[method] = time.perf_counter() - start
        truth = read_pfm(folder / "disp0GT.pfm")
        mask = read_image(folder / "mask0nocc.png")
        if mask.ndim == 3:
            assert np.all(mask[..., 0] == mask[..., 1]) and np.all(mask[..., 0] == mask[..., 2])
            mask = mask[..., 0]
        assert truth.shape == left.shape[:2] == mask.shape
        assert set(np.unique(mask)) <= {0, 128, 255}
        assert np.array_equal(np.isfinite(truth), mask != 0)
        domains = {"all": np.ones(truth.shape, bool), "nocc": mask == 255}
        entry = dict(scene=scene, shape=list(left.shape), methods={})
        for method in METHODS:
            prediction = estimates[method]
            pfm = OUT / "current_prediction.pfm"
            write_pfm(pfm, prediction)
            official, stdout = official_metrics(scene, pfm)
            (logs / f"{scene}__{method}.txt").write_text(stdout, encoding="utf-8")
            own = {name: metrics(prediction, truth, domain) for name, domain in domains.items()}
            for name in domains:
                for key in max_disagreement:
                    difference = abs(own[name][key] - official[name][key])
                    max_disagreement[key] = max(max_disagreement[key], difference)
                    # 官方stdout默认6位有效数字，允许对应的舍入误差。
                    assert np.isclose(own[name][key], official[name][key], rtol=2e-5, atol=2e-6), (scene, method, name, key)
            entry["methods"][method] = dict(params=parameters[method], seconds=runtimes[method], metrics=own)
        entry["common_answered"] = {}
        for method in METHODS[1:]:
            common = np.isfinite(estimates["classroom"]) & np.isfinite(estimates[method])
            entry["common_answered"][method] = {
                "classroom": metrics(estimates["classroom"], truth, common),
                method: metrics(estimates[method], truth, common)}
        known = np.isfinite(truth)
        entry["gt_diagnostic_only"] = dict(min=float(truth[known].min()), max=float(truth[known].max()),
                    fraction_below16=float((truth[known] < 16).mean()),
                    fraction_outside_0_128=float(((truth[known] < 0) | (truth[known] >= 128)).mean()))
        np.savez_compressed(predictions_dir / f"{scene}.npz", **estimates)
        records.append(entry)
        print(scene, " ".join(f"{m}: cov={entry['methods'][m]['metrics']['all']['coverage']:.3f}, err={entry['methods'][m]['metrics']['all']['avgerr']:.3f}" for m in METHODS), flush=True)

    aggregate = {method: {domain: {key: float(np.mean([r["methods"][method]["metrics"][domain][key] for r in records]))
                      for key in ("coverage", "avgerr", "bad_1_0", "bad_2_0", "correct2_fraction")}
                    for domain in ("all", "nocc")} for method in METHODS}
    comparisons = {}
    for method in METHODS[1:]:
        comparisons[method] = {key: sum(r["methods"][method]["metrics"]["all"][key] > r["methods"]["classroom"]["metrics"]["all"][key] for r in records)
                               for key in ("coverage", "correct2_fraction")}
        comparisons[method]["lower_common_mae_pairs"] = sum(r["common_answered"][method][method]["avgerr"] < r["common_answered"][method]["classroom"]["avgerr"] for r in records)
    output = dict(scene_count=len(records), methods=list(METHODS), opencv=cv2.__version__,
                  evaluator_sha256=hashlib.sha256(EVALUATOR.read_bytes()).hexdigest(),
                  official_crosscheck_max_abs_difference=max_disagreement,
                  aggregate_pair_equal_weight=aggregate, improvements_vs_classroom=comparisons, scenes=records)
    (OUT / "results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in output.items() if k != "scenes"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    main(parser.parse_args().limit)
