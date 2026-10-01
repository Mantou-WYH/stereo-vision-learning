"""教师对照：用分段较短搜索范围尝试恢复木箱左侧SGBM灰带。

这是现成接口的多次运行近似，不是修改OpenCV内核的逐位置候选算法。
原预测不改动；全部新预测完成后才读参考评价。直接运行可重现结果图。
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

# 与原课堂配置一致；启动时另以逐像素回归确认，不导入会固定绘图后端的备课脚本。
BASE_PARAMS = dict(minDisparity=0, numDisparities=128, blockSize=5,
                   P1=200, P2=800, disp12MaxDiff=1, preFilterCap=31,
                   uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
                   mode=cv2.STEREO_SGBM_MODE_SGBM)

ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "data/eth3d/delivery_area_1s"
SAVED = ROOT / "notes/eth3d_validation/predictions/delivery_area_1s.npz"
OUT = ROOT / "notes/eth3d_validation/border_range_trial"
COUNTS = tuple(range(16, 128, 16))


def decoded(raw, minimum):
    disparity = raw.astype(np.float32) / 16.0
    disparity[raw <= (minimum - 1) * 16] = np.nan
    return disparity


def read_reference(path):
    with path.open("rb") as stream:
        assert stream.readline().strip() == b"Pf"
        width, height = map(int, stream.readline().split())
        scale = float(stream.readline())
        assert abs(scale) == 1
        values = np.frombuffer(stream.read(), dtype="<f4" if scale < 0 else ">f4")
    return np.flipud(values.reshape(height, width))


def compute(left, right, count):
    params = BASE_PARAMS | {"numDisparities": count}
    raw = cv2.StereoSGBM_create(**params).compute(left, right)
    return decoded(raw, params["minDisparity"])


def segmented(left, right, baseline):
    trial = baseline.copy()
    runs = {}
    for count in COUNTS:
        prediction = compute(left, right, count)
        runs[f"num_{count}"] = prediction
        trial[:, count:count + 16] = prediction[:, count:count + 16]
    assert np.array_equal(trial[:, 128:], baseline[:, 128:], equal_nan=True)
    assert np.isnan(trial[:, :16]).all()
    return trial, runs


def usable(pred):
    return np.isfinite(pred) & (pred > 0)


def summarize(pred, truth, domain):
    known = domain & np.isfinite(truth)
    answered = known & usable(pred)
    errors = np.abs(pred[answered].astype(np.float64) - truth[answered])
    n, a = int(known.sum()), int(answered.sum())
    return {"reference_count": n, "answered": a,
            "coverage_percent": 100 * a / n if n else None,
            "mae_px": float(errors.mean()) if a else None,
            "bad2_percent_of_answered": float(100 * np.mean(errors > 2)) if a else None,
            "within2_count": int(np.sum(errors <= 2)),
            "within2_percent_of_reference": float(100 * np.sum(errors <= 2) / n) if n else None}


def synthetic_check():
    rng = np.random.default_rng(20260928)
    left = rng.integers(0, 256, (72, 320), dtype=np.uint8)
    right = rng.integers(0, 256, left.shape, dtype=np.uint8)
    right[:, :-12] = left[:, 12:]
    baseline = compute(left, right, 128)
    trial, _ = segmented(left, right, baseline)
    area = trial[8:-8, 24:128]
    fraction = float(usable(area).mean())
    error = float(np.median(np.abs(area[usable(area)] - 12)))
    assert fraction > .90 and error < .10, (fraction, error)
    return {"known_disparity_px": 12, "region": "rows 8:-8, columns 24:128",
            "usable_fraction": fraction, "median_error_px": error}


def render(photo, baseline_z, trial_z, stats, save_only):
    import matplotlib
    if save_only:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(17.4, 7.0))
    fig.subplots_adjust(left=.055, right=.965, top=.77, bottom=.28, wspace=.19)
    fig.suptitle("让左侧灰带重新参与匹配：一次分段搜索试验", fontsize=22, y=.955)
    fig.text(.5, .875, "只在原前128列选入新匹配；第128列以后的原结果逐像素保留", ha="center", fontsize=15)
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad("#b8b8b8")
    cmap.set_over("#bd2870")
    axes[0].imshow(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB))
    for ax, z in zip(axes[1:], (baseline_z, trial_z)):
        im = ax.imshow(np.ma.masked_invalid(z), cmap=cmap, vmin=0, vmax=24, interpolation="nearest")
    titles = ["左照片：橙框内是本次关注区域", "原结果：完整搜索范围0～127", "试验结果：边缘逐段缩短范围"]
    for ax, title in zip(axes, titles):
        ax.add_patch(Rectangle((-.5, -.5), 128, 435, fill=False, edgecolor="#ff7744", linewidth=2))
        ax.set_title(title, fontsize=13, pad=12)
        ax.set_xlabel("横向位置 u（像素）", fontsize=11)
        ax.set_ylabel("纵向位置 v（像素）", fontsize=11)
        ax.set_xlim(-.5, 710.5)
        ax.set_ylim(434.5, -.5)
    cb = fig.colorbar(im, cax=fig.add_axes([.39, .19, .55, .025]), orientation="horizontal",
                      extend="max", ticks=[0, 4, 8, 12, 16, 20, 24])
    cb.set_label("前向深度（米）；紫红三角端表示 >24 米；灰色仍未知", fontsize=12)
    new = stats["new_output"]
    fig.text(.055, .17, f"原灰带：55,680 个位置\n本次新增：{new['all_pixels']:,} 个深度", fontsize=13, color="#17635b")
    fig.text(.5, .067, "各段分别运行SGBM再拼接，是边界处理的近似试验；最左16列仍未知。参考答案只用于事后核对。",
             ha="center", fontsize=11, color="#465160")
    fig.savefig(OUT / "comparison.png", dpi=140)
    if save_only:
        plt.close(fig)
    else:
        plt.show()


def main(save_only=False):
    OUT.mkdir(parents=True, exist_ok=True)
    files = [SCENE / name for name in ("im0.png", "im1.png", "calib.txt", "disp0GT.pfm", "mask0nocc.png")] + [SAVED]
    before = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    check = synthetic_check()
    photos = [cv2.imdecode(np.fromfile(SCENE / name, np.uint8), cv2.IMREAD_UNCHANGED) for name in ("im0.png", "im1.png")]
    assert photos[0].shape[:2] == photos[1].shape[:2] == (435, 711)
    grays = [cv2.cvtColor(p, cv2.COLOR_BGR2GRAY) if p.ndim == 3 else p for p in photos]
    with np.load(SAVED) as s:
        baseline = s["classroom"].copy()
    repeat = compute(*grays, 128)
    assert np.array_equal(repeat, baseline, equal_nan=True), "原基线复现不一致，停止对照。"
    start = time.perf_counter()
    trial, runs = segmented(*grays, baseline)
    elapsed = time.perf_counter() - start
    # 先保存全部预测，然后才读取参考；选择规则不依赖答案。
    np.savez_compressed(OUT / "predictions.npz", baseline=baseline, trial=trial, **runs)
    calibration = dict(line.split("=", 1) for line in (SCENE / "calib.txt").read_text().splitlines() if "=" in line)
    assert float(calibration["doffs"]) == 0
    fB = float(calibration["cam0"].lstrip("[").split()[0]) * float(calibration["baseline"]) / 1000
    depths = []
    for pred in (baseline, trial):
        z = np.full(pred.shape, np.nan, dtype=np.float64)
        z[usable(pred)] = fB / pred[usable(pred)].astype(np.float64)
        depths.append(z)
    np.save(OUT / "trial_depth_m.npy", depths[1])
    truth = read_reference(SCENE / "disp0GT.pfm")
    nocc = cv2.imdecode(np.fromfile(SCENE / "mask0nocc.png", np.uint8), cv2.IMREAD_UNCHANGED) == 255
    assert nocc.shape == truth.shape == baseline.shape
    all_pixels = np.ones(truth.shape, dtype=bool)
    band = all_pixels.copy()
    band[:, 128:] = False
    new = usable(trial) & ~usable(baseline)
    assert not new[:, 128:].any()
    stats = {"opencv_version": cv2.__version__, "base_params": BASE_PARAMS,
             "method": "seven full SGBM runs with smaller disparity counts; predetermined 16-column strips",
             "synthetic_check": check, "baseline_exact_reproduction": True,
             "outside_band_unchanged": True, "counts": list(COUNTS),
             "new_output": {"all_pixels": int(new.sum()),
                            "with_reference": int((new & np.isfinite(truth)).sum()),
                            "without_reference": int((new & ~np.isfinite(truth)).sum())},
             "remaining_unknown_in_band": int((band & ~usable(trial)).sum()),
             "seven_runs_elapsed_seconds": elapsed, "regions": {}}
    for region_name, domain in (("whole_all", all_pixels), ("whole_nocc", nocc),
                                ("band_all", band), ("band_nocc", band & nocc),
                                ("new_all", new), ("new_nocc", new & nocc)):
        stats["regions"][region_name] = {name: summarize(pred, truth, domain)
                                        for name, pred in (("baseline", baseline), ("trial", trial))}
    after = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    assert before == after
    stats["input_hashes_unchanged"] = before
    for u, v in ((450, 350), (530, 130)):
        assert depths[0][v, u] == depths[1][v, u]
    (OUT / "results.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    render(photos[0], *depths, stats, save_only)
    print(json.dumps({k: stats[k] for k in ("new_output", "remaining_unknown_in_band", "regions")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-only", action="store_true")
    main(parser.parse_args().save_only)
