"""只从已验证的结果生成教师对照图，不调参、不重算匹配。"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from validate_eth3d_stereo import DATA, OUT, read_image, read_pfm


def main():
    results = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    assert results["scene_count"] == 27
    lookup = {s["scene"]: s for s in results["scenes"]}
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#b8b8b8")
    # 固定展示一个直观的木箱场景与本次误差最大的电柜失败场景，明确选择依据。
    for scene, description in [("delivery_area_1s", "木箱场景：可用结果示例"),
                               ("electro_3s", "电柜场景：保留的失败案例")]:
        left = read_image(DATA / scene / "im0.png")
        truth = read_pfm(DATA / scene / "disp0GT.pfm")
        truth[~np.isfinite(truth)] = np.nan
        predictions = np.load(OUT / "predictions" / f"{scene}.npz")
        fig, axes = plt.subplots(1, 4, figsize=(18, 5), layout="constrained")
        axes[0].imshow(left, cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("左相机照片")
        for ax, data, title in zip(axes[1:], [truth, predictions["classroom"], predictions["opencv_cpp_hh"]],
                                  ["官方标准答案", "原课堂SGBM设置", "参照官方配置段的HH方案"]):
            im = ax.imshow(data, cmap=cmap, vmin=0, vmax=32, interpolation="nearest")
            ax.set_title(title)
        axes[1].set_xlabel("灰色：标准答案未知")
        for ax, method in zip(axes[2:], ["classroom", "opencv_cpp_hh"]):
            stat = lookup[scene]["methods"][method]["metrics"]["all"]
            ax.set_xlabel(f"作答 {stat['coverage']:.1%}；作答处误差 {stat['avgerr']:.2f} px\n灰色：算法未作答")
        for ax in axes:
            ax.set_ylabel("纵向像素 v")
        fig.colorbar(im, ax=list(axes[1:]), extend="max", shrink=.8,
                     label="视差 / 原图像素（超过32以顶端颜色显示）")
        fig.suptitle(f"{description} · {scene}\n同一位置、同一刻度；结果不是米数", fontsize=14)
        fig.savefig(OUT / f"{scene}_comparison.png", dpi=120)
        plt.close(fig)

    lines = ["# 27对逐场景结果", "", "每格为：作答比例 / 各自作答处平均误差（像素）。All口径，不排除左边带。", "",
             "| 场景 | 原课堂 | Aloe范围 | 官方配置段SGBM | 官方配置段HH |",
             "|---|---:|---:|---:|---:|"]
    for scene in results["scenes"]:
        cells = []
        for method in results["methods"]:
            stat = scene["methods"][method]["metrics"]["all"]
            cells.append(f"{stat['coverage']:.2%} / {stat['avgerr']:.3f}")
        lines.append("| " + scene["scene"] + " | " + " | ".join(cells) + " |")
    (OUT / "per_scene_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
