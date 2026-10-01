"""同尺度比较估计与参考；参考仅用于展示，不进入SGBM。"""
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from real_sgbm_lab import load_pair, compute_disparity


def load_reference(shape):
    path = Path(__file__).resolve().parent / 'data/opencv-aloe/aloeGT.png'
    raw = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.ndim != 2 or raw.dtype != np.uint8:
        raise ValueError('参考图应为单通道8位PNG')
    h, w = shape
    assert raw.shape == (1110, 1282)
    scale = w / raw.shape[1]
    assert scale == h / raw.shape[0] == 0.5
    # 最近邻采样，不将无效0与有效值混合；显示用的简化降采样协议。
    sampled = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
    reference = sampled.astype(np.float32) * scale
    reference[sampled == 0] = np.nan
    return reference


def main(save_path=None):
    left, right = load_pair()
    estimate = compute_disparity(left, right)  # 不读取参考文件。
    reference = load_reference(estimate.shape)
    assert estimate.shape == reference.shape
    assert np.isfinite(reference).any()
    assert np.nanmax(reference) <= 127.5
    print('左右两张视差图都对应左相机照片，均为555×641。')
    print('共同单位：缩小后图像的像素。共同色条：0至128。')
    print('估计图灰色：算法无有效结果；参考图灰色：参考文件未知。')
    print('参考采用最近邻缩小并乘0.5；本轮只做可视对照，不是官方榜单评估。')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), layout='constrained')
    axes[0].imshow(cv2.cvtColor(left, cv2.COLOR_BGR2RGB))
    axes[0].set_title('位置参照：左相机照片')
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#b8b8b8')
    for ax, data, title, label in zip(axes[1:], [estimate, reference],
            ['程序估计：SGBM', '数据提供的参考视差'],
            ['灰色：算法没有有效估计', '灰色：参考文件没有答案']):
        plotted = ax.imshow(data, cmap=cmap, vmin=0, vmax=128, interpolation='nearest')
        ax.set_title(title)
        ax.set_xlabel('横向位置 u\n' + label)
    fig.colorbar(plotted, ax=list(axes[1:]), shrink=.75, ticks=[0,32,64,96,128],
                 label='共同刻度：视差（缩小后图像的像素）')
    axes[0].set_xlabel('横向位置 u')
    for ax in axes:
        ax.set_ylabel('纵向位置 v')
    fig.suptitle('同一位置、同一单位、同一颜色刻度：估计是否接近参考？', fontsize=14)
    if save_path:
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    main()
