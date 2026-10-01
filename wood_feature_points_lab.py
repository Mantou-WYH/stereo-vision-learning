"""L09：在已有木箱左照片中检测二维特征点，暂不跟踪或估计运动。"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'notes/feature_tracking/wood_detection'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save-only', action='store_true')
    args = parser.parse_args()
    import cv2
    photo = np.asarray(Image.open(ROOT/'data/eth3d/delivery_area_1s/im0.png').convert('L'))
    corners = cv2.goodFeaturesToTrack(photo, maxCorners=100, qualityLevel=.01,
                                     minDistance=15, blockSize=3, useHarrisDetector=False)
    if corners is None:
        raise RuntimeError('本图未检测到特征点')
    uv = corners.reshape(-1, 2)
    height, width = photo.shape
    assert np.all(np.isfinite(uv))
    assert np.all((uv[:, 0] >= 0) & (uv[:, 0] < width))
    assert np.all((uv[:, 1] >= 0) & (uv[:, 1] < height))
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT/'detected_uv_px.npy', uv)
    report = dict(method='Shi-Tomasi / cv2.goodFeaturesToTrack', opencv_version=cv2.__version__,
                  input='data/eth3d/delivery_area_1s/im0.png', shape_hw=list(photo.shape),
                  parameters=dict(maxCorners=100, qualityLevel=.01, minDistance_px=15, blockSize=3),
                  detected_count=len(uv), output='2D (u,v) coordinates in native image pixels',
                  evidence='Teacher detection and figure check; student run pending',
                  limits='No temporal image, matching, depth filtering, PnP or trajectory estimation in this step')
    (OUT/'checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    import matplotlib
    if args.save_only:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.3))
    fig.subplots_adjust(left=.055, right=.975, top=.79, bottom=.23, wspace=.15)
    fig.suptitle('程序先从照片中挑出一些可定位的标记', fontsize=19, y=.96)
    fig.text(.5, .87, '同一原生木箱照片；绿色圆圈是程序检测结果，照片没有移动。',
             ha='center', fontsize=12)
    for ax, title in zip(axes, ['原左照片：711 × 435', f'Shi-Tomasi检测：{len(uv)}个二维特征点']):
        ax.imshow(photo, cmap='gray', vmin=0, vmax=255)
        ax.set(title=title, xlabel='u 向右（像素）', ylabel='v 向下（像素）')
    axes[1].scatter(uv[:, 0], uv[:, 1], s=42, facecolors='none', edgecolors='#38ef81', linewidths=1.1)
    fig.text(.5, .13, '观察绿色圈附近的小块：箱沿、接缝、纹理交汇等位置较容易被选中。',
             ha='center', fontsize=12)
    fig.text(.5, .05, '当前输出只有像素坐标；尚未得到下一帧对应，尚未计算相机运动。',
             ha='center', fontsize=11, color='#42516a')
    fig.savefig(OUT/'comparison.png', dpi=150)
    if args.save_only:
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    main()
