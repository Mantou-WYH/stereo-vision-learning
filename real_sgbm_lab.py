"""真实照片首次SGBM实验：只认识视差图，不做深度或真值评估。

参数参考：https://docs.opencv.org/4.12.0/d2/d85/classcv_1_1StereoSGBM.html
"""
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def load_pair():
    folder = Path(__file__).resolve().parent / 'data' / 'opencv-aloe'
    images = []
    for name in ('aloeL.jpg', 'aloeR.jpg'):
        # imdecode配合fromfile，兼容Windows中文路径。
        image = cv2.imdecode(np.fromfile(folder / name, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'无法读取图片：{name}')
        image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2),
                           interpolation=cv2.INTER_AREA)
        images.append(image)
    if images[0].shape != images[1].shape:
        raise ValueError('左右图尺寸必须一致')
    return images


def compute_disparity(left, right, block_size=5):
    if block_size < 1 or block_size % 2 == 0:
        raise ValueError('窗口宽度必须为正奇数')
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    # 固定的演示起点，不是经过验证集选择的最优参数。
    # 默认5×5二维窗口；对照实验只改block_size。
    # P1/P2固定200/800，不随窗口面积重新计算，避免同时改变变量。
    matcher = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=128, blockSize=block_size,
        P1=8*5*5, P2=32*5*5, disp12MaxDiff=1,
        preFilterCap=31, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM)
    raw = matcher.compute(left_gray, right_gray)
    # 返回值以1/16像素编码。minDisparity=0时，无效标记为-16。
    disparity = raw.astype(np.float32) / 16.0
    disparity[raw <= -16] = np.nan
    return disparity


def main(save_path=None):
    left, right = load_pair()
    disparity = compute_disparity(left, right)
    valid = np.isfinite(disparity)
    assert disparity.shape == left.shape[:2]
    assert valid.any()
    assert np.all((disparity[valid] >= 0) & (disparity[valid] < 128))
    print('OpenCV：', cv2.__version__)
    print('左右图片均缩小一半，计算尺寸：', left.shape[:2])
    print(f'算法保留结果比例：{valid.mean():.1%}（不代表正确率）')
    print('视差单位为缩小后图像的像素；零视差不是无效值。')
    print('灰色表示未输出有效估计，原因可能包括搜索边界、遮挡或过滤。')
    print('没有读取参考答案；没有相机标定，未计算米制距离。')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), layout='constrained')
    for ax, image, title in zip(axes[:2], [left, right], ['输入：左相机照片', '输入：右相机照片']):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
    colors = plt.get_cmap('viridis').copy()
    colors.set_bad('#b8b8b8')
    plot = axes[2].imshow(disparity, cmap=colors, vmin=0, vmax=128, interpolation='nearest')
    axes[2].set_title('输出：左图各位置的估计视差')
    fig.colorbar(plot, ax=axes[2], shrink=.75, ticks=[0, 32, 64, 96, 128],
                 label='视差（缩小后图像的像素）')
    axes[2].legend(handles=[Patch(facecolor='#b8b8b8', label='灰色：未输出有效估计')],
                   loc='upper center', bbox_to_anchor=(.5, -.18))
    for ax in axes:
        ax.set_xlabel('横向像素位置 u')
        ax.set_ylabel('纵向像素位置 v')
    fig.suptitle('同一盆芦荟：照片显示外观；视差图显示对应点的位置差，不是米数', fontsize=14)
    if save_path:
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    main()
