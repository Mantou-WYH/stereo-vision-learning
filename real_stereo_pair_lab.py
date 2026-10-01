"""先认识真实左右照片；本轮不计算深度、不调SGBM参数。"""
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image


def main(save_path=None):
    folder = Path(__file__).resolve().parent / 'data' / 'opencv-aloe'
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout='constrained')
    for ax, name, title in zip(axes, ['aloeL.jpg', 'aloeR.jpg'],
                               ['左相机照片', '右相机照片']):
        with Image.open(folder / name) as source:
            im = source.convert('RGB')
        print(title, im.size)
        ax.imshow(im)
        ax.set_title(title)
        ax.set_xlabel('横向像素位置 u')
        ax.set_ylabel('纵向像素位置 v')
    fig.suptitle('同一个场景，两个观察位置：先找同一片叶子在两图中的位置', fontsize=14)
    if save_path:
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    main()
