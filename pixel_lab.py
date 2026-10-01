import numpy as np
import matplotlib.pyplot as plt

# 生成高 4 像素、宽 6 像素的黑色图像
image = np.zeros((4, 6), dtype=np.uint8)

# 行号 1、列号 4 的像素设为白色
image[2, 4] = 255

# 显示图像，固定亮度范围
plt.imshow(image, cmap="gray", vmin=0, vmax=255,
           interpolation="nearest", origin="upper")

plt.xticks(range(6))
plt.yticks(range(4))
plt.xlabel("u (x)")
plt.ylabel("v (y)")

import sys

print("数组形状：", image.shape)
print("数据类型：", image.dtype)
print("指定像素值：", image[1, 4])
print("解释器：", sys.executable)
print("Python：", sys.version)
print("NumPy：", np.__version__)
plt.show()