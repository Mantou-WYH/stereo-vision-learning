# OpenCV 官方示例原件

- 来源：https://github.com/opencv/opencv/blob/4.12.0/samples/python/stereo_match.py
- 固定版本：4.12.0；下载核验日期：2026-09-26。
- `stereo_match.py` 保持下载原样；SHA-256：`0f2e5c4f27d4007e4fd959fe0b5d7a78a6296a079197dd213119b07b3532307a`。
- 同目录LICENSE取自相同标签的OpenCV仓库，只说明上游源码授权，不替图片或模型指定许可证。
- 实际运行入口是项目根目录 `compare_official_stereo.py`，调用本文件的main；只适配中文路径读取、非阻塞输出和跳过PLY写盘。
- 原示例的Q矩阵使用猜测焦距，不能把输出当米制点云。原main仍会打印 `out.ply saved`，但适配器已跳过该写盘，并在结果JSON的ply字段记录原因；本次没有生成该PLY。
- 官方示例是一份可复现参考实现，不等于最佳参数或官方榜单基线。源码的 `window_size=3` 仅用于P1/P2，实际传入 `blockSize=16`；本次未擅自修改。
