# 双目视觉实验学习 / Stereo Vision Learning

用小型可视化实验理解双目匹配、米制深度、三维点云、表面融合和相机运动估计。教学保持 L00—L11 架构，逐步连接本科研究准备。

Small visual experiments for stereo matching, metric depth, point clouds, surface fusion, and camera motion estimation.

## 内容与入口

| 主题 | 示例入口 |
|---|---|
| 像素、投影与坐标变换 | `pixel_lab.py`、`projection_lab.py`、`camera_pose_lab.py` |
| 标定、畸变与极线 | `calibration_distortion_lab.py`、`epipolar_rectification_lab.py` |
| 窗口匹配与视差 | `ssd_assignment.py`、`stereo_matching_1d_lab.py`、`row_disparity_lab.py` |
| 真实双目与评估 | `real_sgbm_lab.py`、`validate_eth3d_stereo.py` |
| 深度与点云 | `wood_point_depth_lab.py`、`wood_depth_map_lab.py`、`wood_point_cloud_lab.py` |
| 模拟多帧与表面融合 | `two_view_mapping_lab.py`、`two_view_voxel_lab.py`、`two_view_tsdf_lab.py` |
| PnP、异常对应与跟踪 | `pnp_pose_lab.py`、`pnp_ransac_lab.py`、`tracked_pnp_motion_lab.py` |
| 用估计位姿放置点云 | `estimated_pose_mapping_lab.py` |

课程导航见 [教学路线](docs/教学路线.md)。部分 assignment 保留课堂练习状态；代码存在不表示已独立实现或完成验收。

## 运行

沿用已有 Python 环境即可，不需要重新配置 CUDA 或 ROS。常用依赖为 NumPy、Matplotlib、OpenCV、Pillow，以及部分表面实验使用的 scikit-image。新环境可参考 `requirements.txt`；它是依赖清单，不是完整环境锁定文件。

先运行无需外部数据的模拟实验，例如：

```powershell
python two_view_mapping_lab.py
python tracked_pnp_motion_lab.py --save-only
python estimated_pose_mapping_lab.py
```

后两步按顺序运行：第一步保存跟踪与 PnP 的输入和估计位姿，第二步读取它们放置点云。生成的 `notes/` 结果保留在本地。

真实照片实验依赖本地 `data/`，部分还需要先生成保存的视差或深度结果。来源和恢复顺序见 [数据与结果准备](docs/数据与结果准备.md)。仓库不包含原始数据、模型权重、论文 PDF 或课堂截图。

## 研究方向

计划研究离线单目大模型 Teacher 为轻量双目 Student 提供监督，并评估深度及建图结果。训练、创新性、实时性与发表均需后续实验验证。

当前公开内容是教学代码与整理说明。详细学习日志和个人材料在本地保存，公开仓库不承担完整实验归档。

## 来源与许可

`third_party/opencv_4_12_0/` 保留 OpenCV 官方示例、来源说明与原许可证。第三方数据和论文适用各自的使用条件。本项目原创内容尚未选定开源许可证；公开可见不代表已授权任意再分发。
