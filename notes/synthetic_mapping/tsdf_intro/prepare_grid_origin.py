"""从已保存真实木箱点云展示人工规则网格的来源；不计算TSDF。"""
from pathlib import Path
from itertools import product, combinations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
with np.load(ROOT/'notes/eth3d_validation/point_cloud_lab/wood_cloud_classroom.npz') as data:
    xyz, uv = data['xyz_m'], data['pixel_uv']
    a = xyz[(uv[:,0] == 450) & (uv[:,1] == 350)][0]
    roi = (np.abs(uv[:,0]-450) <= 12) & (np.abs(uv[:,1]-350) <= 12)
    local = (xyz[roi]-a)*100  # 相机轴方向保持不变，局部原点移到A，米转厘米。
    visible = np.all(np.abs(local) <= 4.5, axis=1)
    local = local[visible]
# 教师选择网格相位：以A为局部原点，采样中心位于±1、±3厘米。
grid = np.array(list(product([-3.,-1.,1.,3.], repeat=3)))
cube = np.array(list(product([-1.,1.],repeat=3)))
edges = [(i,j) for i,j in combinations(range(8),2) if np.count_nonzero(cube[i]!=cube[j])==1]
assert len(cube)==8 and len(edges)==12
assert all(np.isclose(np.linalg.norm(cube[i]-cube[j]),2.) for i,j in edges)
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False
fig=plt.figure(figsize=(16,7.2))
fig.suptitle('小立方体的来源：程序在点云所在空间建立规则网格',fontsize=22,y=.96)
fig.text(.5,.886,'灰点：木箱A附近已有的估计点云　蓝点：程序安排的固定采样位置　黄线：选出的相邻8点',
         ha='center',fontsize=12,color='#42516a')
titles=['① 已有双目点云：箱面上的采样点','② 加入人为网格：采样位置间距2厘米','③ 取相邻8个采样位置，组成小立方体']
for k,left in enumerate([.025,.355,.685]):
    ax=fig.add_axes([left,.24,.29,.55],projection='3d')
    if k<2:
        ax.scatter(local[:,0],local[:,2],local[:,1],c='#555e68',s=5,alpha=.5,linewidths=0,depthshade=False)
    if k==1:
        ax.scatter(grid[:,0],grid[:,2],grid[:,1],c='#1585bc',s=17,depthshade=False)
    if k>=1:
        for i,j in edges:
            p=cube[[i,j]]
            ax.plot(p[:,0],p[:,2],p[:,1],color='#d49b17',lw=2.5)
        ax.scatter(cube[:,0],cube[:,2],cube[:,1],c='#1585bc',s=45,depthshade=False)
    ax.set(title=titles[k],xlabel='局部 X（厘米）',ylabel='局部 Z（厘米）',zlabel='局部 Y（厘米）')
    lim=1.8 if k==2 else 4.5
    ax.set(xlim=(-lim,lim),ylim=(-lim,lim),zlim=(lim,-lim))
    ticks=[-1,0,1] if k==2 else [-4,0,4]
    ax.set_xticks(ticks);ax.set_yticks(ticks);ax.set_zticks(ticks)
    ax.set_box_aspect((1,1,1));ax.view_init(elev=22,azim=-58)
fig.text(.5,.145,'蓝点不需要与灰点重合；蓝点所在位置甚至可以在空气中。它们用于保存TSDF值和权重。',
         ha='center',fontsize=13,color='#42516a')
fig.text(.5,.083,'这里蓝点对应体素中心；提取表面时，把相邻8个中心作为小立方体的8个角。右图已放大，请看刻度。',
         ha='center',fontsize=11,color='#42516a')
fig.text(.5,.027,f'左图复用已有ETH3D木箱点云，局部显示{len(local)}点；网格为教师选定，仅作位置示意，尚未填入真实TSDF值。',
         ha='center',fontsize=10,color='#42516a')
fig.savefig(OUT/'point_cloud_to_grid.png',dpi=130)
plt.close(fig)
print(f'局部显示{len(local)}个已有点；新增64个示意采样位置；小立方体8角、12边、边长2厘米。')
