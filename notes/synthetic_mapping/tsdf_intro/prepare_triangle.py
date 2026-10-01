"""教师示意：一个小立方体内，三条变号边的零点构成一个三角形。"""
from itertools import product, combinations
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUT = Path(__file__).resolve().parent
corners = np.array(list(product([0., 2.], repeat=3)))  # 局部厘米
field = (1.2-corners.sum(axis=1))/(np.sqrt(3)*5.)
edges = [(i,j) for i,j in combinations(range(8),2)
         if np.count_nonzero(corners[i] != corners[j]) == 1]
crossings = []
for i,j in edges:
    if field[i]*field[j] < 0:
        t = field[i]/(field[i]-field[j])
        crossings.append(corners[i]+t*(corners[j]-corners[i]))
triangle = np.array(crossings)
assert len(edges) == 12 and triangle.shape == (3,3)
assert np.allclose(triangle.sum(axis=1),1.2)
assert np.allclose(np.sort(triangle,axis=1),[[0,0,1.2]]*3)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei','SimHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(13,7.5))
fig.suptitle('一个三角形的三个顶点，从哪里来？',fontsize=22,y=.96)
fig.text(.5,.89,'局部放大：相邻8个距离采样位置构成小立方体；蓝色正值、橙色负值，均已有观测',
         ha='center',fontsize=12,color='#42516a')
for left, fill in [(.055,False),(.555,True)]:
    ax=fig.add_axes([left,.23,.39,.59],projection='3d')
    for i,j in edges:
        p=corners[[i,j]]
        ax.plot(p[:,0],p[:,1],p[:,2],color='#adb9c4',lw=1.1)
    for p,v in zip(corners,field):
        ax.scatter(*p,c='#078ac5' if v>0 else '#ec842b',s=65,depthshade=False)
        ax.text(*(p+np.array([0,0,.10])), '+' if v>0 else '−',ha='center',fontsize=17)
    for i,p in enumerate(triangle):
        ax.scatter(*p,c='#158361',s=65,depthshade=False)
        ax.text(*(p+np.array([.08,.02,.13])), 'PQR'[i],fontsize=13,color='#126d50')
    if fill:
        ax.add_collection3d(Poly3DCollection([triangle],facecolor='#50b18e',alpha=.7,
                                            edgecolor='#126d50',linewidth=2))
    else:
        for p in triangle:
            ax.plot([0,p[0]],[0,p[1]],[0,p[2]],color='#158361',lw=2.5)
    ax.set(title='② 连接 P、Q、R，得到一小片表面' if fill else '① 在三条正负变号的边上，计算零点 P、Q、R',
           xlabel='局部 X（厘米）',ylabel='局部 Y（厘米）',zlabel='局部 Z（厘米）',
           xlim=(-.15,2.15),ylim=(-.15,2.15),zlim=(-.15,2.3))
    ax.set_xticks([0,1,2]);ax.set_yticks([0,1,2]);ax.set_zticks([0,1,2])
    ax.set_box_aspect((1,1,1));ax.view_init(elev=23,azim=-55)
fig.text(.5,.135,'绿色顶点是沿边插值算出的表面位置；不要求原始点云恰好在这里有一个点。',
         ha='center',fontsize=13,color='#42516a')
fig.text(.5,.072,'这是最简单的一种局部情况。其他正负分布可能生成多个三角形；由相应连接规则决定。',
         ha='center',fontsize=11,color='#42516a')
fig.text(.5,.025,'教师设定的局部平面示意，仅说明表面提取；没有新增真实测量或完整TSDF融合。',
         ha='center',fontsize=10,color='#42516a')
fig.savefig(OUT/'triangle_from_zero_crossings.png',dpi=140)
plt.close(fig)
(OUT/'triangle_from_zero_crossings.json').write_text(json.dumps(dict(
    units='local centimetres',corner_positions=corners.tolist(),corner_values=field.tolist(),
    vertices=triangle.tolist(),plane='x+y+z=1.2 cm',
    role='teacher-generated single-case illustration, not a general marching-cubes implementation'
),ensure_ascii=False,indent=2),encoding='utf-8')
