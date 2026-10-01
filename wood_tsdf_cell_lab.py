"""L08局部示意：原木箱深度查询八个采样位置，提取一小片表面。

使用前向差 D-Z 的TSDF约定，仅处理本例四正四负的单个计算单元。
非完整多帧TSDF实现；保留原双目估计误差。
"""
import argparse
import json
from itertools import product, combinations
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'notes/synthetic_mapping/tsdf_intro/real_cell'
F, CX, CY, BASELINE = 711.415, 376.739, 222.302, .0599701
MU = .05


def prepare_cell():
    with np.load(ROOT/'notes/eth3d_validation/point_cloud_lab/wood_cloud_classroom.npz') as cloud:
        a = cloud['xyz_m'][(cloud['pixel_uv'][:,0] == 450) & (cloud['pixel_uv'][:,1] == 350)][0]
    offsets = np.array(list(product([-.01,.01],repeat=3)))
    points = a+offsets
    image_uv = np.column_stack([F*points[:,0]/points[:,2]+CX,
                                F*points[:,1]/points[:,2]+CY])
    uv = np.floor(image_uv+.5).astype(int)
    with np.load(ROOT/'notes/eth3d_validation/predictions/delivery_area_1s.npz') as pred:
        disparity = pred['classroom'][uv[:,1],uv[:,0]]
    assert np.all(np.isfinite(disparity) & (disparity > 0))
    depth = F*BASELINE/disparity
    sdf = depth-points[:,2]
    assert np.all(np.abs(sdf) < MU)
    values = sdf/MU
    edges = [(i,j) for i,j in combinations(range(8),2)
             if np.count_nonzero(offsets[i] != offsets[j]) == 1]
    intersections = {}
    for i,j in edges:
        if values[i]*values[j] < 0:
            # 本例的变号边沿Z方向；通用情形需完整的连接规则。
            assert np.array_equal(offsets[i,:2],offsets[j,:2])
            t = values[i]/(values[i]-values[j])
            assert 0 <= t <= 1
            hit = points[i]+t*(points[j]-points[i])
            assert abs((1-t)*values[i]+t*values[j]) < 1e-12
            intersections[tuple(offsets[i,:2])] = hit
    assert len(edges) == 12 and len(intersections) == 4
    vertices = np.array([intersections[key] for key in [(-.01,-.01),(.01,-.01),(.01,.01),(-.01,.01)]])
    faces = np.array([[0,1,2],[0,2,3]])
    assert np.all(vertices >= points.min(axis=0)-1e-12)
    assert np.all(vertices <= points.max(axis=0)+1e-12)
    return a, offsets, points, image_uv, uv, depth, sdf, values, edges, vertices, faces


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save-only',action='store_true')
    args=parser.parse_args()
    a, offsets, points, image_uv, uv, depth, sdf, values, edges, vertices, faces=prepare_cell()
    OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUT/'cell.npz',origin_A_m=a,sample_xyz_m=points,
                        projected_uv=image_uv,sampled_uv=uv,observed_depth_m=depth,
                        sdf_m=sdf,tsdf=values,weights=np.ones(8),
                        vertices_m=vertices,triangles=faces,truncation_m=MU)
    report=dict(kind='single-frame local surface extraction with projective forward difference D-Z',
                evidence='teacher preparation; student execution pending',
                sampling_spacing_m=.02, truncation_m=MU,
                positive_samples=int((values>0).sum()), negative_samples=int((values<0).sum()),
                crossing_edges=4, vertices=4, triangles=2,
                limitations='One cell with this specific sign pattern. No multi-frame fusion, no reference-depth accuracy validation, no generic Marching Cubes implementation.',
                corners=[dict(offset_cm=(o*100).tolist(),pixel_uv=q.tolist(),
                              depth_m=float(d),sample_z_m=float(p[2]),sdf_mm=float(s*1000),tsdf=float(v))
                         for o,q,d,p,s,v in zip(offsets,uv,depth,points,sdf,values)])
    (OUT/'checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    import matplotlib
    if args.save_only:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    fig=plt.figure(figsize=(13,8))
    fig.suptitle('用原木箱深度：从8个距离记录，提取一小片表面',fontsize=21,y=.96)
    fig.text(.5,.89,'固定A附近2厘米计算单元；读取原估计深度，没有重新匹配。坐标以A为原点，单位厘米。',
             ha='center',fontsize=12,color='#42516a')
    local=(points-a)*100
    hits=(vertices-a)*100
    for left,fill in [(.05,False),(.55,True)]:
        ax=fig.add_axes([left,.25,.40,.56],projection='3d')
        for i,j in edges:
            p=local[[i,j]]
            ax.plot(p[:,0],p[:,2],p[:,1],color='#b7c1cb',lw=1)
        for p,s in zip(local,sdf):
            ax.scatter(p[0],p[2],p[1],c='#078ac5' if s>0 else '#ed842b',s=60,depthshade=False)
            if not fill:
                ax.text(p[0],p[2],p[1]-.14,f'{s*1000:+.1f}',ha='center',fontsize=11)
        ax.scatter(hits[:,0],hits[:,2],hits[:,1],c='#168361',s=65,depthshade=False)
        if fill:
            polygons=[hits[t][:,[0,2,1]] for t in faces]
            ax.add_collection3d(Poly3DCollection(polygons,facecolors=['#55b590','#a0d8c2'],
                                                alpha=.8,edgecolor='#126d50',linewidth=2))
        ax.set(title='② 4个零点连接成2片三角形' if fill else '① 8个采样位置各自查询深度（标注：距离差毫米）',
               xlabel='局部 X（厘米）',ylabel='局部 Z（厘米）',zlabel='局部 Y（厘米）',
               xlim=(-1.4,1.4),ylim=(-1.4,1.4),zlim=(1.4,-1.4))
        ax.set_xticks([-1,0,1]);ax.set_yticks([-1,0,1]);ax.set_zticks([-1,0,1])
        ax.set_box_aspect((1,1,1));ax.view_init(elev=23,azim=-57)
    fig.text(.5,.17,'蓝点：表面前方的正值　橙点：表面后方近邻的负值　绿点：沿变号边计算的表面零点',
             ha='center',fontsize=12,color='#42516a')
    fig.text(.5,.10,'本例四条边变号，所以得到四个顶点；画出其中一条对角线，将四边形分成两片三角形。',
             ha='center',fontsize=12)
    fig.text(.5,.035,'单帧局部演示，保留原深度估计误差；不是整幅网格、多帧融合或精度验收。',
             ha='center',fontsize=11,color='#42516a')
    fig.savefig(OUT/'comparison.png',dpi=140)
    print('8个采样位置：4正、4负；4条边产生零点；生成4个顶点、2片三角形。')
    print('输出位置：',OUT)
    if args.save_only:
        plt.close(fig)
    else:
        plt.show()


if __name__=='__main__':
    main()
