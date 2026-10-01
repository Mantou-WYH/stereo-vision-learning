"""L08：两帧合成深度按已知位姿写入同一TSDF网格，再提取可见表面。

NumPy实现投影TSDF，使用现有scikit-image提取网格。教学用有限体积，
理想输入与已知位姿，不等同Open3D实作验收或真实地图精度验证。
"""
import argparse
import json
from itertools import product
from pathlib import Path
import numpy as np
from skimage.measure import marching_cubes
from two_view_mapping_lab import build_views, FX, FY, CX, CY, WIDTH, HEIGHT

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'notes/synthetic_mapping/tsdf_two_views'
STEP=.025
MU=.08
ORIGIN=np.array([-1.007,-.807,2.907])
LIMIT=np.array([1.02,.82,4.92])


def make_grid():
    axes=[np.arange(a,b+STEP/2,STEP) for a,b in zip(ORIGIN,LIMIT)]
    xyz=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1)
    return xyz


def integrate(view,xyz,field,weights):
    """R_wc的列为相机轴在地图中的方向；行向量反变换使用R_wc。"""
    shape=weights.shape
    camera=(xyz.reshape(-1,3)-view['camera']) @ view['R_wc']
    z=camera[:,2]
    safe_z=np.where(z>0,z,1.)
    u=np.floor(FX*camera[:,0]/safe_z+CX+.5).astype(int)
    v=np.floor(FY*camera[:,1]/safe_z+CY+.5).astype(int)
    in_image=(z>0)&(u>=0)&(u<WIDTH)&(v>=0)&(v<HEIGHT)
    selected=np.flatnonzero(in_image)
    depth=view['depth'][v[selected],u[selected]]
    valid=np.isfinite(depth)&(depth>0)
    selected=selected[valid];depth=depth[valid]
    # 与常见投影TSDF一致：把前向差换为该像素射线方向的距离差。
    ray_factor=np.sqrt(1+((u[selected]-CX)/FX)**2+((v[selected]-CY)/FY)**2)
    sdf=(depth-z[selected])*ray_factor
    accepted=sdf > -MU
    selected=selected[accepted]
    observed=np.minimum(sdf[accepted]/MU,1.)
    old_w=weights.reshape(-1)[selected].astype(float)
    old_f=field.reshape(-1)[selected].astype(float)
    new_f=(old_f*old_w+observed)/(old_w+1)
    field.reshape(-1)[selected]=new_f
    weights.reshape(-1)[selected]+=1
    assert weights.shape==shape
    assert np.all(np.abs(field[weights>0])<=1.000001)
    return dict(updated_samples=len(selected),new_samples=int((old_w==0).sum()),
                repeated_samples=int((old_w>0).sum()))


def valid_cells(weights):
    good=np.ones(tuple(n-1 for n in weights.shape),dtype=bool)
    for offset in product([0,1],repeat=3):
        sl=tuple(slice(o,o+n-1) for o,n in zip(offset,weights.shape))
        good &= weights[sl]>0
    return good


def extract(field,weights):
    # 未知采样仅以+1占位供算法运行，随后移除所有触及未知角点的单元面片。
    values=np.where(weights>0,field,1.).astype(np.float32)
    vertices,faces,_,_=marching_cubes(values,level=0,allow_degenerate=False)
    centers=vertices[faces].mean(axis=1)
    cell=np.floor(centers+1e-7).astype(int)
    good=valid_cells(weights)
    inside=np.all(cell>=0,axis=1)&np.all(cell<np.array(good.shape),axis=1)
    keep=np.zeros(len(faces),dtype=bool)
    q=cell[inside]
    keep[inside]=good[q[:,0],q[:,1],q[:,2]]
    faces=faces[keep]
    assert len(faces)>0
    assert np.all(good[cell[keep,0],cell[keep,1],cell[keep,2]])
    used,remap=np.unique(faces,return_inverse=True)
    vertices=vertices[used].astype(float)*STEP+ORIGIN
    faces=remap.reshape(-1,3)
    assert np.isfinite(vertices).all()
    return vertices,faces


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save-only',action='store_true')
    args=parser.parse_args()
    first,second=build_views()
    xyz=make_grid()
    field=np.zeros(xyz.shape[:-1],dtype=np.float32)
    weights=np.zeros(field.shape,dtype=np.uint8)
    OUT.mkdir(parents=True,exist_ok=True)
    snapshots=[];stats=[]
    for number,view in enumerate([first,second],start=1):
        info=integrate(view,xyz,field,weights)
        vertices,faces=extract(field,weights)
        snapshots.append((vertices,faces))
        info.update(frames=number,vertices=len(vertices),triangles=len(faces),
                    known_samples=int((weights>0).sum()),unknown_samples=int((weights==0).sum()),
                    samples_seen_twice=int((weights==2).sum()))
        stats.append(info)
        np.savez_compressed(OUT/f'after_{number}.npz',tsdf=field,weights=weights,
                            vertices_m=vertices,triangles=faces,
                            origin_m=ORIGIN,spacing_m=STEP,truncation_m=MU)
        print(json.dumps(info,ensure_ascii=False))
    # 可见性核验：第一帧仅正面；第二帧带来向后延伸的右侧区域。
    # 射线长度修正及图像最近采样会使线性零点插值产生微小偏差。
    first_plane_residual=float(np.max(np.abs(snapshots[0][0][:,2]-3.1)))
    assert first_plane_residual<1e-4
    p=snapshots[1][0]
    assert np.any((np.abs(p[:,0]-.8)<.04)&(p[:,2]>4))
    assert stats[1]['samples_seen_twice']>0 and stats[1]['new_samples']>0
    report=dict(kind='teacher-run NumPy projective TSDF plus scikit-image Marching Cubes',
                depth_pose_source='existing two-view ideal synthetic box, teacher-set known poses',
                voxel_spacing_m=STEP,truncation_m=MU,grid_shape=list(weights.shape),
                unknown_policy='Only triangles from cells with all eight observed samples are retained.',
                stages=stats,first_plane_max_abs_residual_m=first_plane_residual,
                evidence='Teacher preparation; student execution and independent implementation pending.',
                limits='No Open3D call, no noise/pose-error experiment, no real-data accuracy claim.')
    (OUT/'checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    import matplotlib
    if args.save_only:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    fig=plt.figure(figsize=(14,10))
    fig.suptitle('从局部面片到箱面：两帧深度逐次加入同一份TSDF',fontsize=21,y=.965)
    fig.text(.5,.911,'上：原来的两个模拟视角；下：加入第1帧、再加入第2帧后的表面网格。地图坐标单位米。',
             ha='center',fontsize=12,color='#42516a')
    for left,view,title in [(.10,first,'第1帧：看见正面'),(.59,second,'第2帧：看见正面和右侧面')]:
        ax=fig.add_axes([left,.635,.31,.225]);ax.imshow(view['picture'],interpolation='nearest')
        ax.set(title=title,xlabel='图像 u（像素）',ylabel='图像 v（像素）')
    for k,left in enumerate([.055,.555]):
        ax=fig.add_axes([left,.165,.39,.365],projection='3d')
        vertices,faces=snapshots[k]
        polygons=vertices[faces][:,:,[0,2,1]]
        mesh=Poly3DCollection(polygons,facecolor='#62b99b',edgecolor='#367d65',linewidth=.12,alpha=1)
        ax.add_collection3d(mesh)
        ax.set(title='加入两帧：正面 + 新侧面' if k else '只加入第1帧：正面',
               xlabel='地图 X（米）',ylabel='地图 Z（米）',zlabel='Y 向下（米）',
               xlim=(-1,1),ylim=(2.9,4.9),zlim=(.8,-.8))
        ax.set_xticks([-1,0,1]);ax.set_yticks([3.1,4,4.7]);ax.set_zticks([-.6,0,.6])
        ax.set_box_aspect((2,2,1.6));ax.view_init(elev=22,azim=-53)
        fig.text(left+.195,.102,f'{len(faces):,} 片三角形',ha='center',fontsize=12)
    fig.text(.5,.049,'绿色表示提取的网格；上图蓝/橙表示箱子表面颜色。未观测的区域不生成面片。',
             ha='center',fontsize=11,color='#42516a')
    fig.text(.5,.019,'理想合成深度和已知位姿；采样间距2.5厘米，截断8厘米。教师演示，学生运行待确认。',
             ha='center',fontsize=10,color='#42516a')
    fig.savefig(OUT/'comparison.png',dpi=140)
    if args.save_only:
        plt.close(fig)
    else:
        plt.show()


if __name__=='__main__':
    main()
