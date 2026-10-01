"""教师受控诊断：固定数据，分别检查搜索边带、唯一性、斑点过滤和惩罚。

不改课堂基线；参考只用于离线诊断。这是单图探索，不是独立测试集结论。
"""
import json
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from real_sgbm_lab import load_pair, compute_disparity
from sgbm_reference_lab import load_reference


def run_match(left, right, **changes):
    parameters = dict(minDisparity=0, numDisparities=128, blockSize=5,
                      P1=200, P2=800, disp12MaxDiff=1, preFilterCap=31,
                      uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
                      mode=cv2.STEREO_SGBM_MODE_SGBM)
    parameters.update(changes)
    raw = cv2.StereoSGBM_create(**parameters).compute(left, right)
    result = raw.astype(np.float32)/16
    result[raw <= -16] = np.nan
    return result


def stats(result, gt, mask):
    known = np.isfinite(gt) & mask
    answered = np.isfinite(result) & known
    error = np.abs(result[answered]-gt[answered])
    return dict(known=int(known.sum()), answered=int(answered.sum()),
                missing=int((known & ~answered).sum()),
                coverage=float(answered.sum()/known.sum()) if known.any() else None,
                mae=float(error.mean()) if error.size else None,
                bad2=float((error>2).mean()) if error.size else None)


def boundary_mask(gt, radius=2, jump=2):
    # 只在两个参考值都有效时检测相邻视差突变；不把参考缺失边界当深度边界。
    known = np.isfinite(gt)
    edge = np.zeros(gt.shape, dtype=bool)
    horizontal = known[:,1:] & known[:,:-1] & (np.abs(gt[:,1:]-gt[:,:-1])>jump)
    vertical = known[1:] & known[:-1] & (np.abs(gt[1:]-gt[:-1])>jump)
    edge[:,1:] |= horizontal
    edge[:,:-1] |= horizontal
    edge[1:] |= vertical
    edge[:-1] |= vertical
    return cv2.dilate(edge.astype(np.uint8), np.ones((2*radius+1,2*radius+1),np.uint8)).astype(bool)


def main():
    left, right = load_pair()
    gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    gt = load_reference(left.shape[:2])
    configs = {
        'baseline': {}, 'unique0': {'uniquenessRatio':0},
        'unique5': {'uniquenessRatio':5}, 'unique20': {'uniquenessRatio':20},
        'unique30': {'uniquenessRatio':30}, 'speckle0': {'speckleWindowSize':0},
        'unique0_speckle0': {'uniquenessRatio':0, 'speckleWindowSize':0},
        'P2_400': {'P2':400}, 'P2_1600': {'P2':1600},
        'block3': {'blockSize':3},
        # 不用0冒充关闭：将容许差放宽至整个搜索范围，明确属于诊断而非推荐。
        'consistency128': {'disp12MaxDiff':128},
        'all_loose': {'uniquenessRatio':0,'speckleWindowSize':0,'disp12MaxDiff':128}}
    results = {name: run_match(gray_l,gray_r,**p) for name,p in configs.items()}
    base = results['baseline']
    assert np.array_equal(base, compute_disparity(left,right), equal_nan=True)
    known = np.isfinite(gt)
    full = np.ones(gt.shape, dtype=bool)
    interior = full.copy()
    interior[:,:128] = False
    band = boundary_mask(gt) & interior
    baseline_valid = np.isfinite(base)
    report = dict(protocol='half resolution; nearest-neighbor GT; gt>0; bad iff abs error>2; exploratory one-scene diagnosis', configs=configs,
                  baseline_missing=dict(total=int((known & ~baseline_valid).sum()),
                  left128=int((known & ~baseline_valid & ~interior).sum()),
                  elsewhere=int((known & ~baseline_valid & interior).sum())), runs={})
    for name, result in results.items():
        valid = np.isfinite(result)
        added = known & ~baseline_valid & valid
        lost = known & baseline_valid & ~valid
        common = known & baseline_valid & valid
        report['runs'][name] = dict(
            full=stats(result,gt,full), interior=stats(result,gt,interior),
            boundary=stats(result,gt,band), away=stats(result,gt,interior & ~band),
            added=stats(result,gt,added), lost_count=int(lost.sum()),
            common_baseline=stats(base,gt,common), common_current=stats(result,gt,common))
    report['boundary_sensitivity'] = {}
    for radius in (1,2,4):
        b = boundary_mask(gt,radius) & interior
        answered = known & baseline_valid & interior
        bad = answered & (np.abs(base-gt)>2)
        report['boundary_sensitivity'][str(radius)] = dict(
            boundary=stats(base,gt,b), away=stats(base,gt,interior & ~b),
            fraction_of_answered=float((answered & b).sum()/answered.sum()),
            fraction_of_bad=float((bad & b).sum()/bad.sum()))
    folder = Path(__file__).resolve().parent/'notes'
    (folder/'sgbm_diagnosis_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,2,figsize=(12,8),layout='constrained')
    axes[0].imshow(cv2.cvtColor(left,cv2.COLOR_BGR2RGB))
    overlay=np.zeros((*gt.shape,4))
    overlay[band]=[1,.25,0,.65]
    axes[0].imshow(overlay)
    axes[0].set_title('橙色：参考视差明显跳变附近（仅分析用）',fontsize=11)
    loose=results['unique0']
    added=known & ~baseline_valid & np.isfinite(loose)
    state=np.zeros(gt.shape,dtype=np.uint8)
    state[added & (np.abs(loose-gt)<=2)]=1
    state[added & (np.abs(loose-gt)>2)]=2
    axes[1].imshow(state,cmap=ListedColormap(['#dddddd','#2a9d8f','#d84d40']),vmin=0,vmax=2,interpolation='nearest')
    axes[1].set_title('仅关闭唯一性检查：新增答案准不准？',fontsize=11)
    axes[1].legend(handles=[Patch(facecolor=c,label=t) for c,t in
                   [('#dddddd','灰：不是本次新增，或无参考'),('#2a9d8f','绿：新增，误差≤2像素'),('#d84d40','红：新增，误差>2像素')]],
                   loc='upper center',bbox_to_anchor=(.5,-.16),fontsize=10)
    for ax in axes:
        ax.set_xlabel('横向像素位置 u')
        ax.set_ylabel('纵向像素位置 v')
    fig.suptitle('固定同一对芦荟照片：定位边界问题，检查放宽过滤的代价',fontsize=14)
    fig.savefig(folder/'sgbm_diagnosis_preview.png',dpi=120)
    plt.close(fig)


if __name__=='__main__':
    main()
