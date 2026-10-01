"""L09：把已有深度与二维跟踪对应组装为3D–2D输入，不求位姿。"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'notes/feature_tracking/wood_3d_2d_inputs'


def main():
    depth = np.load(ROOT/'notes/eth3d_validation/wood_depth_map_classroom.npy')
    with np.load(ROOT/'notes/feature_tracking/wood_lk_translation/correspondences.npz') as data:
        uv0 = data['first_uv_px'].copy()
        uv1 = data['estimated_second_uv_px'].copy()
        tracked = data['usable'].copy()
    calibration = dict(line.split('=', 1) for line in
                       (ROOT/'data/eth3d/delivery_area_1s/calib.txt').read_text().splitlines()
                       if '=' in line)
    K = np.array([[float(x) for x in row.split()]
                  for row in calibration['cam0'].strip('[]').split(';')])
    # 本批检测坐标为整数；保持像素、深度与对应的索引相同。
    pixel = np.floor(uv0+.5).astype(int)
    assert np.array_equal(pixel, uv0)
    assert depth.shape == (435, 711)
    z = depth[pixel[:, 1], pixel[:, 0]]
    usable = tracked & np.isfinite(z) & (z > 0)
    source_indices = np.flatnonzero(usable)
    xyz = np.column_stack([(uv0[usable, 0]-K[0, 2])*z[usable]/K[0, 0],
                           (uv0[usable, 1]-K[1, 2])*z[usable]/K[1, 1], z[usable]])
    second_uv = uv1[usable]
    reprojected = np.column_stack([K[0, 0]*xyz[:, 0]/xyz[:, 2]+K[0, 2],
                                  K[1, 1]*xyz[:, 1]/xyz[:, 2]+K[1, 2]])
    roundtrip = np.linalg.norm(reprojected-uv0[usable], axis=1)
    assert np.all(np.isfinite(xyz)) and np.all(xyz[:, 2] > 0)
    assert roundtrip.max() < 1e-8  # 只检查几何计算与索引自洽。
    p_row = int(np.flatnonzero(source_indices == 50)[0])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT/'pnp_input_example.npz', points_in_first_camera_m=xyz,
                        second_uv_px=second_uv, first_uv_px=uv0[usable],
                        original_feature_indices=source_indices, K=K,
                        usable_for_input=usable)
    report = dict(tracked_points=int(tracked.sum()), usable_3d_2d_pairs=len(xyz),
                  excluded_missing_depth=int((tracked & ~usable).sum()),
                  coordinate_frame='First left camera: X right, Y down, Z forward; metres',
                  max_roundtrip_error_px=float(roundtrip.max()),
                  example=dict(original_feature_index=50, output_row=p_row,
                               first_uv_px=uv0[50].tolist(), estimated_depth_m=float(z[50]),
                               estimated_xyz_m=xyz[p_row].tolist(),
                               estimated_second_uv_px=second_uv[p_row].tolist()),
                  sources='Saved classroom SGBM-derived depth; LK correspondences from generated 2D translation test',
                  evidence='Teacher input assembly and checks; no student implementation/run',
                  limits='No PnP solve: generated global 2D translation is not a real 3D camera motion model. '
                         'Roundtrip consistency does not validate estimated depth accuracy.')
    (OUT/'checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
