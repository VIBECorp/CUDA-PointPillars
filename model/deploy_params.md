# rail_v4 PointPillars — Jetson AGX Orin 배포 파라미터

모델: `pointpillar.onnx` (CUDA-PointPillars TensorRT용, PPScatterPlugin 포함)
검증: `pointpillar_raw.onnx`를 onnxruntime으로 PyTorch 대비 최대 오차 1.8e-5 확인 (2026-08-12)
소스 체크포인트: `../ckpt/checkpoint_epoch_80.pth` (val AP_BEV@0.25 0.958, @0.5 0.782)

## ONNX 입출력

| 텐서 | shape | 설명 |
|---|---|---|
| `voxels` (in) | (75000, 10, 10) | 필라별 확장 특징 [x,y,z,i, xc,yc,zc, xp,yp,zp] |
| `voxel_idxs` (in) | (75000, 4) int32 | (frame, z, y, x) 인덱스 |
| `voxel_num` (in) | (1,) int32 | 유효 필라 수 |
| `cls_preds` (out) | (1, 320, 320, 4) | 앵커 4개 × 클래스 1 |
| `box_preds` (out) | (1, 320, 320, 28) | 앵커 4개 × 7 |
| `dir_cls_preds` (out) | (1, 320, 320, 8) | 앵커 4개 × dir 2빈 |

## C++ 파라미터 (src/pointpillar/*.hpp 와 반드시 동기화)

`VoxelizationParameter`:
```cpp
param.min_range = {-3.2, -3.2, -0.5};
param.max_range = { 3.2,  3.2,  1.5};
param.voxel_size = {0.01, 0.01, 2.0};
param.grid_size = {640, 640, 1};        // compute_grid_size 사용 가능
param.max_voxels = 75000;               // ONNX 입력과 일치 필수
param.max_points_per_voxel = 10;        // ONNX 입력과 일치 필수
param.max_points = 300000;
param.num_feature = 4;                  // x,y,z,intensity(그레이/255)
```

`PostProcessParameter` (기본값이 KITTI 3클래스이므로 전부 교체):
```cpp
param.min_range = {-3.2, -3.2, -0.5};
param.max_range = { 3.2,  3.2,  1.5};
param.feature_size = {320, 320};
param.num_classes = 1;                  // Rail
param.num_anchors = 4;
param.len_per_anchor = 4;
// {l, w, h, rot}
param.anchors = { 2.4, 0.1, 0.21, 0.0,
                  2.4, 0.1, 0.21, 0.785,
                  2.4, 0.1, 0.21, 1.57,
                  2.4, 0.1, 0.21, 2.355 };   // 배열 크기 16으로 수정 필요 (기본 24)
param.anchor_bottom_heights = {0.0};    // Float3 -> 클래스 1개면 첫 원소만 사용되는지 코드 확인
param.score_thresh = 0.1;               // 필요시 0.3 권장 (과예측 감소)
param.dir_offset = 0.78539;
param.nms_thresh = 0.01;
```

주의: 원본 코드는 3클래스/6앵커(고정 배열 24) 기준이라 `anchors[]` 배열 크기와
`anchor_bottom_heights` 처리 부분의 소스 수정이 필요하다. postprocess 커널이
`num_anchors`, `num_classes`를 파라미터로 쓰는지 상수로 가정하는지 빌드 전에 확인할 것.

## 런타임 전처리 (필수 — 이걸 빼면 검출이 무의미해짐)

학습 데이터는 프레임별로 다음 전처리가 적용되었다. Orin 런타임에서도 동일하게 재현해야 한다:
1. **중력 정렬**: 지면 평면 RANSAC → 법선을 +z로 회전 (카메라 피치가 프레임마다 다름)
2. **지면 z=0**: 학습 데이터는 "가까운 레일 옆 반경 0.5m 국소 지면" 기준
   (런타임에서는 직전 프레임 검출 레일 박스 기준으로 재현, 첫 프레임은 전역 RANSAC 부트스트랩)
3. intensity 채널 = RGB 그레이스케일/255 (컬러 없으면 0.5 고정으로 실험)
4. 검출 박스는 역변환해서 원 좌표계로 복귀

참고 구현: OpenPCDet/tools/convert_pointcept_rail.py (RANSAC/정렬),
tools/refine_ground_by_rail.py (국소 지면).

## TensorRT 엔진 빌드 (Orin에서)

```bash
trtexec --onnx=pointpillar.onnx --fp16 \
  --plugins=libpointpillar_core.so   # PPScatterPlugin 빌드 산출물
```
CUDA-PointPillars의 CMake 빌드가 플러그인을 포함한다. dense_shape=[640,640]은
ONNX 노드 속성에 포함되어 있음.
