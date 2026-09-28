# PointPillars Inference with TensorRT

This repository contains sources and model for [pointpillars](https://arxiv.org/abs/1812.05784) inference using TensorRT.

Overall inference has below phases:

- Voxelize points cloud into 10-channel features
- Run TensorRT engine to get detection feature
- Parse detection feature and apply NMS

## Rail ONNX → TensorRT 10 엔진 변환 (Jetson)

`tool/build_trt_engine.sh`는 모델 경로를 받아 FP16 `.plan`과 ROS 노드용
`.plan.grid`를 생성한다. 현재 작업 디렉터리와 관계없이 실행할 수 있다.
Python 3의 `tensorrt` 10.x 바인딩과, 같은 환경에서 빌드한
`build/libpointpillar_core.so`(PPScatterPlugin)가 필요하다.
`onnx` Python 패키지나 OpenPCDet는 이미 export된 ONNX 변환에는 필요 없다.

```bash
# CUDA-PointPillars 디렉터리에서 최초 1회 플러그인 빌드 (Orin)
CUDASM=87 cmake -S . -B build
cmake --build build --target pointpillar_core -j2

# 새 모델 추가 후 변환: model/my_rail.onnx → model/my_rail.plan + .plan.grid
bash tool/build_trt_engine.sh model/my_rail.onnx

# 현재 제공된 모델을 다른 이름으로 변환
bash tool/build_trt_engine.sh model/pointpillar_v11_g320.onnx \
  --output model/pointpillar_v11_g320_rebuilt.plan

# 파싱·입출력 호환성만 확인 (엔진 생성 없음)
bash tool/build_trt_engine.sh model/pointpillar_v11_g320.onnx --check-only

# 기존 엔진을 명시적으로 교체
bash tool/build_trt_engine.sh model/my_rail.onnx --force
```

인자를 생략하면 저장소의 `model/pointpillar.onnx`를 사용한다. 상대 경로 인자는
호출한 디렉터리 기준이고, 기본 플러그인 경로는 스크립트 위치 기준이다.
출력이 이미 있으면 기본적으로 거부한다. `--output`으로 새 경로를 지정하거나
`--force`를 사용한다. 빌드 실패 시 기존 엔진은 유지된다.
`--plugin /path/to/libpointpillar_core.so`, `--workspace-mib 2048`로 라이브러리와
TensorRT 작업 메모리 한도를 지정할 수 있다. 다른 Python 환경은 `PYTHON`으로 선택한다.

지원 인터페이스는 현재 rail `pointpillar_ros` 런타임 기준이다:

- 입력: `voxels (75000,32,10)`, `voxel_idxs (75000,4)`, `voxel_num (1,)`.
- 출력: `cls_preds`, `box_preds`, `dir_cls_preds`; 채널 수 각각 4, 28, 8.
- 출력 격자 160×160 / 320×320에서 입력 격자 320 / 640을 판별한다.
- 엔진 입력 형식은 FP16 / INT32 / INT32, 출력은 FP32로 맞춘다.

일반 `.pth` 체크포인트나 KITTI용 다른 입출력 구조는 직접 변환 대상이 아니다.
학습 환경에서 PPScatterPlugin을 포함한 호환 ONNX로 먼저 export해야 한다.
이 도구는 FP16 빌드를 설정하며 INT8 캘리브레이션이나 양자화를 수행하지 않는다.
Q/DQ가 포함된 ONNX는 원래 양자화 의미를 유지하므로 별도 정확도 검증이 필요하다.
입출력 검사만으로 학습 범위·앵커·전처리까지 호환됨을 보장하지는 않는다.

생성한 `.plan`과 `.plan.grid`는 함께 배포한다. ROS launch의 `model_path`를
새 `.plan`의 절대 경로로 설정하고 노드를 재시작해야 적용된다.
엔진은 실제 배포할 GPU/TensorRT 환경에서 생성한다. 스크립트는 빌드 후
역직렬화와 I/O 형식을 검사하지만 실제 포인트클라우드 검출 정확도는 별도 검증 대상이다.

구현 참고: [NVIDIA TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/python-api-docs.html),
[I/O formats](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/io-formats-sparsity.html).

아래 내용은 원본 KITTI 모델의 학습·export 및 과거 TensorRT 8 환경 참고 문서다.

## Prerequisites

### Prepare Model && Data

We provide a [Dockerfile](docker/Dockerfile) to ease environment setup. Please execute the following command to build the docker image after nvidia-docker installation:
```
cd docker && docker build . -t pointpillar
```
We can then run the docker with the following command: 
```
nvidia-docker run --rm -ti -v /home/$USER/:/home/$USER/ --net=host --rm pointpillar:latest
```
For model exporting, please run the following command to clone pcdet repo and install custom CUDA extensions:
```
git clone https://github.com/open-mmlab/OpenPCDet.git
cd OpenPCDet && git checkout 846cf3e && python3 setup.py develop
```
Download [PTM](https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view) to ckpts/, then use below command to export ONNX model:
```
python3 tool/export_onnx.py --ckpt ckpts/pointpillar_7728.pth --out_dir model
```
Use below command to evaluate on kitti dataset, follow [Evaluation on Kitti](tool/eval/README.md) to get more detail for dataset preparation.
```
sh tool/evaluate_kitti_val.sh
```

### Setup Runtime Environment

- Nvidia Jetson Orin + CUDA 11.4 + cuDNN 8.9.0 + TensorRT 8.6.11

## Compile && Run

```shell
sudo apt-get install git-lfs && git lfs install
git clone https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars.git
cd CUDA-PointPillars && . tool/environment.sh
mkdir build && cd build
cmake .. && make -j$(nproc)
cd ../ && sh tool/build_trt_engine.sh
cd build && ./pointpillar ../data/ ../data/ --timer
```

## FP16 Performance && Metrics

Average perf in FP16 on the training set(7481 instances) of KITTI dataset.

```
| Function(unit:ms) | Orin   |
| ----------------- | ------ |
| Voxelization      | 0.18   |
| Backbone & Head   | 4.87   |
| Decoder & NMS     | 1.79   |
| Overall           | 6.84   |
```

3D moderate metrics on the validation set(3769 instances) of KITTI dataset.

```
|                   | Car@R11 | Pedestrian@R11 | Cyclist@R11  | 
| ----------------- | --------| -------------- | ------------ |
| CUDA-PointPillars | 77.00   | 52.50          | 62.26        |
| OpenPCDet         | 77.28   | 52.29          | 62.68        |
```

## Note

- Voxelization has random output since GPU processes all points simultaneously while points selection for a voxel is random.

## References

- [Detecting Objects in Point Clouds with NVIDIA CUDA-Pointpillars](https://developer.nvidia.com/blog/detecting-objects-in-point-clouds-with-cuda-pointpillars/)
- [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784)
