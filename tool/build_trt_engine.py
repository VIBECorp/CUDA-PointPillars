#!/usr/bin/env python3
"""Build a TensorRT 10 FP16 engine for the rail pointpillar_ros runtime."""
import argparse
import ctypes
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('onnx', type=Path, nargs='?', default=ROOT / 'model/pointpillar.onnx',
                        help='ONNX path (default: repository model/pointpillar.onnx)')
    parser.add_argument('-o', '--output', type=Path, help='output .plan (default: beside ONNX)')
    parser.add_argument('--plugin', type=Path, default=ROOT / 'build/libpointpillar_core.so')
    parser.add_argument('--workspace-mib', type=int, default=2048)
    parser.add_argument('--force', action='store_true', help='replace existing plan/grid after successful build')
    parser.add_argument('--check-only', action='store_true', help='parse and validate without building or writing')
    args = parser.parse_args()
    source = args.onnx.expanduser().resolve()
    output = (args.output.expanduser().resolve() if args.output else source.with_suffix('.plan'))
    plugin = args.plugin.expanduser().resolve()
    grid_path = Path(str(output) + '.grid')
    if not source.is_file() or source.suffix.lower() != '.onnx':
        parser.error(f'ONNX file not found or invalid extension: {source}')
    if output.suffix != '.plan' or output == source:
        parser.error('output must be a .plan path distinct from the ONNX input')
    if not plugin.is_file():
        parser.error(f'build the plugin first: {plugin}')
    if args.workspace_mib <= 0:
        parser.error('--workspace-mib must be positive')
    if not args.check_only and not args.force and (output.exists() or grid_path.exists()):
        parser.error(f'output already exists: {output} / {grid_path}; use -o or --force')

    import tensorrt as trt
    if int(trt.__version__.split('.')[0]) != 10:
        raise RuntimeError(f'TensorRT 10.x required; found {trt.__version__}')
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, '')
    plugin_handle = ctypes.CDLL(str(plugin), mode=ctypes.RTLD_GLOBAL)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    onnx_parser = trt.OnnxParser(network, logger)
    if not onnx_parser.parse_from_file(str(source)):
        raise RuntimeError('\n'.join(str(onnx_parser.get_error(i)) for i in range(onnx_parser.num_errors)))

    # The C++ runtime passes buffers in this order and assumes fixed rail shapes.
    expected_inputs = [('voxels', (75000, 32, 10), trt.float32),
                       ('voxel_idxs', (75000, 4), trt.int32),
                       ('voxel_num', (1,), trt.int32)]
    if network.num_inputs != 3 or network.num_outputs != 3:
        raise RuntimeError('expected three rail model inputs and three outputs')
    for i, (name, shape, dtype) in enumerate(expected_inputs):
        tensor = network.get_input(i)
        if (tensor.name, tuple(tensor.shape), tensor.dtype) != (name, shape, dtype):
            raise RuntimeError(f'incompatible input {tensor.name}: {tuple(tensor.shape)} {tensor.dtype}; '
                               f'expected {name} {shape} {dtype}')
    shape = tuple(network.get_output(0).shape)
    if len(shape) != 4 or shape[1] != shape[2] or shape[1] not in (160, 320):
        raise RuntimeError(f'unsupported rail output grid: {shape}')
    grid = shape[1] * 2
    for i, (name, channels) in enumerate([('cls_preds', 4), ('box_preds', 28), ('dir_cls_preds', 8)]):
        tensor = network.get_output(i)
        expected = (1, grid // 2, grid // 2, channels)
        if tensor.name != name or tuple(tensor.shape) != expected or tensor.dtype != trt.float32:
            raise RuntimeError(f'incompatible output {tensor.name}: {tuple(tensor.shape)} {tensor.dtype}')
        tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
    for i in range(3):
        network.get_input(i).allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
    network.get_input(0).dtype = trt.float16  # runtime voxelizer supplies half buffers
    print(f'TensorRT {trt.__version__}: {source}\nFP16 engine -> {output}\ngrid -> {grid}', flush=True)
    if args.check_only:
        print('Model interface check passed. No engine was built.')
        return

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mib << 20)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError('TensorRT engine build failed; existing output was not changed')
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError('built engine could not be deserialized')
    expected_types = {'voxels': trt.float16, 'voxel_idxs': trt.int32, 'voxel_num': trt.int32,
                      'cls_preds': trt.float32, 'box_preds': trt.float32, 'dir_cls_preds': trt.float32}
    for name, dtype in expected_types.items():
        if engine.get_tensor_dtype(name) != dtype or engine.get_tensor_format(name) != trt.TensorFormat.LINEAR:
            raise RuntimeError(f'engine I/O format incompatible with runtime: {name}')

    output.parent.mkdir(parents=True, exist_ok=True)
    # Stage both outputs before replacing files. Never expose a partially written engine.
    staged = []
    try:
        for target, data in [(output, bytes(serialized)), (grid_path, f'{grid}\n'.encode())]:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
                staged.append((Path(handle.name), target))
                handle.write(data)
        for temporary, target in staged:
            if args.force:
                os.replace(temporary, target)
            else:
                os.link(temporary, target)  # no accidental overwrite, even after a long build
                temporary.unlink()
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
    print(f'Saved {output} ({output.stat().st_size:,} bytes) and {grid_path}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, ImportError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
