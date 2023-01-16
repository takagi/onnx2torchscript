import onnx2torchscript as o2ts
import torch

import argparse
import sys
import time
from typing import List

def main(argv: List[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('test_dir')
    parser.add_argument('--enable_optimization', action='store_true', help='Enable JIT optimization')
    parser.add_argument('--use_custom_stream', action='store_true', help='Use a custom CUDA stream')
    parser.add_argument('--print_code', action='store_true', help='Print code after warming up')
    parser.add_argument('--print_graph', action='store_true', help='Print graph after warming up')
    args = parser.parse_args()

    if args.enable_optimization:
        print('Optimization enabled')
    else:
        torch._C._jit_set_profiling_mode(False)  # disable optimization
        print('Optimization disabled')

    if args.use_custom_stream:
        stream = torch.cuda.Stream()
        torch.cuda.set_stream(stream)

    print('Convert ONNX to TorchScript...')
    int_tensors_on_gpu = [
        'Constant_0_gpu_model_1991_gpu_value',
        'Constant_1_gpu_model_1995_gpu_value',
        'Constant_2_model_1996_value',
        'Constant_3_gpu_model_1997_gpu_value',
        'Constant_7_gpu_model_2005_gpu_value',
    ]  # these are int tensors, but must be located on GPU
    ts, datas = o2ts.onnx_testdir_to_torchscript(args.test_dir)
    for k, v in ts.state_dict().items():
        if v.dtype == torch.int64:
            if k in int_tensors_on_gpu:
                v = v.to('cuda')
                setattr(ts, k, v)
        else:
            setattr(ts, k, v.to('cuda'))

    for inputs, outputs in datas:
        for i in range(len(inputs)):
            inputs[i] = inputs[i].to("cuda")

        # Print code
        if args.print_code:
            print(ts.code)

        # Warm up
        print('Warming up...')
        for i in range(5):
            print(f'loop {i+1}:')
            torch.cuda.nvtx.range_push('Warming up')
            actual_outs = ts(*inputs)
            torch.cuda.nvtx.range_pop()

        # Print graph
        if args.print_graph:
            print(ts.graph_for(*inputs))

        # Print code
        if args.print_code:
            print(ts.code)

        # Benchmark
        def bench():
            start = time.time()
            n = 0
            while True:
                #torch.cuda.nvtx.range_push('Iter')
                ts(*inputs)
                #torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize()
                n += 1
                elapsed = time.time() - start
                if elapsed > 1.0:
                    break
            return elapsed / n * 1000  # msec
        print('Run benchmark...')
        print(f'bench: {bench()}[msec]')

        # Verify outputs
        print('Verify outputs...')
        if not isinstance(actual_outs, (list, tuple)):
            actual_outs = (actual_outs,)
        assert len(actual_outs) == len(outputs)

        for e_o, a_o in zip(outputs, actual_outs):
            a_o = a_o.to('cpu')
            assert torch.allclose(e_o, a_o)


if __name__ == "__main__":
    main(sys.argv)
