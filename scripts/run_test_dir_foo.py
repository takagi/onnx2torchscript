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
        #'Constant_0_model_1991_value',
        'Constant_0_gpu_model_1991_gpu_value',
        #'Constant_1_model_1995_value',
        'Constant_1_gpu_model_1995_gpu_value',
        'Constant_2_model_1996_value',
        #'Constant_3_model_1997_value',
        'Constant_3_gpu_model_1997_gpu_value',
        #'Constant_4_model_2002_value',
        #'Constant_5_model_2003_value',
        #'Constant_7_model_2005_value',
        'Constant_7_gpu_model_2005_gpu_value',
        #'Constant_26_model_2032_value',
        #'Constant_29_model_2036_value',

        # 'ConstantOfShape_79_model_2086_value',
        # 'Constant_80_5302_value',
        # 'ConstantOfShape_98_model_2105_value',
        # 'Constant_99_5303_value',

        #'Constant_136_model_2143_value',
        #'Constant_165_model_2174_value',
        # 'Constant_168_model_2178_value',
        # 'Constant_172_model_2183_value',
        # 'Constant_175_model_2187_value',
        # 'ConstantOfShape_185_model_2196_value',
        # 'Constant_186_5310_value',
        # 'Constant_195_model_2208_value',
        # 'Constant_198_model_2212_value',
        # 'Constant_201_model_2216_value',
        # 'ConstantOfShape_213_model_2227_value',
        # 'Constant_214_5312_value',
        # 'Constant_223_model_2239_value',
        # 'Constant_226_model_2243_value',
        # 'Constant_229_model_2247_value',
        # 'Constant_232_model_2251_value',
        # 'ConstantOfShape_246_model_2264_value',
        # 'Constant_247_5314_value',
        # 'Constant_256_model_2276_value',
        # 'Constant_259_model_2280_value',
        # 'ConstantOfShape_269_model_2289_value',
        # 'Constant_270_5316_value',
        # 'ConstantOfShape_297_model_2320_value',
        # 'Constant_298_5318_value',
        # 'ConstantOfShape_320_model_2345_value',
        # 'Constant_321_5320_value',
        # 'Constant_505_model_2547_value',
        # 'Constant_514_model_2557_value',
        # 'Constant_517_model_2561_value',
        # 'ConstantOfShape_525_model_2569_value',
        # 'Constant_526_5331_value',
        # 'Constant_534_model_2579_value',
        # 'Constant_543_model_2589_value',
        # 'Constant_546_model_2593_value',
        # 'ConstantOfShape_554_model_2601_value',
        # 'Constant_555_5332_value',
        # 'Constant_567_model_2615_value',
        # 'Constant_578_model_2627_value',
        # 'Constant_581_model_2631_value',
        # 'Constant_584_model_2635_value',
        # 'ConstantOfShape_594_model_2645_value',
        # 'Constant_595_5333_value',
        # 'Constant_603_model_2655_value',
        # 'Constant_614_model_2667_value',
        # 'Constant_617_model_2671_value',
        # 'Constant_620_model_2675_value',
        # 'ConstantOfShape_630_model_2685_value',
        # 'Constant_631_5334_value',
        # 'Constant_652_model_2708_value',
        # 'Constant_661_model_2718_value',
        # 'Constant_664_model_2722_value',
        # 'ConstantOfShape_672_model_2730_value',
        # 'Constant_673_5335_value',
        # 'Constant_681_model_2740_value',
        # 'Constant_690_model_2750_value',
        # 'Constant_693_model_2754_value',
        # 'ConstantOfShape_701_model_2762_value',
        # 'Constant_702_5336_value',
        # 'Constant_791_model_2868_value',
        # 'Constant_794_model_2872_value',
        # 'ConstantOfShape_804_model_2881_value',
        # 'Constant_805_5338_value',
        # 'Constant_814_model_2893_value',
        # 'Constant_817_model_2897_value',
        # 'Constant_820_model_2901_value',
        # 'ConstantOfShape_832_model_2912_value',
        # 'Constant_833_5340_value',
        # 'Constant_841_model_2923_value',
        # 'ConstantOfShape_855_model_2937_value',
        # 'Constant_856_5341_value',
        # 'ConstantOfShape_855_model_2937_value',
        # 'Constant_856_5341_value',
        # 'ConstantOfShape_879_model_2961_value',
        # 'Constant_880_5342_value',

        #'Constant_1086_model_3187_value',
        #'Constant_1095_model_3197_value',
        #'Constant_1098_model_3201_value',
        #'ConstantOfShape_1106_model_3209_value',
        #'Constant_1107_5353_value',
        #'Constant_1115_model_3219_value',
        #'Constant_1124_model_3229_value',
        #'Constant_1127_model_3233_value',
        #'ConstantOfShape_1135_model_3241_value',
        #'Constant_1136_5354_value',
        #'Constant_1148_model_3255_value',
        #'Constant_1159_model_3267_value',
        #'Constant_1162_model_3271_value',
        #'Constant_1165_model_3275_value',
        #'ConstantOfShape_1175_model_3285_value',
        #'Constant_1176_5355_value',
        #'Constant_2775_model_5056_value',
        #'ConstantOfShape_2789_model_5070_value',
        #'Constant_2790_5411_value',
        #'Constant_2820_model_5107_value',
        #'Constant_2829_model_5117_value',
        #'Constant_2838_model_5127_value',
        #'Constant_2841_model_5131_value',
        #'ConstantOfShape_2849_model_5139_value',
        #'Constant_2850_5416_value',
        #'Constant_2858_model_5149_value',
        #'Constant_2867_model_5159_value',
        #'Constant_2870_model_5163_value',
        #'ConstantOfShape_2878_model_5171_value',
        #'Constant_2879_5417_value',
        #'Constant_2960_model_5257_value',
        #'Constant_2969_model_5267_value',
        #'Constant_2978_model_5277_value',
        #'Constant_2981_model_5281_value',
        #'ConstantOfShape_2989_model_5289_value',
        #'Constant_2990_5427_value',
    ]  # these are int tensors, but must be located on GPU
    ts, datas = o2ts.onnx_testdir_to_torchscript(args.test_dir)
    for k, v in ts.state_dict().items():
        if v.dtype == torch.int64:
            print((k, v.data_ptr(), v.dtype, v.shape, v.device))
            # if k == 'Constant_0_model_1991_value':
            #     k = 'Constant_0_model_1991_value_gpu'
            #     v = v.to('cuda')
            #     setattr(ts, k, v.to('cuda'))
            #     print((k, v.data_ptr(), v.dtype, v.shape, v.device))
            # el
            if k in int_tensors_on_gpu:
                v = v.to('cuda')
                setattr(ts, k, v)
                print((k, v.data_ptr(), v.dtype, v.shape, v.device))
                #breakpoint()
        else:
            setattr(ts, k, v.to('cuda'))
    #breakpoint()
    #ts.to('cuda')


    # for k, v in ts.state_dict().items():
    #     assert getattr(v, '__name__', None) is None
    #     setattr(v, '__name__', k)
    #     print((k, v.data_ptr()))


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
                torch.cuda.nvtx.range_push('Iter')
                ts(*inputs)
                torch.cuda.nvtx.range_pop()
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
