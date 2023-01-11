import onnx2torchscript as o2ts
import torch

import os
import sys
from typing import List


def main(argv: List[str]) -> None:
    assert len(argv) == 2

    test_dir = argv[1]

    ts, datas = o2ts.onnx_testdir_to_torchscript(test_dir)
#    ts.to('cuda')
    with open(os.path.join(test_dir, "model.pt"), "wb") as f:
        torch.jit.save(ts, f)
    for inputs, outputs in datas:
#        for i in range(len(inputs)):
#            inputs[i] = inputs[i].to("cuda")
        actual_outs = ts(*inputs)
        if not isinstance(actual_outs, (list, tuple)):
            actual_outs = (actual_outs,)
        assert len(actual_outs) == len(outputs)

        # import onnx
        # m = o2ts.OnnxModule(onnx.load_model(argv[1] + '/model.onnx'))
        # actual_outs = m(*inputs)
        # if not isinstance(actual_outs, (list, tuple)):
        #     actual_outs = (actual_outs,)
        # assert len(actual_outs) == len(outputs)

        for e_o, a_o in zip(outputs, actual_outs):
            a_o = a_o.to('cpu')
            d = (e_o - a_o).abs() / a_o.abs()
            foo = (d > 1e-2).sum()
            print(a_o.shape)
            print(foo)
            print(d)
            print((e_o[-1], a_o[-1], (e_o-a_o)[-1]))
            #print((d.max(), d.min(), d.mean()))

            

            print('max {:.3e}, min {:.3e}, absmin {:.3e}, mean {:.3e}, absmean {:.3e}, std {:.3e}'.format(
                e_o.max().item(), e_o.min().item(), e_o.abs().min().item(),
                e_o.mean().item(), e_o.abs().mean().item(), e_o.std().item()))
            print('max {:.3e}, min {:.3e}, absmin {:.3e}, mean {:.3e}, absmean {:.3e}, std {:.3e}'.format(
                a_o.max().item(), a_o.min().item(), a_o.abs().min().item(),
                a_o.mean().item(), a_o.abs().mean().item(), a_o.std().item()))

            #assert torch.allclose(e_o, a_o)


if __name__ == "__main__":
    main(sys.argv)
