from typing import Any, Callable, Dict, Set, Tuple
import onnx2torchscript as o2t

import torch
import numpy as np
import onnx
import onnx.backend.test
import tempfile
from onnx.backend.base import Backend, BackendRep

from onnx2torchscript.onnx2torchscript import onnx2ts


def run_op_test(f: Callable, *args, opset_version: int = 11) -> onnx.ModelProto:
    if isinstance(f, torch.nn.Module):
        mod = f
    else:
        class M(torch.nn.Module):
            def forward(self, *args):
                return f(*args)
        mod = M()

    tmp = tempfile.NamedTemporaryFile()
    torch.onnx.export(mod, args, tmp.name, opset_version=opset_version)
    m: onnx.ModelProto = onnx.load_model(tmp.name)
    ts = o2t.onnx2ts(m, args)
    call_func_count = 0
    for n in ts.graph.nodes():
        if n.kind() == "prim::CallFunction":
            call_func_count += 1
    assert call_func_count == len(m.graph.node)
    assert torch.allclose(mod(*args), ts(*args))

    return m


def test_add():
    run_op_test(lambda a, b: a + b, torch.randn(10), torch.randn(10))


def test_gemm():
    run_op_test(
        lambda a, b: torch.mm(a, b),
        torch.randn(10, 10), torch.randn(10, 10))
    run_op_test(
        lambda c, a, b: torch.addmm(c, a, b),
        torch.randn(10, 10), torch.randn(10, 10),
        torch.randn(10, 10))


def test_initializer():
    class M(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.v = torch.nn.parameter.Parameter(torch.rand(10))

        def forward(self, a):
            return a * self.v

    m = run_op_test(M(), torch.rand(10))
    assert len(m.graph.initializer) == 1


_has_mps: bool = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


class TorchScriptBackendRep(BackendRep):
    def __init__(self, model: onnx.ModelProto, device: str):
        super().__init__()
        self.model = model
        if device == "CUDA":
            if _has_mps:
                self.device = "mps"
            else:
                assert torch.backends.cuda.is_avaiable()
                self.device = "cuda"
        else:
            assert device == "CPU"
            self.device = "cpu"

        self.device = torch.device(self.device)

    def run(self, inputs: Any, **kwargs) -> Tuple[Any, ...]:
        ins = []
        for i in inputs:
            if isinstance(i, np.ndarray):
                ins.append(torch.from_numpy(i.copy()))
            elif isinstance(i, list):
                ins.append([torch.from_numpy(j.copy()) for j in i])
            else:
                raise f"Unsupported input: {i}"
        self.ts = o2t.onnx2ts(self.model, ins)
        self.ts.to(device=self.device)
        ins = [i.to(self.device) for i in ins]
        ret = self.ts(*ins)
        if not isinstance(ret, (list, tuple)):
            ret = (ret,)
        return tuple([t.detach().cpu().numpy() for t in ret])
    


_to_torch_dtype: Dict[int, torch.dtype] = {
    onnx.TensorProto.DataType.FLOAT: torch.float,
    onnx.TensorProto.DataType.UINT8: torch.uint8,
    onnx.TensorProto.DataType.INT8: torch.int8,
    onnx.TensorProto.DataType.INT16: torch.int16,
    onnx.TensorProto.DataType.INT32: torch.int32,
    onnx.TensorProto.DataType.INT64: torch.int64,
    onnx.TensorProto.DataType.DOUBLE: torch.double,
    onnx.TensorProto.DataType.BOOL: torch.bool,
}


class TorchScriptBackend(Backend):
    @classmethod
    def prepare(cls, model: onnx.ModelProto, device: str, **kwargs):
        return TorchScriptBackendRep(model, device)

    @classmethod
    def is_compatible(cls, model: onnx.ModelProto, device: str = "CPU", **kwargs: Any) -> bool:
        domain2opset: Dict[str, int] = {}
        for o in model.opset_import:
            domain2opset[o.domain] = o.version

        for n in model.graph.node:
            s = o2t.get_onnx_ts(n.op_type, domain2opset[n.domain], n.domain)
            if s is None:
                # print(n.op_type)
                return False

        return True

    @classmethod
    def supports_device(cls, device: str) -> bool:
        if device == "CPU":
            return True
        elif _has_mps and device == "CUDA":
            return True
        return False


backend_test = onnx.backend.test.runner.Runner(TorchScriptBackend, __name__)
backend_test.xfail("test_operator_non_float_params")
backend_test.xfail("uint16")
backend_test.xfail("uint32")
backend_test.xfail("uint64")
backend_test.xfail("test_div_uint8")
backend_test.xfail("test_reshape_zero")
backend_test.xfail("test_identity_opt")
backend_test.xfail("test_identity_sequence")
backend_test.exclude("test_arg.*_select_last_index")
backend_test.exclude("test_BatchNorm")
backend_test.exclude("test_batchnorm_.*training_mode")
backend_test.exclude("conv_with_autopad_same")
backend_test.exclude("conv_with_strides_and_asymmetric_padding")

if _has_mps:
    backend_test.exclude("test_and.*_cuda")
    backend_test.exclude("test_arg.*_cuda")
    backend_test.exclude("test_det.*_cuda")
    backend_test.exclude("test_not.*_cuda")
    backend_test.exclude("test_or.*_cuda")
    backend_test.exclude("test_greater_equal.*_cuda")
    backend_test.exclude("test_less_equal.*_cuda")
    backend_test.exclude("test_gemm_.*_bias_cuda")
    backend_test.exclude("test_pow_.*_cuda")
    backend_test.exclude("test_round.*_cuda")
    backend_test.exclude("test_xor.*_cuda")
    backend_test.exclude("test_tri[lu]_zero_cuda")
    backend_test.exclude("test_Conv1d_dilated_cuda")
    backend_test.exclude("test_Conv1d_stride_cuda")
    backend_test.exclude("test_Conv3d.*_cuda")
    backend_test.exclude("test_PoissonNLLLLoss_no_reduce_cuda")
    backend_test.exclude("test_operator_add_broadcast_cuda")
    backend_test.exclude("test_operator_add_size1_broadcast_cuda")
    backend_test.exclude("test_operator_add_size1_right_broadcast_cuda")
    backend_test.exclude("test_operator_add_size1_singleton_broadcast_cuda")
    backend_test.exclude("test_operator_addconstant_cuda")
    backend_test.exclude("test_operator_mm_cuda")

globals().update(backend_test.enable_report().test_cases)
