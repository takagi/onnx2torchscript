from math import ceil
from optparse import Option
from typing import Callable, Dict, List, Optional, Union, Tuple
from torch import Tensor
from onnx2torchscript import onnx_op
import torch
import onnx


binary_ops: Dict[str, Callable] = {
    "Add-1": torch.add,
    "And-1": torch.logical_and,
    "BitwiseAnd-18": torch.bitwise_and,
    "BitwiseOr-18": torch.bitwise_xor,
    "Div-1": torch.true_divide,
    "Equal-1": torch.eq,
    "Greater-1": torch.greater,
    "Less-1": torch.less,
    "MatMul-1": torch.matmul,
    "Mul-1": torch.mul,
    "Or-7": torch.logical_or,
    "Sub-1": torch.sub,
    "Xor-1": torch.logical_xor,
}

for k, o in binary_ops.items():
    op_type, ver = k.split('-')
    @onnx_op(op_type, int(ver))
    def op(x: Tensor, y: Tensor) -> Tensor:
        return o(x, y)
    op.__nmae__ = f"op_{op_type}_{ver}"

unary_ops: Dict[str, Callable] = {
    "Abs-1": torch.abs,
    "Acos-7": torch.acos,
    "Acosh-9": torch.acosh,
    "Asin-7": torch.asin,
    "Asinh-9": torch.asinh,
    "Atan-7": torch.atan,
    "Atanh-9": torch.atanh,
    "Ceil-1": torch.ceil,
    "Cos-7": torch.cos,
    "Cosh-9": torch.cosh,
    "Det-11": torch.det,
    "Exp-1": torch.exp,
    "Floor-1": torch.floor,
    "IsNaN-9": torch.isnan,
    "Log-1": torch.log,
    "Neg-1": torch.neg,
    "Not-1": torch.logical_not,
    "Reciprocal-1": torch.reciprocal,
    "Relu-1": torch.relu,
    "Round-11": torch.round,
    "Sigmoid-1": torch.sigmoid,
    "Sign-9": torch.sign,
    "Sin-7": torch.sin,
    "Sinh-9": torch.sinh,
    "Sqrt-1": torch.sqrt,
    "Tan-7": torch.tan,
    "Tanh-1": torch.tanh,
}

for k, o in unary_ops.items():
    op_type, ver = k.split('-')
    @onnx_op(op_type, int(ver))
    def op(x: Tensor) -> Tensor:
        return o(x)
    op.__nmae__ = f"op_{op_type}_{ver}"


@onnx_op("Gemm", 1)
def op_Gemm(
    a: Tensor, b: Tensor, c: Optional[Tensor] = None,
    # *,  # Commenting out due to kwargs unsupported in trace mode
    alpha: float = 1.0, beta: float = 1.0, transA: int = 0, transB: int = 0
) -> Tensor:
    if transA:
        a = a.swapaxes(-1, -2)
    if transB:
        b = b.swapaxes(-1, -2)
    if c is not None:
        return torch.addmm(c, a, b, beta=beta, alpha=alpha)
    else:
        return torch.mm(a, b) * alpha


@onnx_op("Constant", 1)
def op_Constant(
    # *,
    value: Tensor
) -> Tensor:
    return value


@onnx_op("Conv", 1)
def op_Conv(
    x: Tensor, w: Tensor, b: Optional[Tensor] = None,
    # *,
    dilations: Optional[List[int]] = None,  group: int = 1, kernel_shape: Optional[List[int]] = None,
    pads: Optional[List[int]] = None, strides: Optional[List[int]] = None,
) -> Tensor:
    if dilations is None:
        dilations = [1]
    if strides is None:
        strides = [1]
    if pads is None:
        pads = [0]
    elif all([p == pads[0] for p in pads]):
        pads = [pads[0]]
    return torch.convolution(
        x, w, b,
        stride=strides, padding=pads, dilation=dilations, groups=group, 
        transposed=False, output_padding=[0])


@onnx_op("BatchNormalization", 1)
def op_BatchNorm(
    x: Tensor, scale: Tensor, b: Tensor,
    input_mean: Tensor, input_var: Tensor,
    # *,
    epsilon: float = 1e-05, momentum: float = 0.9, training_mode: int = 0,
) -> Tuple[Tensor, Tensor, Tensor]:
    return torch.native_batch_norm(
        x, scale, b, input_mean, input_var, 
        training=training_mode != 0, momentum=momentum, eps=epsilon)


@onnx_op("Softmax", 1)
def op_Softmax(
    x: Tensor,
    # *,
    axis: int = -1
) -> Tensor:
    return torch.softmax(x, dim=axis)


@onnx_op("LogSoftmax", 1)
def op_LogSoftmax(
    x: Tensor,
    # *,
    axis: int = -1
) -> Tensor:
    return torch.log_softmax(x, dim=axis)


@onnx_op("Trilu", 14)
def op_Trilu(
    input: Tensor,
    k: Optional[Tensor] = None,
    # *,
    upper: int = 1
) -> Tensor:
    if k is None:
        k = torch.scalar_tensor(0)
    if upper:
        return torch.triu(input, diagonal=k.item())
    else:
        return torch.tril(input, diagonal=k.item())


@onnx_op("Where", 9)
def op_Where(cond: Tensor, x: Tensor, y: Tensor) -> Tensor:
    return torch.where(cond != 0, x, y)


@onnx_op("TopK", 1)
def op_TopK(
    x: Tensor, k: Tensor,
    # *,
    axis: int = -1, largest: int = 1, sorted: int = 1,
) -> Tuple[Tensor, Tensor]:
    return torch.topk(x, k.item(), dim=axis, largest=largest != 0, sorted=sorted != 0)


OnnxAny = Union[Tensor, List[Tensor], Optional[Tensor]]

@onnx_op("Identity", 1)
def op_Identity(input: OnnxAny) -> OnnxAny:
    return input


@onnx_op("Reshape", 1)
def op_Reshape(
    data: Tensor, shape: Tensor,
    # *,
    allowzero: int = 0
) -> Tensor:
    return torch.reshape(data, torch.jit.annotate(List[int], shape.tolist()))


@onnx_op("BitShift", 11)
def op_BitShift(
    x: Tensor, y: Tensor,
    # *,
    direction: str,
) -> Tensor:
    if direction == "LEFT":
        return torch.bitwise_left_shift(x, y)
    else:
        assert direction == "RIGHT"
        return torch.bitwise_right_shift(x, y)


@onnx_op("Shape", 1)
def op_Shape(
    data: Tensor,
    # *,
    end: Optional[int] = None, start: int = 0,
) -> Tensor:
    s = data.shape
    if end is None:
        end = len(s)
    return torch.tensor(s[start:end])


@onnx_op("Transpose", 1)
def op_Transpose(
    data: Tensor,
    # *,
    perm: Optional[List[int]] = None,
) -> Tensor:
    if perm is None:
        l = list(range(data.dim()))
        l.reverse()
        return data.permute(l)
    return torch.permute(data, perm)


@onnx_op("Tile", 1)
def op_Tile(input: Tensor, repeats: Tensor) -> Tensor:
    return torch.tile(input, torch.jit.annotate(List[int], repeats.tolist()))


@onnx_op("Pow", 1)
def op_Pow(x: Tensor, y: Tensor) -> Tensor:
    return torch.pow(x, y).to(x.dtype)


@onnx_op("ArgMax", 1)
def op_ArgMax(
    data: Tensor,
    # *,
    axis: int = 0, keepdims: int = 1, select_last_index: int = 0,
) -> Tensor:
    return torch.argmax(data, dim=axis, keepdim=keepdims != 0)


@onnx_op("ArgMin", 1)
def op_ArgMin(
    data: Tensor,
    # *,
    axis: int = 0, keepdims: int = 1, select_last_index: int = 0,
) -> Tensor:
    return torch.argmin(data, dim=axis, keepdim=keepdims != 0)


@onnx_op("Size", 1)
def op_Size(data: Tensor) -> Tensor:
    return torch.scalar_tensor(data.numel(), dtype=torch.int64)


@onnx_op("Einsum", 12)
def op_Einsum(
    inputs: List[Tensor],
    # *,
    equation: str,
) -> Tensor:
    return torch.einsum(equation, inputs)


@onnx_op("Max", 1)
def op_Max(inputs: List[Tensor]) -> Tensor:
    ret = inputs[0]
    for i in inputs[1:]:
        ret = torch.maximum(ret, i)
    return ret


@onnx_op("Min", 1)
def op_Min(inputs: List[Tensor]) -> Tensor:
    ret = inputs[0]
    for i in inputs[1:]:
        ret = torch.minimum(ret, i)
    return ret


@onnx_op("Mean", 1)
def op_Mean(inputs: List[Tensor]) -> Tensor:
    ret = inputs[0]
    for i in inputs[1:]:
        ret = ret + i
    return ret / len(inputs)


@onnx_op("Sum", 1)
def op_Sum(inputs: List[Tensor]) -> Tensor:
    ret = inputs[0]
    for i in inputs[1:]:
        ret = ret + i
    return ret


@onnx_op("Mod", 1)
def op_Mod(
    a: Tensor, b: Tensor,
    # *,
    fmod: int = 0,
) -> Tensor:
    if fmod:
        return torch.fmod(a, b)
    else:
        return a % b


@onnx_op("PRelu", 1)
def op_PRelu(x: Tensor, slope: Tensor) -> Tensor:
    return torch.prelu(x, slope)


@onnx_op("SequenceLength", 11)
def op_SequenceLenght(x: List[Tensor]) -> Tensor:
    return torch.scalar_tensor(len(x), dtype=torch.int64)


@onnx_op("ReduceMax", 1)
def op_ReduceMax(
    data: Tensor,
    # *,
    axes: Optional[List[int]] = None,
    keepdims: int = 1,
) -> Tensor:
    if axes is None:
        ret = torch.max(data)
        if keepdims != 0:
            return torch.reshape(ret, [1] * data.dim())
        else:
            return ret
    else:
        return torch.amax(data, dim=axes, keepdim=keepdims != 0)


@onnx_op("ReduceMin", 1)
def op_ReduceMin(
    data: Tensor,
    # *,
    axes: Optional[List[int]] = None,
    keepdims: int = 1,
) -> Tensor:
    if axes is None:
        ret = torch.min(data)
        if keepdims != 0:
            return torch.reshape(ret, [1] * data.dim())
        else:
            return ret
    else:
        return torch.amin(data, dim=axes, keepdim=keepdims != 0)


@onnx_op("ReduceL1", 1)
def op_ReduceL1(
    data: Tensor,
    # *,
    axes: Optional[List[int]] = None,
    keepdims: int = 1,
) -> Tensor:
    data = torch.abs(data)
    if axes is None:
        ret = torch.sum(data)
        if keepdims != 0:
            return torch.reshape(ret, [1] * data.dim())
        else:
            return ret
    else:
        return torch.sum(data, dim=axes, keepdim=keepdims != 0)


@onnx_op("ReduceL2", 1)
def op_ReduceL2(
    data: Tensor,
    # *,
    axes: Optional[List[int]] = None,
    keepdims: int = 1,
) -> Tensor:
    data = data * data
    if axes is None:
        ret = torch.sqrt(torch.sum(data))
        if keepdims != 0:
            return torch.reshape(ret, [1] * data.dim())
        else:
            return ret
    else:
        return torch.sqrt(torch.sum(data, dim=axes, keepdim=keepdims != 0))


@onnx_op("ReduceSum", 1)
def op_ReduceSum_1(
    data: Tensor,
    # *,
    axes: Optional[List[int]] = None,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
) -> Tensor:
    if axes is None:
        ret = torch.sum(data)
        if keepdims != 0:
            return torch.reshape(ret, [1] * data.dim())
        else:
            return ret
    else:
        return torch.sum(data, dim=axes, keepdim=keepdims != 0)


@onnx_op("ReduceSum", 13)
def op_ReduceSum_13(
    data: Tensor,
    axes: Optional[Tensor] = None,
    # *,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
) -> Tensor:
    if axes is None:
        ret = torch.sum(data)
        if keepdims != 0:
            return torch.reshape(ret, [1] * data.dim())
        else:
            return ret
    else:
        if axes.numel() == 0 and noop_with_empty_axes != 0:
            return data
        return torch.sum(data, dim=torch.jit.annotate(List[int], axes.tolist()), keepdim=keepdims != 0)


@onnx_op("ReduceSumSquare", 1)
def op_ReduceSumSquare(
    data: Tensor,
    # *,
    axes: Optional[List[int]] = None,
    keepdims: int = 1,
) -> Tensor:
    data = data * data
    if axes is None:
        ret = torch.sum(data)
        if keepdims != 0:
            return torch.reshape(ret, [1] * data.dim())
        else:
            return ret
    else:
        return torch.sum(data, dim=axes, keepdim=keepdims != 0)


@onnx_op("ReduceMean", 1)
def op_ReduceMean(
    data: Tensor,
    # *,
    axes: Optional[List[int]] = None,
    keepdims: int = 1,
) -> Tensor:
    if axes is None:
        ret = torch.mean(data)
        if keepdims != 0:
            return torch.reshape(ret, [1] * data.dim())
        else:
            return ret
    else:
        return torch.mean(data, dim=axes, keepdim=keepdims != 0)


@onnx_op("ReduceProd", 1)
def op_ReduceProd(
    data: Tensor,
    # *,
    axes: Optional[List[int]] = None,
    keepdims: int = 1,
) -> Tensor:
    if axes is None:
        ret = torch.prod(data)
        if keepdims != 0:
            return torch.reshape(ret, [1] * data.dim())
        else:
            return ret
    else:
        assert len(axes) == 1
        return torch.prod(data, dim=axes[0], keepdim=keepdims != 0)


@onnx_op("Clip", 1)
def op_Clip(input: Tensor, min: Optional[Tensor] = None, max: Optional[Tensor] = None) -> Tensor:
    if min is not None and max is not None:
        return torch.clamp(input, min, max)
    elif min is not None:
        return torch.clamp_min(input, min)
    elif max is not None:
        return torch.clamp_max(input, max)
    return input


@onnx_op("Concat", 1)
def op_Concat(
    inputs: List[Tensor],
    # *,
    axis: int,
) -> Tensor:
    return torch.concat(inputs, dim=axis)


@onnx_op("Unsqueeze", 13)
def op_Unsqueeze_13(data: Tensor, axes: Tensor) -> Tensor:
    axes_list = torch.jit.annotate(List[int], axes.tolist())
    axes_list = [a if a >= 0 else a + data.dim() for a in axes_list]
    ret = data
    for a in sorted(axes_list):
        ret = torch.unsqueeze(ret, dim=a)
    return ret


@onnx_op("Unsqueeze", 1)
def op_Unsqueeze_1(
    data: Tensor,
    # *,
    axes: List[int],
) -> Tensor:
    axes_list = [a if a >= 0 else a + data.dim() for a in axes]
    ret = data
    for a in sorted(axes_list):
        ret = torch.unsqueeze(ret, dim=a)
    return ret


@onnx_op("Cast", 1)
def op_Cast(
    input: Tensor,
    # *,
    to: int,
) -> Tensor:
    _to_torch_dtype: Dict[int, torch.dtype] = {
        1: torch.float,
        2: torch.uint8,
        3: torch.int8,
        5: torch.int16,
        6: torch.int32,
        7: torch.int64,
        11: torch.double,
        9: torch.bool,
    }
    return  input.to(_to_torch_dtype[to])


@onnx_op("Compress", 9)
def op_Compress(
    input: Tensor, condition: Tensor,
    # *,
    axis: Optional[int] = None
) -> Tensor:
    if axis is None:
        input = torch.flatten(input)
        axis = 0
    return torch.index_select(input, dim=axis, index=torch.nonzero(condition).squeeze())


@onnx_op("GlobalMaxPool", 1)
def op_GlobalMaxPool(x: Tensor) -> Tensor:
    dims = torch.jit.annotate(List[int], [])
    for i in range(x.dim() - 2):
        dims.append(i + 2)
    return torch.amax(x, dim=dims, keepdim=True)


@onnx_op("GlobalAveragePool", 1)
def op_GlobalAveragePool(x: Tensor) -> Tensor:
    dims = torch.jit.annotate(List[int], [])
    for i in range(x.dim() - 2):
        dims.append(i + 2)
    return torch.mean(x, dim=dims, keepdim=True)


@onnx_op("IsInf", 10)
def op_IsInf(
    x: Tensor,
    # *,
    detect_negative: int = 1, detect_positive: int = 1,
) -> Tensor:
    ret = torch.isinf(x)
    if detect_positive != 0 and detect_negative == 0:
        return torch.logical_and((x > 0), ret)
    if detect_positive == 0 and detect_negative != 0:
        return torch.logical_and((x < 0), ret)
    return ret


@onnx_op("GridSample, 16")
def op_GridSample(
    x: Tensor, grid: Tensor,
    # *,
    align_corners: int = 0, mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> Tensor:
    # ref: aten/src/ATen/native/GridSamplerUtils.h
    i = 0
    if mode == "bilinear":
        i = 0
    elif mode == "nearest":
        i = 1
    else:
        assert mode == "bicubic"
        i = 2
    p = 0
    if padding_mode == "zeros":
        p = 0
    elif padding_mode == "border":
        p = 1
    else:
        assert padding_mode == "reflection"
        p = 2
    return torch.grid_sampler(
        x, grid,
        interpolation_mode=i, padding_mode=p,
        align_corners=align_corners != 0)


@onnx_op("AveragePool", 1)
def op_AveragePool(
    x: Tensor,
    # *,
    ceil_mode: int = 0, count_include_pad: int = 0,
    kernel_shape: Optional[List[int]] = None,
    pads: Optional[List[int]] = None,
    strides: Optional[List[int]] = None,
) -> Tensor:
    assert kernel_shape is not None
    if strides is None:
        strides = [1]
    if pads is None:
        pads = [0]
    else:
        for i in range(len(kernel_shape)):
            assert pads[i] ==  pads[i + len(kernel_shape)]
        pads = pads[0:len(kernel_shape)]

    if len(kernel_shape) == 1:
        return torch.nn.functional.avg_pool1d(
            x, kernel_shape, stride=strides, padding=pads,
            count_include_pad=count_include_pad != 0,
            ceil_mode=ceil_mode != 0)
    elif len(kernel_shape) == 2:
        return torch.nn.functional.avg_pool2d(
            x, kernel_shape, stride=strides, padding=pads,
            count_include_pad=count_include_pad != 0,
            ceil_mode=ceil_mode != 0)
    else:
        assert len(kernel_shape) == 3
        return torch.nn.functional.avg_pool3d(
            x, kernel_shape, stride=strides, padding=pads,
            count_include_pad=count_include_pad != 0,
            ceil_mode=ceil_mode != 0)


@onnx_op("MaxPool", 1)
def op_MaxPool(
    x: Tensor,
    # *,
    ceil_mode: int = 0, count_include_pad: int = 0,
    dilations: Optional[List[int]] = None,
    kernel_shape: Optional[List[int]] = None,
    pads: Optional[List[int]] = None,
    strides: Optional[List[int]] = None,
) -> Tuple[Tensor, Tensor]:
    assert kernel_shape is not None
    if strides is None:
        strides = [1]
    if pads is None:
        pads = [0]
    else:
        for i in range(len(kernel_shape)):
            assert pads[i] ==  pads[i + len(kernel_shape)]
        pads = pads[0:len(kernel_shape)]
    if dilations is None:
        dilations = [1]

    if len(kernel_shape) == 1:
        return torch.nn.functional.max_pool1d(
            x, kernel_shape, stride=strides, padding=pads,
            dilation=dilations,
            return_indices=True,
            ceil_mode=ceil_mode != 0)
    elif len(kernel_shape) == 2:
        return torch.nn.functional.max_pool2d(
            x, kernel_shape, stride=strides, padding=pads,
            dilation=dilations,
            return_indices=True,
            ceil_mode=ceil_mode != 0)
    else:
        assert len(kernel_shape) == 3
        return torch.nn.functional.max_pool3d(
            x, kernel_shape, stride=strides, padding=0,
            dilation=dilations,
            return_indices=True,
            ceil_mode=ceil_mode != 0)
