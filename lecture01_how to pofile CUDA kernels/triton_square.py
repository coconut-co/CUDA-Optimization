import triton    
import triton.language as tl
import torch

# Triton 允许开发者用类似 Python 的代码编写 GPU 内核, 使用 Just-In-Time (JIT) 编译器将 Triton 代码转换为 GPU 可执行代码
# 装饰器，将函数标记为内核函数
@triton.jit
def square_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)              # 相当于 blockIdx.x， 每个 block 块负责一行
    row_start_ptr = input_ptr + row_idx * input_row_stride      # 定位当前线程块需要处理的矩阵行
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    
    # 加载数据
    # 将行加载到SRAM中，由于BLOCK_SIZE可能大于n_cols，因此使用掩码
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))

    # 计算该行元素的平方
    square_output = row * row
    
    # 将结果写回DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, square_output, mask=col_offsets < n_cols)

def square(x):
    n_rows, n_cols = x.shape
    # 块大小设置为大于输入矩阵列数的最小2的幂
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # 通过增加每行分配的warp数量，提高并行处理效率
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    # 分配输出张量
    y = torch.empty_like(x)
    # 提交内核执行，每个输入矩阵的行对应一个内核实例
    square_kernel[(n_rows, )](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y

torch.manual_seed(0)
x = torch.randn(1823, 781, device='cuda')
y_triton = square(x)
y_torch = torch.square(x)
torch.allclose(y_triton, y_torch), (y_triton, y_torch)
#for i in range(1):
#    print(y_torch[i])
#    print(y_triton[i])

