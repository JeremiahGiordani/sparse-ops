import tvm
from tvm import te
import numpy as np

def tvm_setup(weight_np, input_np, bias_np):
    M, N = weight_np.shape

    # Step 1: Placeholders
    A = te.placeholder((M, N), name='A')
    x = te.placeholder((N,), name='x')
    b = te.placeholder((M,), name='b')
    k = te.reduce_axis((0, N), name='k')

    # Step 2: matvec without bias
    matvec = te.compute(
        (M,),
        lambda i: te.sum(A[i, k] * x[k], axis=k),
        name='matvec'
    )

    # Step 3: add bias in a second compute
    y = te.compute(
        (M,),
        lambda i: matvec[i] + b[i],
        name='y'
    )

    # Step 4: Schedule and build
    s = te.create_schedule(y.op)
    func = tvm.build(s, [A, x, b, y], target="llvm")

    # Step 5: Create TVM NDArrays
    dev = tvm.cpu()
    A_tvm = tvm.nd.array(weight_np, dev)
    x_tvm = tvm.nd.array(input_np, dev)
    b_tvm = tvm.nd.array(bias_np, dev)
    y_tvm = tvm.nd.array(np.zeros((M,), dtype=np.float32), dev)

    return func, A_tvm, x_tvm, b_tvm, y_tvm

