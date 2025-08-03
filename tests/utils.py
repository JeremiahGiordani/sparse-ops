def dump_array_info(name, A):
    print(f"=== {name} ===")
    print("  type:",        type(A))
    print("  dtype:",       A.dtype)
    print("  shape:",       A.shape)
    print("  ndim:",        A.ndim)
    print("  size:",        A.size)
    print("  nbytes:",      A.nbytes)
    print("  flags:")
    for flag in ["C_CONTIGUOUS","F_CONTIGUOUS","OWNDATA","WRITEABLE","ALIGNED"]:
        print(f"    {flag:12s} =", A.flags[flag])
    print("  strides:",     A.strides)
    # raw data pointer & alignment
    ptr = A.__array_interface__["data"][0]
    print("  data ptr:",    hex(ptr))
    print("  ptr % 64 =",    ptr % 64)
    # check each row’s start address mod 64
    if A.ndim == 2:
        R, C = A.shape
        print("  row-start %64:", [((ptr + i * A.strides[0]) % 64) for i in range(min(R,5))], "…")
    # base object (None = owns own data, else a view)
    print("  base:",        A.base)
    print()

# assume x_np and x_output exist:
dump_array_info("x_np     ", x_np)
dump_array_info("x_output ", x_output)
