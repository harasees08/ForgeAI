"""Run a small CUDA computation directly through the NVIDIA driver."""

import ctypes as ct
import sys


# One block with 256 threads. Each computes out[i] = left[i] + right[i].
VECTOR_ADD = b"""
.version 8.7
.target sm_120
.address_size 64

.visible .entry vector_add(.param .u64 out, .param .u64 left, .param .u64 right)
{
    .reg .b32 %i, %a, %b, %sum;
    .reg .b64 %offset, %out, %left, %right, %address;
    mov.u32 %i, %tid.x;
    mul.wide.u32 %offset, %i, 4;
    ld.param.u64 %out, [out];
    ld.param.u64 %left, [left];
    ld.param.u64 %right, [right];
    add.u64 %address, %left, %offset;
    ld.global.u32 %a, [%address];
    add.u64 %address, %right, %offset;
    ld.global.u32 %b, [%address];
    add.u32 %sum, %a, %b;
    add.u64 %address, %out, %offset;
    st.global.u32 [%address], %sum;
    ret;
}
"""


def run_driver_check():
    """Verify device execution independently of Python ML packages."""
    if sys.platform == "win32":
        driver = ct.WinDLL("nvcuda.dll")
    elif sys.platform.startswith("linux"):
        driver = ct.CDLL("libcuda.so.1")
    else:
        raise RuntimeError("The driver check requires Windows or Linux.")

    def call(name, *args):
        status = getattr(driver, name)(*args)
        if status:
            error = ct.c_char_p()
            driver.cuGetErrorName(status, ct.byref(error))
            detail = error.value.decode() if error.value else "unknown error"
            raise RuntimeError(f"{name}: {detail} ({status})")

    call("cuInit", 0)
    version = ct.c_int()
    call("cuDriverGetVersion", ct.byref(version))
    device = ct.c_int()
    call("cuDeviceGet", ct.byref(device), 0)
    major, minor = ct.c_int(), ct.c_int()
    call("cuDeviceGetAttribute", ct.byref(major), 75, device)
    call("cuDeviceGetAttribute", ct.byref(minor), 76, device)
    capability = (major.value, minor.value)
    if capability < (12, 0):
        raise RuntimeError(f"The smoke kernel targets sm_120. Found {capability}.")
    name = ct.create_string_buffer(256)
    call("cuDeviceGetName", name, len(name), device)

    context = ct.c_void_p()
    call("cuCtxCreate_v2", ct.byref(context), 0, device)
    try:
        module, kernel = ct.c_void_p(), ct.c_void_p()
        call("cuModuleLoadData", ct.byref(module), ct.c_char_p(VECTOR_ADD))
        call("cuModuleGetFunction", ct.byref(kernel), module, ct.c_char_p(b"vector_add"))

        count = 256
        vector = ct.c_uint32 * count
        left = vector(*range(count))
        right = vector(*(i * i for i in range(count)))
        output = vector()
        size = ct.c_size_t(ct.sizeof(output))
        buffers = [ct.c_uint64() for _ in range(3)]
        for buffer in buffers:
            call("cuMemAlloc_v2", ct.byref(buffer), size)
        out_dev, left_dev, right_dev = buffers
        call("cuMemcpyHtoD_v2", left_dev, left, size)
        call("cuMemcpyHtoD_v2", right_dev, right, size)
        params = (ct.c_void_p * len(buffers))(*(ct.addressof(buf) for buf in buffers))
        call("cuLaunchKernel", kernel, 1, 1, 1, count, 1, 1, 0, None, params, None)
        call("cuCtxSynchronize")
        call("cuMemcpyDtoH_v2", output, out_dev, size)
    finally:
        # Destroying the private context releases its device allocations and module.
        call("cuCtxDestroy_v2", context)

    expected = [a + b for a, b in zip(left, right, strict=True)]
    if list(output) != expected:
        raise RuntimeError("CUDA vector addition differs from the CPU reference.")
    return {
        "gpu": name.value.decode(),
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "driver_cuda_version": f"{version.value // 1000}.{version.value % 1000 // 10}",
        "vector_elements": count,
        "checksum": sum(output),
        "cpu_reference_match": True,
    }
