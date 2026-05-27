import sys
sys.argv += ["build_ext", "--build-lib", "./build"]
import os
os.environ["DISTUTILS_USE_SDK"] = "1"
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

cutlass_path = "D:/Project/cutlass/include"
ROOT = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(ROOT, "build")
os.makedirs(build_dir, exist_ok=True)

setup(
    name="attention_kernels",
    ext_modules=[
        CUDAExtension(
            name="attention_extension",
            sources=[
                os.path.join(ROOT, "csrc/src/kernel/attention_binding.cpp"),
                os.path.join(ROOT, "csrc/src/cublas_handle.cu"),
                os.path.join(ROOT, "csrc/src/kernel/attention_kernel.cu"),
                os.path.join(ROOT, "csrc/src/kernel/softmax_kernel.cu"),
            ],
            libraries=["cublas"],
            include_dirs=[os.path.join(ROOT, "csrc/include")],
            extra_compile_args={
                "nvcc": ["-O3", "-lineinfo"],
                "cxx": ["/O2"],
            },
        ),
        CUDAExtension(
            name="flash_attention_simt_extension",
            sources=[
                os.path.join(ROOT, "csrc/src/kernel/flash_attention_simt_binding.cpp"),
                os.path.join(ROOT, "csrc/src/kernel/flash_attention_simt_kernel.cu"),
            ],
            include_dirs=[os.path.join(ROOT, "csrc/include"), cutlass_path],
            extra_compile_args={
                "nvcc": ["-O3", "-lineinfo"],
                "cxx": ["/O2"],
            },
        ),
        CUDAExtension(
            name="flash_attention_tensor_op_extension",
            sources=[
                os.path.join(ROOT, "csrc/src/kernel/flash_attention_tensor_op_binding.cpp"),
                os.path.join(ROOT, "csrc/src/kernel/flash_attention_tensor_op_kernel.cu"),
            ],
            include_dirs=[os.path.join(ROOT, "csrc/include"), cutlass_path],
            extra_compile_args={
                "nvcc": ["-O3", "-lineinfo"],
                "cxx": ["/O2"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
