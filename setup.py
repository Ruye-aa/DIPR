from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='registration',
    version='1.0.0',
    ext_modules=[
        CUDAExtension(
            name='registration.ext',
            sources=[
                'registration/extensions/extra/cloud/cloud.cpp',
                'registration/extensions/cpu/grid_subsampling/grid_subsampling.cpp',
                'registration/extensions/cpu/grid_subsampling/grid_subsampling_cpu.cpp',
                'registration/extensions/cpu/radius_neighbors/radius_neighbors.cpp',
                'registration/extensions/cpu/radius_neighbors/radius_neighbors_cpu.cpp',
                'registration/extensions/pybind.cpp',
            ],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
