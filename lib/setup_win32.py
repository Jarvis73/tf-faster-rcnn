from distutils.core import setup
from Cython.Distutils import build_ext
from distutils.extension import Extension
import numpy as np

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

ext_modules = [
    Extension(
        "utils.cython_bbox",
        ["utils/bbox.pyx"],
        extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
        include_dirs = [numpy_include]
    ),
    Extension(
        "nms.cpu_nms",
        ["nms/cpu_nms.pyx"],
        extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
        include_dirs = [numpy_include]
    )
]

setup(
    name='tf_faster_rcnn',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
)
