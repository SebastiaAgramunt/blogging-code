from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages
from pathlib import Path
import sysconfig

__version__ = '0.0.1'

REPO_PATH = Path(__file__).resolve().parent

PACKAGE_NAME = 'package_example'

PYTHON_LIB_INCLUDES = sysconfig.get_path('include')
PACKAGE_LIB_INCLUDES = REPO_PATH / 'include'

SRC_FILES = [
    str(REPO_PATH / 'src' / 'matmul.cpp'),
    str(REPO_PATH / 'src' / 'bindings.cpp'),
]

EXTRA_COMPILE_ARGS = ['-O3', '-std=c++17']

ext_modules = [
    Pybind11Extension(
        'package_example._core',  # ⬅️ submodule inside the package
        SRC_FILES,
        include_dirs=[PYTHON_LIB_INCLUDES, str(PACKAGE_LIB_INCLUDES)],
        extra_compile_args=EXTRA_COMPILE_ARGS,
        define_macros=[('VERSION_INFO', __version__)],
    ),
]

setup(
    name=PACKAGE_NAME,
    version=__version__,
    author='Sebastia Agramunt Puig',
    author_email='contact@agramunt.me',
    url='https://github.com/sebastiaagramunt/blogging-code/tree/main/python-cpp-boilerplate',
    description='Example package with C++ extension',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
    python_requires='>=3.9,<3.14',
    install_requires=[
        'numpy>=1.20',
    ],
    extras_require={'test': ['pytest', 'ruff', 'mypy', 'pre-commit']},
)
