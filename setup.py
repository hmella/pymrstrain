#!/usr/bin/env python
import os
import sys

import setuptools
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


def read(fname):
    try:
        content = codecs.open(
            os.path.join(os.path.dirname(__file__), fname),
            encoding='utf-8'
            ).read()
    except Exception:
        content = ''
    return content


__version__ = '0.0.1'


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


ext_modules = [
    Extension(
        'ImageBuilding',
        ['PyMRStrain/src/ImageBuilding.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            '/usr/include/eigen3/'
        ],
        language='c++'
    ),
    Extension(
        'Connectivity',
        ['PyMRStrain/src/Connectivity.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            '/usr/include/eigen3/'
        ],
        language='c++'
    ),
    Extension(
        'TrajToImage',
        ['PyMRStrain/src/TrajToImage.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            '/usr/include/eigen3/'
        ],
        language='c++'
    ),
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.

    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)


setup(name='PyMRStrain',
      version='0.1',
      packages=['PyMRStrain'],
      description='Strain Phantom Library',
      long_description=read('README.md'),
      ext_modules=ext_modules,
      install_requires=['pybind11>=2.2'],
      cmdclass={'build_ext': BuildExt},
      zip_safe=False,
      author='Hernan Mella',
      author_email='hmella@uc.cl'
      )
