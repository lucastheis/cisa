#!/usr/bin/env python

import os
import sys
import numpy

sys.path.append('./code')

from distutils.core import setup, Extension
from distutils.ccompiler import CCompiler, get_default_compiler
from tools import parallelCCompiler
from numpy.distutils.intelccompiler import IntelCCompiler
from numpy import any

# heuristic for figuring out which compiler is being used (icc, gcc)
if any(['intel' in arg for arg in sys.argv]) or 'intel' in get_default_compiler():
	# icc-specific options
	include_dirs=[
		'/opt/intel/mkl/include']
	library_dirs=[
		'/opt/intel/mkl/lib',
		'/opt/intel/lib']
	libraries = [
		'mkl_intel_lp64',
		'mkl_intel_thread',
		'mkl_core',
		'mkl_def',
		'iomp5']
	extra_compile_args = [
		'-DEIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS',
		'-DEIGEN_USE_MKL_ALL',
		'-wd1224',
		'-openmp']
	extra_link_args = []

	for path in ['/opt/intel/mkl/lib/intel64', '/opt/intel/lib/intel64']:
		if os.path.exists(path):
			library_dirs += [path]
else:
	# gcc-specific options
	include_dirs = []
	library_dirs = []
	libraries = []
	extra_compile_args = []
	extra_link_args = []

	if sys.platform != 'darwin':
		extra_compile_args += ['-Wno-cpp', '-fopenmp']
		libraries += ['gomp']
		

if sys.platform != 'darwin':
	extra_compile_args += ['-std=c++0x']

modules = [
	Extension('isa',
		language='c++',
		sources=[
			'code/isa/src/isainterface.cpp',
			'code/isa/src/gsminterface.cpp',
			'code/isa/src/pyutils.cpp',
			'code/isa/src/isa.cpp',
			'code/isa/src/gsm.cpp',
			'code/isa/src/utils.cpp',
			'code/isa/src/module.cpp',
			'code/isa/src/callbacktrain.cpp',
			'code/isa/src/distribution.cpp'],
		include_dirs=[
			'code',
			'code/isa/include',
			'code/liblbfgs/include',
			os.path.join(numpy.__path__[0], 'core/include/numpy')] + include_dirs,
		library_dirs=[] + library_dirs,
		libraries=[] + libraries,
		extra_link_args=[
			'code/liblbfgs/lib/.libs/liblbfgs.a'] + extra_link_args,
		extra_compile_args=[
			'-Wno-parentheses',
			'-Wno-write-strings'] + extra_compile_args)]

# enable parallel compilation
CCompiler.compile = parallelCCompiler

setup(
	name='isa',
	version='0.4.3',
	author='Lucas Theis',
	author_email='lucas@theis.io',
	description='A C++ implementation of overcomplete ISA.',
	url='http://github.com/lucastheis/cisa',
	license='MIT',
	ext_modules=modules)
