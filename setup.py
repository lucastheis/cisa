import os
import numpy
from distutils.core import setup, Extension
from distutils.ccompiler import new_compiler

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
			os.path.join(numpy.__path__[0], 'core/include/numpy')],
		library_dirs=[],
		libraries=[
			'gomp'],
		extra_link_args=[
			'code/liblbfgs/lib/.libs/liblbfgs.a'],
		extra_compile_args=[
			'-std=c++0x',
			'-fopenmp',
			'-Wno-parentheses',
			'-Wno-write-strings'])]

setup(
	name='isa',
	version='0.1',
	description='',
	ext_modules=modules)
