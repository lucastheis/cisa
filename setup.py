import os
import numpy
from distutils.core import setup, Extension
from distutils.ccompiler import new_compiler

modules = [
	Extension('isa',
		language='c++',
		sources=[
			'code/isa/src/module.cpp',
			'code/isa/src/distribution.cpp',
			'code/isa/src/isa.cpp',
			'code/isa/src/gsm.cpp'],
		include_dirs=[
			'code',
			'code/isa/include',
			os.path.join(numpy.__path__[0], 'core/include/numpy')],
		library_dirs=[],
		libraries=[],
		extra_link_args=[
			'-lgomp'],
		extra_compile_args=[
			'-fopenmp',
			'-Wno-parentheses',
			'-Wno-write-strings'])]

setup(
	name='isa',
	version='0.1',
	description='',
	ext_modules=modules)
