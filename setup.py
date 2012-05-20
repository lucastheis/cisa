import os
import numpy
from distutils.core import setup, Extension
from distutils.ccompiler import new_compiler

modules = [
	Extension('isa',
		language='c++',
		sources=[
			'code/isa/src/module.cpp',
			'code/isa/src/isa.cpp'],
		include_dirs=[
			'code',
			'code/isa/include',
			os.path.join(numpy.__path__[0], 'core/include/numpy')],
		library_dirs=[],
		libraries=[],
		extra_link_args=[],
		extra_compile_args=[
			'-g',
			'-Wno-write-strings'])]

setup(
	name='isa',
	version='0.1',
	description='',
	ext_modules=modules)
