#!/usr/bin/env python

import os
import sys
import numpy
from distutils.core import setup, Extension

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
			'-fopenmp',
			'-Wno-parentheses',
			'-Wno-write-strings'] + ['-std=c++0x'] if sys.platform != 'darwin' else [])]

setup(
	name='isa',
	version='0.3.1',
	author='Lucas Theis',
	author_email='lucas@theis.io',
	description='A C++ implementation of overcomplete ISA.',
	url='http://github.com/lucastheis/cisa',
	license='MIT',
	ext_modules=modules)
