# -*- coding: utf-8 -*-
"""
@author: Vladimir Shteyn
@email: vladimir.shteyn@googlemail.com

Copyright Vladimir Shteyn, 2018

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from sys import platform
from setuptools import setup, Extension
from os import path
from numpy import get_include

REQUIREMENTS = []
with open('requirements.txt', 'r') as f: 
	raw = f.read() 
	REQUIREMENTS = raw.replace(' ', '').split('\n')

with open('README.rst', 'r') as f:
    README = f.read()

def create_extension(name, sources): 
    """
    parent_package: str 
        
    """
    if platform == 'darwin':#MacOS
        link_args = []
    else:
        link_args = ['-static-libgcc'] 
    
    compile_args = ['-O3', '-fno-exceptions', '-ffast-math', 
                    '-march=nocona', '-mtune=nocona']
    
    return Extension(name, sources, include_dirs=[get_include()], 
                     extra_compile_args=compile_args, extra_link_args=link_args
                     )

if __name__ == '__main__':
    from setuptools import find_packages
    # path_as_list = ['calibration', 'localization', 'obj']    
    # parent_directory = path.sep.join(path_as_list) + path.sep
    # sources = [parent_directory + 'cgauss.c', 
            #    parent_directory + 'cgauss2dWeighted.c', 
            #    parent_directory + 'cgauss2dModel.c' ]
    
    # module_name = 'cgauss'
    # pkg = path.extsep.join(path_as_list + [module_name])
    # ext = [create_extension(pkg, sources)]
    ver = '0.1'
    url = r'https://github.com/MisterVladimir/endocytosis'
	
    setup(name='endocytosis',
#          package_dir={'':''}, 
          version=ver, 
          packages=find_packages(), 
#          ext_package=pkg, 
#          ext_modules=ext, 
          python_requires='>=3', 
          install_requires=REQUIREMENTS, 
          include_package_data=True, 
          author='Vladimir Shteyn',
          author_email='vladimir.shteyn@googlemail.com',
          url=url, 
		  download_url=r'{0}/archive/{1}.tar.gz'.format(url, ver), 
          long_description = README,
          license = "GNUv3",
          classifiers=[
              'Intended Audience :: Science/Research',
              'Topic :: Scientific/Engineering :: Medical Science Apps.' , 
			  'Programming Language :: Python :: 3.5' ]) 
