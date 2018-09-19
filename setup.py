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
from sys import platform, maxsize, version_info
import os
import re
from setuptools import setup, Extension
from Cython.Build import cythonize
from numpy import get_include
import subprocess
import ctypes

with open('README.md', 'r') as f:
    README = f.read()


def check_java():
    """
    Check whether Java Runtime Environment and Java Development Kit directories
    exist as a proxy for whether they're installed. This code is mostly ripped
    off from javabridge.locate version 1.0.17, with a few bug fixes.
    """
    is_linux = platform.startswith('linux')
    is_mac = platform == 'darwin'
    is_win = platform.startswith("win")

    if 'JAVA_HOME' in os.environ:
        return True
    elif is_mac:
        # Use the "java_home" executable to find the location
        # see "man java_home"
        libc = ctypes.CDLL("/usr/lib/libc.dylib")
        if maxsize <= 2**32:
            arch = "i386"
        else:
            arch = "x86_64"
        try:
            result = subprocess.check_output(["/usr/libexec/java_home",
                                              "--arch", arch])
            path = result.strip().decode("utf-8")
            for place_to_look in (
                            os.path.join(os.path.dirname(path), "Libraries"),
                            os.path.join(path, "jre", "lib", "server")):
                # In "Java for OS X 2015-001" libjvm.dylib is a symlink to
                # libclient.dylib
                # which is i686 only, whereas libserver.dylib contains both 
                # architectures.
                for file_to_look in ('libjvm.dylib',
                                     'libclient.dylib',
                                     'libserver.dylib'):
                    lib = os.path.join(place_to_look, file_to_look)
                    #
                    # dlopen_preflight checks to make sure the dylib
                    # can be loaded in the current architecture
                    #
                    if os.path.exists(lib) and libc.dlopen_preflight(
                                                    lib.encode('utf-8')) != 0:
                        return True
        except:
            return False
            # logger.error("Failed to run /usr/libexec/java_home, defaulting to"
            #              "best guess for Java", exc_info=1)
            # return "/System/Library/Frameworks/JavaVM.framework/Home"

    elif is_linux:
        def get_out(cmd):
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)
            o, ignore = p.communicate()
            if p.poll() != 0:
                raise Exception("Error finding javahome on linux: %s" % cmd)
            o = o.strip().decode('utf-8')
            return o
        try:
            java_bin = get_out(["bash", "-c", "type -p java"])
            java_dir = get_out(["readlink", "-f", java_bin])
            java_version_string = get_out(["bash", "-c", "java -version"])
            if re.match('^openjdk', java_version_string) is not None:
                jdk_dir = os.path.join(java_dir, "..", "..", "..")
            elif re.match('^java', java_version_string) is not None:
                jdk_dir = os.path.join(java_dir, "..", "..")
            # jdk_dir = os.path.abspath(jdk_dir)
            return True
        except Exception:
            return False

    elif is_win:
        if version_info.major == 2:
            import _winreg as winreg
        else:
            import winreg
        java_key_path = 'SOFTWARE\\JavaSoft\\'
        jre_path = java_key_path + 'JRE'
        jdk_path = java_key_path + 'JDK'
        try:
                _ = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, jre_path)
                _ = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, jdk_path)
        except:
            return False
        else:
            return True


def get_requirements(*args):
    if 'required' not in args:
        args += ('required',)
    with open('requirements.txt', 'r') as f:
        raw = f.read().replace(' ', '')
        req_dict = {}
        for item in args:
            try:
                req_dict[item] = [t for t in raw.split('[{0}]'.format(item))[1]
                                  .split('[')[0].split('\n') if t]
            except IndexError:
                # item not in requirements.txt
                pass

        if not check_java() and 'java' in req_dict:
            del req_dict['java']

        return sum(req_dict.values(), [])


def create_extension(path_as_list, sources, module_name, cy=False):
    # def create_extension(name, sources):
    """
    """
    if platform == 'darwin':
        # MacOS
        link_args = []
    else:
        link_args = ['-static-libgcc']

    compile_args = ['-O3', '-fno-exceptions', '-ffast-math',
                    '-march=nocona', '-mtune=nocona']

    parent_directory = os.path.sep.join(path_as_list) + os.path.sep
    sources = [parent_directory + s for s in sources]
    name = os.path.extsep.join(path_as_list + [module_name])

    ext = Extension(name, sources, include_dirs=[get_include()],
                    extra_compile_args=compile_args, extra_link_args=link_args)

    if cy:
        ext = cythonize(ext)[0]

    return ext

if __name__ == '__main__':
    from setuptools import find_packages
    kwargs = [{'path_as_list': ['endocytosis', 'simulation', 'obj'],
               'sources': ['cygauss2d.pyx'],
               'module_name': 'cygauss2d',
               'cy': True},
              {'path_as_list': ['endocytosis', 'contrib', 'gohlke'],
               'sources': ['psf.c'],
               'module_name': '_psf'},
              {'path_as_list': ['endocytosis', 'contrib', 'gohlke'],
               'sources': ['tifffile.c'],
               'module_name': '_tifffile'}]
    ext = [create_extension(**k) for k in kwargs]
    ver = '0.1'
    url = r'https://github.com/MisterVladimir/endocytosis'
    setup(name='endocytosis',
          version=ver,
          packages=find_packages(),
          # ext_package=pkg,
          ext_modules=ext,
          python_requires='>=3.6',
          install_requires=get_requirements('java'),
          include_package_data=True,
          author='Vladimir Shteyn',
          author_email='vladimir.shteyn@googlemail.com',
          url=url,
          download_url=r'{0}/archive/{1}.tar.gz'.format(url, ver),
          long_description=README,
          license="GNUv3",
          classifiers=[
              'Intended Audience :: Science/Research',
              'Topic :: Scientific/Engineering :: Medical Science Apps.',
              'Topic :: Scientific/Engineering :: Image Recognition',
              'Programming Language :: Python :: 3.6'])
