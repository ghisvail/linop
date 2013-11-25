#!/usr/bin/env python

import os
import sys

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('linop')
    config.get_version(os.path.join('linop', 'version.py'))
    return config


description = 'linop: a Pythonic abstraction for linear mathematical operators'

long_description = '''
linop: a Pythonic abstraction for linear mathematical operators

A friendy fork from the linop module of the pykrylov package, developped by
Dominique Orban <dominique.orban@gmail.com> and available at
https://github.com/dpo/pykrylov.

This project means to provde a standalone set of classes to abstract the
creation and management of linear operators, to be used as a common basis for
the development of advanced mathematical frameworks.
'''


def setup_package():

    from numpy.distutils.core import setup

    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(local_path)
    sys.path.insert(0, local_path)
    sys.path.insert(0, os.path.join(local_path, 'linop'))  # for version

    try:
        setup(
            name='linop',
            maintainer="Ghislain Vaillant",
            maintainer_email="ghisvail@gmail.com",
            description=description,
            long_description=long_description,
            url="https://github.com/ghisvail/linop",
            #download_url="http://github.com/dpo/pykrylov/tarball/0.1.1",
            #license='LICENSE',
            classifiers=[
                "Development Status :: 3 - Alpha",
                "Intended Audience :: Science/Research",
                "Intended Audience :: Developers",
                "License :: OSI Approved",
                "Programming Language :: Python",
                "Topic :: Software Development",
                "Topic :: Scientific/Engineering",
                "Operating System :: OS Independent",
            ],
            install_requires=["numpy"],
            configuration=configuration)
    finally:
        del sys.path[0]
        os.chdir(old_path)

    return

if __name__ == '__main__':
    setup_package()
