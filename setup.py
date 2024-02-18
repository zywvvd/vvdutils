import sys
import setuptools
from setuptools import find_packages, setup
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration
from os.path import join, dirname, realpath

str_version = '2.1.76'



def configuration(parent_package='', top_path=''):
    # this will automatically build the scattering extensions, using the
    # setup.py files located in their subdirectories
    config = Configuration(None, parent_package, top_path)

    pkglist = setuptools.find_packages()
    for i in pkglist:
        config.add_subpackage(i)
    config.add_data_files(join('mtutils', 'assets', '*.json'))
    config.add_data_files(join('mtutils', 'assets', '*.jpg'))

    return config


if __name__ == '__main__':
    pass
    setup(
        configuration=configuration,
        name='mtutils',
        version=str_version,
        description='Commonly used function library by VVD',
        url='https://github.com/zywvvd/utils_vvd',
        author='zywvvd',
        author_email='zywvvd@mail.ustc.edu.cn',
        license='MIT',
        packages=['mtutils'],
        zip_safe=False,
        install_requires= ['numba', 'func_timeout', 'pypinyin', 'opencv-python', 'scikit-learn', 'pathlib2', 'tqdm', 'pytest', 'matplotlib', 'pandas'],
        python_requires='>=3')