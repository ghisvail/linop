from setuptools import setup

f = open('README.txt')
try:
    README = f.read()
finally:
    f.close()

setup(
    name='linop',
    version='0.8.1',
    author='Ghislain Vaillant',
    author_email='ghisvail@gmail.com',
    description='A pythonic abstraction for linear mathematical operators',
    long_description=README,
    license='BSD',
    url='https://github.com/ghisvail/linop',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
    ],
    keywords=['linear', 'operator', 'mathematics'],
    packages=['linop'],
    tests_require=['nose', 'numpy', 'scipy'],
    test_suite="nose.collector",
    use_2to3=True,
    zip_safe=False
)
