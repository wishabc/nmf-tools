from setuptools import setup
import os
import re


def read_version():
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'nmf_tools/__init__.py')
    with open(path, 'r') as fh:
        return re.search(r'__version__\s?=\s?[\'"](.+)[\'"]', fh.read()).group(1)


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name='nmf-tools',
    version=read_version(),
    description='',
    long_description=readme(),
    long_description_content_type='text/markdown',
    author='Alexandr Boytsov',
    author_email='sboytsov@altius.org',
    url='https://github.com/wishabc/nmt-tools.git',
    license='GPL3',
    packages=['nmf-tools'],
    entry_points={
        'console_scripts': [
           # 'nmf-tools = nmf-tools.__main__:main'
        ]
    },
    install_requires=['pandas', 'numpy', 'matplotlib>=3.3.3',
                    'scikit-learn', 'seaborn', 'statsmodels',
                    'jupyterlab'],
    python_requires='>=3.8',
    data_files=[("", ["LICENSE"])]
)