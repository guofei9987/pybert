from setuptools import setup, find_packages
import os
import pybert

this_directory = os.path.abspath(os.path.dirname(__file__))


# 读取文件内容
def read_file(filename):
    with open(os.path.join(this_directory, filename), encoding='utf-8') as f:
        long_description = f.read()
    return long_description


# 获取依赖
def read_requirements(filename):
    return [line.strip() for line in read_file(filename).splitlines()
            if not line.startswith('#')]


setup(name='pybert',
      python_requires='>=3.5',
      version=pybert.__version__,
      description='一行代码训练 BERT',
      long_description=read_file('README.md'),
      long_description_content_type="text/markdown",
      url='https://github.com/guofei9987/pybert',
      author='Guo Fei',
      author_email='guofei9987@foxmail.com',
      license='MIT',
      packages=find_packages(),
      platforms=['linux', 'windows', 'macos'],
      install_requires=read_requirements('requirements.txt'),
      zip_safe=False)
