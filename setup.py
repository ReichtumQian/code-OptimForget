from setuptools import setup, find_packages

setup(name='optforget',
      version='0.1',
      author='Yixiao Qian',
      author_email='yixiaoqian@zju.edu.cn',
      description='',
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      url='https://github.com/ReichtumQian/PIDAO/code-OptiForget',
      packages=find_packages(),
      classifiers=[
          'Programming Language :: Python :: 3',
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent',
      ],
      python_requires='>=3.8')
