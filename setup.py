from setuptools import setup, find_packages

setup(
    name='i-if-learn',  # 包名不建议用大写或下划线
    version='0.1.0',
    packages=find_packages(),  # 自动查找含 __init__.py 的目录
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'umap-learn',
    ],
    author='MC',
    author_email='12112941@mail.sustech.edu.cn',
    description='A package for high dimensional clustering method.',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mc25800852/i_IF_learn',
)
