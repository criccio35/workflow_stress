# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md', encoding='utf-8') as f:
    readme = f.read()

with open('LICENSE', encoding='utf-8') as f:
    license = f.read()

setup(
    name='workflow_stress',
    version='0.1.0',
    description='Condition-specific co-expression network analysis',
    author='Camila Riccio',
    author_email='camila.riccio95@gmail.com',
    url='https://github.com/criccio35/workflow_stress',
    license='GNU GPL V3',
    packages=find_packages(exclude=('test', 'docs'))
)
