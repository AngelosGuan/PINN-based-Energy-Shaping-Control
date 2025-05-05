from setuptools import setup, find_packages

setup(
    name='PINN-based Energy Shaping Control',
    version='0.1',
    author='Angelos Guan',
    author_email='angelosguan@gmail.com',
    description='A PINN-based method for Energy Shaping Control.',
    url='https://github.com/AngelosGuan/PINN-based-Energy-Shaping-Control',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'matplotlib',
        'scipy',
    ],
    python_requires='>=3.8',
)