from setuptools import setup

setup(
    name='locon',
    packages=['locon'],
    version='0.0.2',
    url='https://github.com/KohakuBlueleaf/LoCon',
    description='LoRA for Convolution Network',
    author='Shih-Ying Yeh(KohakuBlueLeaf)',
    author_email='apolloyeh0123@gmail.com',
    zip_safe=False,
    install_requires=[
        'torch',
        'safetensors',
        'diffusers',
        'transformers'
    ],
)