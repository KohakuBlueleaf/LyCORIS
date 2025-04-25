from setuptools import setup, find_packages


setup(
    name="lycoris_lora",
    packages=find_packages(),
    version="3.2.0.post2",
    url="https://github.com/KohakuBlueleaf/LyCORIS",
    description="Lora beYond Conventional methods, Other Rank adaptation Implementations for Stable diffusion",
    author="Shih-Ying Yeh(KohakuBlueLeaf), Yu-Guan Hsieh, Zhidong Gao",
    author_email="kohaku@kblueleaf.net",
    zip_safe=False,
    install_requires=["torch", "einops", "toml", "tqdm"],
    python_requires=">=3.10",
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: GPU :: NVIDIA CUDA :: 11.8",
        "Environment :: GPU :: NVIDIA CUDA :: 12",
        "Environment :: GPU :: NVIDIA CUDA :: 12 :: 12.1",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
)
