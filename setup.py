from setuptools import setup, find_packages


setup(
    name="lycoris_lora",
    packages=find_packages(),
    version="2.1.0.post3",
    url="https://github.com/KohakuBlueleaf/LyCORIS",
    description="Lora beYond Conventional methods, Other Rank adaptation Implementations for Stable diffusion",
    author="Shih-Ying Yeh(KohakuBlueLeaf), Yu-Guan Hsieh, Zhidong Gao",
    author_email="apolloyeh0123@gmail.com",
    zip_safe=False,
    install_requires=["torch", "safetensors", "diffusers", "transformers"],
)
