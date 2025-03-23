from setuptools import find_packages, setup

setup(
    name="opt_cwm",
    version="1.0",
    packages=find_packages(include=["data", "models", "utils"]),
    description="Self-Supervised Learning of Motion Concepts by Optimizing Counterfactuals",
    author="Stefan Stojanov, David Wendt, Seungwoo Kim, Rahul Venkatesh, Kevin Feigelis, Jiajun Wu, Daniel LK Yamins",
    python_requires=">=3.10",
    install_requires=[
        "prettytable",
        "easydict",
        "dacite",
        "numpy==1.26.4",
        "torch==2.0.0",
        "torchvision==0.15.1",
        "einops",
        "matplotlib",
        "timm",
        "decord",
        "h5py",
        "scipy",
    ],
)
