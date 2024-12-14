from setuptools import setup, find_packages

setup(
    name="presumm",
    packages=['presumm'],
    version="1.0.0",
    python_requires=">=3.6",
    install_requires=[
        "multiprocess", # ==0.70.9
        "numpy", # ==1.17.2
        "pyrouge", # ==0.1.3
        "pytorch-transformers", # ==1.2.0
        "tensorboardX", # ==1.9
        "torch" # ==1.1.0
    ],
)