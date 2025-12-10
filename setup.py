from setuptools import find_packages, setup

setup(
    name='geom2vec',
    version='0.2.0',
    description="Geom2Vec (geometry-to-vector) is a framework for compute vector representation of molecular "
                "conformations using pretrained graph neural networks (GNNs).",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "scipy",
        "MDAnalysis",
        "mdtraj",
        "einops",
        "tqdm",
    ],
    extras_require={
        "torch": [
            "torch",
        ],
        "pyg": [
            "torch-geometric",
            "torch-scatter",
            "torch-cluster",
        ],
        "optim": [
            "grokfast_pytorch",
            "adam-atan2-pytorch",
        ],
        "pretrain": [
            "ase",
            "tensorboard",
            "pytorch_lightning",
            "torch",
            "torch-geometric",
            "torch-scatter",
            "torch-cluster",
        ],
        "viz": [
            "matplotlib",
        ],
    },
)
