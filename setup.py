from setuptools import setup, find_packages

setup(
    name='geom2vec',
    version='0.0.1',
    description="Geom2Vec (geometry-to-vector) is a framework for compute vector representation of molecular "
                "conformations using pretrained graph neural networks (GNNs).",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'MDAnalysis',
        'tqdm',
        'tensorboard',
        'ase',
        'rdkit',
        'matplotlib',
        'einops',
        'grokfast_pytorch',
        'adam-atan2-pytorch'
    ]
)

