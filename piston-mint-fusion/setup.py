from setuptools import setup, find_packages

setup(
    name="piston-mint-fusion",
    version="0.1.0",
    description="PIsToN + MINT Fusion Model for Protein Interface Quality Prediction",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "biopython>=1.79",
        "scikit-learn>=1.0.0",
        "einops>=0.7.0",
        "tqdm",
        "plotly",
        "ml-collections",
    ],
)
