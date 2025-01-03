from setuptools import setup, find_packages

# Read the contents of your README file
from pathlib import Path

setup(
    name="ProDomino",  # Your package name
    version="0.1.0",  # Initial release version
    author="Niopek Lab",  # Your author name
    url="https://github.com/Niopek-Lab/ProDomino",  # URL of your package's repository or website
    packages=find_packages(),  # Automatically find and include all packages
    classifiers=[
        "Programming Language :: Python :: 3",  # Supported Python versions
        "License :: OSI Approved :: MIT License",  # License type (adjust as needed)
        "Operating System :: OS Independent",  # OS support (cross-platform)
    ],
    python_requires=">=3.6",  # Minimum Python version requirement
    install_requires=[
        "lightning==2.3.0",
        "biopython==1.81",
        "py3Dmol==2.1.0",
        "torchmetrics==1.4.0",
        "fair-esm==2.0.0",
        "fairscale==0.4.13",
        "fsspec==2024.3.0"
    ],
    extras_require={
        "dev": [
            "pytest",  # Example of extra requirements for development
            "flake8",
        ],
    },
    include_package_data=True,  # To include non-code files specified in MANIFEST.in
)