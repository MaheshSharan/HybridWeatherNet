from setuptools import setup, find_packages

setup(
    name="weather_bias_correction",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "xarray",
        "requests",
        "tqdm",
        "netCDF4",
        "pytest",
        "torch>=2.0.0",
        "pytorch-lightning>=2.0.0",
        "torch-geometric>=2.3.0",
        "streamlit>=1.24.0",
    ],
    python_requires=">=3.8",
    author="SeoYea-Ji",
    description="A package for correcting biases in weather forecasts using deep learning",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)