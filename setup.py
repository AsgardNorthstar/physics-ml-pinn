from setuptools import setup, find_packages

setup(
    name="aagjuuk-core",
    version="0.1.0",
    author="Aagjuuk Labs",
    description="High-performance, continuous 3D anisotropic physical-informed edge solvers.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/asgardnorthstar/aagjuuk-core",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "matplotlib",
        "streamlit",
        "numpy-stl"
    ],
)
