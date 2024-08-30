from setuptools import setup, find_packages

setup(
    name="ggd-py-utils",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    description="A collection of utility functions for my projects.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Gerardo Ignacio Galdames Díaz",
    author_email="gerardogaldames@gmail.com",
    url="https://github.com/ggaldamesd/ggd-utils",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "requests>=2.26.0"
    ],
)
