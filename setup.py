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
        "unidecode==1.3.8",
        "nltk==3.8.1",
        "pandas==2.2.2",
        "tqdm==4.66.2",
        "imbalanced-learn==0.12.3",
        "scikit-learn==1.5.0",
        "colorama==0.4.6",
        "chime==0.7.0",
        "numpy==1.19.0"
    ],
)
