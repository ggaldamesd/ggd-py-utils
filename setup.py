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
        # "nltk==3.8.1",
        # "pandas==2.2.2",
        # "tqdm==4.66.2",
        # "imbalanced-learn==0.12.3",
        # "scikit-learn==1.5.0",
        # "numpy==1.26.4",
        "faiss-cpu==1.8.0",
        "unidecode==1.3.8",
        "colorama==0.4.6",
        "chime==0.7.0",
        "fasttext-wheel==0.9.2",
    ],
)
