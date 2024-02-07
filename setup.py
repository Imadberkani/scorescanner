from setuptools import setup, find_packages
import os


here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here,'REDAME.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="scorescanner",  
    version="0.1.0",  
    description="scorescanner helps you understand the main factors influencing the target of predictive tabular machine learning models.",  
    long_description=long_description,  
    long_description_content_type="text/markdown",  
    author="Imad Berkani",  
    author_email="berkaniimad75@gmail.com",  
    license="MIT",  
    classifiers=[
        "Programming Language :: Python :: 3", 
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),  
    include_package_data=True,  # Inclut les fichiers spécifiés dans MANIFEST.in
    install_requires=[  
        "ipykernel==6.29.1",
        "nbformat==5.9.2",
        "numpy==1.26.4",
        "optbinning==0.19.0",
        "pandas==1.5.3",
        "plotly==5.18.0",
        "ppscore==1.3.0",
        "scikit-learn==1.4.0",
        "scipy==1.12.0",
        "statsmodels==0.14.1",
        "tqdm==4.66.1"
        
    ],
     test_suite="tests",
)
