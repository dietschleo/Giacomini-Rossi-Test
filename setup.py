from setuptools import setup, find_packages

setup(
    name="giacomini_rossi_test",
    version="0.1.0",
    description="Implementation of the Giacomini‑Rossi fluctuation test for forecast comparisons",
    author="Your Name",
    author_email="you@example.com",
    url="https://github.com/dietschleo/Giacomini-Rossi-Test",
    packages=find_packages(),
    install_requires=[
        "numpy==1.26.4",
        "statsmodels==0.14.1",
        "pandas>=1.0",
        "scipy>=1.0",
        "scikit-learn>=1.0",
        # enumerate any additional dependencies based on your modules
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
