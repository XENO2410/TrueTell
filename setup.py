from setuptools import setup, find_packages

setup(
    name="truetell",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "transformers==4.36.0",
        "torch==2.5.1",
        "streamlit==1.40.2",
    ],
    python_requires=">=3.8",
)