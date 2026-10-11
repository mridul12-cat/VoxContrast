from setuptools import setup, find_packages

setup(
    name="voxcontrast",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.22.0",
        "scipy>=1.9.0",
        "soundfile>=0.12.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "pydantic>=2.0.0",
        "python-multipart>=0.0.6",
    ],
)
