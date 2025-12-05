from setuptools import setup, find_packages


setup(
    name="orpheus-speech",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["snac", "vllm", "requests"],
    author="Amu Varma",
    author_email="amu@canopylabs.com",
    description="Orpheus Text-to-Speech System",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/canopyai/orpheus-tts",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
