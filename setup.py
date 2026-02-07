"""Setup script for PaperForge AI."""
from setuptools import setup, find_packages

setup(
    name="paperforge-ai",
    version="0.1.0",
    description="Convert research papers to runnable Python code using AI",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "openai>=1.0.0",
        "PyMuPDF>=1.23.0",
        "typer>=0.9.0",
        "rich>=13.0.0",
        "typing-extensions>=4.0.0",
    ],
    entry_points={
        "console_scripts": [
            "paperforge-ai=src.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
