from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="kaag",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Knowledge and Aptitude Augmented Generation framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/kaag",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.4.0",
        "pyyaml>=5.4.0",
        "textblob>=0.15.3",
        "requests>=2.26.0",
        "openai>=0.27.0",
        "anthropic>=0.2.0",
        "sqlalchemy>=1.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.5b1",
            "isort>=5.8.0",
            "mypy>=0.812",
        ],
    },
)