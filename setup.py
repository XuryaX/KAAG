from setuptools import setup, find_packages

setup(
    name="kaag",
    version="0.1.0",
    packages=find_packages(exclude=["tests*"]),
    include_package_data=True,
    install_requires=[
        "networkx>=2.5",
        "jinja2",
        "pyyaml",
        "requests",
        "numpy",
        "scikit-learn",
        "spacy",
        "textblob",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "kaag-evaluate=scripts.evaluation:main",
        ],
    },
    python_requires=">=3.7",
    author="Shaurya Chaudhuri",
    author_email="shaurya@aroundai.co",
    description="Knowledge and Aptitude Augmented Generation framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/aroundAI/kaag",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)