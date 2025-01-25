"""
Package setup file.
"""
from setuptools import setup, find_packages

setup(
    name="gandalf-mapper",
    version="0.1.0",
    description="Your wise guide through website sitemaps, analyzing content with the precision of a wizard.",
    author="Abhishek",
    author_email="abhishekkr23rs@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "flask[async]",
        "gunicorn",
        "crawl4ai",
        "beautifulsoup4",
        "aiohttp",
        "lxml",
        "sentence-transformers",
        "numpy",
        "torch",
        "python-dotenv",
        "pydantic",
        "tenacity",
        "structlog",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-asyncio",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
            "isort",
        ]
    },
    python_requires=">=3.11",
    entry_points={
        "console_scripts": [
            "gandalf-mapper=run:main",
        ]
    },
) 