from setuptools import setup, find_packages

setup(
    name="crmstudio",
    version="0.1.0",
    description="IRB Model Monitoring Toolkit for Banks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial",
    ],
    keywords=["banking", "irb", "model-monitoring", "risk-management"],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.1",
    install_requires=[
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
        ],
    },
    project_urls={
        "Homepage": "https://github.com/yourusername/crmstudio",
        "Documentation": "https://crmstudio.readthedocs.io",
        "Source": "https://github.com/yourusername/crmstudio.git",
    },
)