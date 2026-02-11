from setuptools import setup, find_packages

setup(
    name="lvmh-rag",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
    python_requires=">=3.11",
    author="Ton Nom",
    author_email="ton.email@example.com",
    description="RAG system pour analyse financière LVMH",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ton-username/lvmh-rag",
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
    ],
)