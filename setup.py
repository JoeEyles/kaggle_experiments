from setuptools import setup, find_packages

setup(
    name="kaggle",
    version="1.0",
    author="Your Name",
    author_email="your_email@example.com",
    description="A short description of your project",
    packages=find_packages(),  # Automatically find and include all packages
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Specify the required Python version
)
