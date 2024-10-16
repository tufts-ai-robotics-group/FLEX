from setuptools import setup, find_packages

setup(
    name="flex",  # Replace with your package name
    version="0.1.0",  # Initial release version
    author="Wenchang Gao",
    author_email="wenchanggao93@gmail.com",
    url="https://github.com/tufts-ai-robotics-group/FLEX",  # Optional, replace with your project URL
    packages=find_packages(),  # Automatically finds and includes all packages
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Specify the minimum Python version
)
