import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="image-regist",                     # This is the name of the package
    version="0.0.4",                        # The initial release version
    author="Muhamad Fikri Sunandar",                     # Full name of the author
    description="Image Registration Library",
    long_description=long_description,      # Long description read from the the readme file
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(include=['imageregist']),    # List of all python modules to be installed
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],                                      # Information to filter the project on PyPi website
    python_requires='>=3.6',                # Minimum version requirement of the package
    py_modules=["image-regist"],             # Name of the python package
    install_requires=['numpy==1.21.6', 'Pillow==9.3.0', 'scipy==1.7.3']                     # Install other dependencies if any
)