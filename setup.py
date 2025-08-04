from setuptools import setup, find_packages

__package_name__ = "PACKAGE_NAME"
__version__ = "1.0.0"
__description__ = "This package provides methods to analyze and randomly generate Boolean functions and networks, with a focus on research applications in systems biology."

setup(
      name = __package_name__,
      version = __version__,
      description = __description__,
      long_description = __description__,
      
      author = "Claus Kadelka",
      author_email = "ckadelka@iastate.edu",
      url = "https://github.com/ckadelka/BooleanNetworkToolbox",
      
      packages = find_packages(),
      
      classifiers = [
          "Programming Language :: Python :: 3",
      ],
      
      install_requires = [
          "numpy",
          "networkx",
          "scipy"
      ]
)