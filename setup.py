from setuptools import setup, find_packages
import re

INIT_FILE = "cratenet/__init__.py"

with open(INIT_FILE) as fid:
    file_contents = fid.read()
    match = re.search(r"^__version__\s?=\s?['\"]([^'\"]*)['\"]", file_contents, re.M)
    if match:
        version = match.group(1)
    else:
        raise RuntimeError("Unable to find version string in %s" % INIT_FILE)

with open("requirements.txt") as f:
    required = f.read().splitlines()

packages = find_packages(exclude=("tests", "demos", "data", "resources", "bin", "out",))

setup(name="cratenet",
      version=version,
      description="CraTENet",
      long_description="CraTENet is an attention-based deep neural network for thermoelectric transport properties.",
      license="MIT",
      classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Developers',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3.6',
      ],
      url='http://github.com/lantunes/CraTENet',
      author="Luis M. Antunes",
      author_email="lantunes@gmail.com",
      packages=packages,
      keywords=["machine learning", "materials science", "materials informatics", "chemistry"],
      python_requires='>=3.6',
      install_requires=required)
