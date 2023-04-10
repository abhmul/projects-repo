from setuptools import setup
from setuptools import find_packages


setup(
    name="pyprojects",
    version="0.0.1",
    description="Repository for all my experiments",
    author="Abhijeet Mulgund",
    author_email="abhmul@gmail.com",
    url="https://github.com/abhmul/projects-repo",
    license="MIT",
    install_requires=[],
    extras_require={
        "tests": [
            "pytest",
            # 'pytest-pep8',
            "pytest-xdist",
            "coveralls",
            "codecov",
            "pytest-cov",
            "python-coveralls",
        ]
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(),
)

