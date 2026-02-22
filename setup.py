from setuptools import setup, find_packages
from typing import List

def getRequirements(filepath: str) -> List[str]:
    requirements = []
    with open(filepath) as f_obj:
        requirements = f_obj.read().splitlines()
    requirements = [req for req in requirements if req and not req.startswith('#') and '-e .' not in req]
    return requirements

setup(
    name='SSS',
    version='0.1.0',
    packages=find_packages(),
    install_requires=getRequirements('requirements.txt')
)