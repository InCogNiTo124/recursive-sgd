import setuptools

with open('README.md', 'r') as file:
    long_description = file.read()

setuptools.setup(
    name="recursive_sgd",
    version="0.5",
    author="Marijan Smetko",
    author_email="marijan.smetko@gmail.com",
    description="Train a neural network - with recursion!",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/InCogNiTo124/recursive-sgd/',
    entry_points={
        'console_scripts': ['sgd=recursive_sgd.cli:main'],    
    },
    packages=setuptools.find_packages(),
    classifiers=[],
    python_requires='>=3.4',
    install_requires=['numpy>=1.17']
)
