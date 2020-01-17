import setuptools

long_description=""

setuptools.setup(
    name="recursive_sgd",
    version="0.4.0",
    author="Marijan Smetko",
    author_email="marijan.smetko@gmail.com",
    description="Train a neural network - with recursion!",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/InCogNiTo124/recursive-sgd/',
    packages=setuptools.find_packages(),
    classifiers=[],
    python_requires='>=3.4'
)
