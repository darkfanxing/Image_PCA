# MNIST with PCA Analysis
## Setup Project Enviroment
1. download and unzip the [MNIST data](http://yann.lecun.com/exdb/mnist/), and save them to src/pca_weights/
```bash
pip install pipenv
pipenv shell
pipenv install
```

then run the project

```bash
python src/make_full_component.py
python src/main.py
```