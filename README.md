UCSL : Unsupervised Clustering driven by Supervised Learning
===============================

This repository contains codes and examples for UCSL : Unsupervised Clustering driven by Supervised Learning (UCSL) algorithm.

We aim at unsupervisedly identify subtypes and drive it with a supervised task (classification or regression).

The algorithm is implemented in a pedagogical, easy-to-use sklearn fashion with fit(), fit_predict(), predict() and predict_clusters() functions.

Method codes are available in "ucsl" directory, examples implemented in notebook are implemented in "examples" directory.

Installation
------------
Unless you already have sklearn, Numpy and Scipy installed, you need to install them:

```
sudo apt-get install python-numpy python-scipy
```

Clone the repository from github
```
git clone https://github.com/rlouiset/UCSL.git
```

Add `UCSL` in your `$PYTHONPATH`


Dataset
-------

```python
import numpy as np
```
