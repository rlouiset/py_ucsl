class BaseML(object):
    """ Base Machine Learning classifier / clustering method
    """
    def __init__(self, name="Base empty Machine Learning object"):
        self.name = name

    def fit(self, X_train, y_train):
        pass

    def fit_transform(self, X_train, y_train=None):
        return X_train

    def predict(self, X_val):
        return X_val