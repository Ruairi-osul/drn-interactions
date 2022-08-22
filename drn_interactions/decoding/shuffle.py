import numpy as np


def shuffle_X_df(X, y):
    """Shuffle X as a dataframe"""
    X = X.copy().sample(frac=1).reset_index(drop=True)
    return X, y


def shuffle_both(X, y):
    """Shuffle the data"""
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]


def shuffle_X(X, y):
    """Shuffle X"""
    perm = np.random.permutation(X.shape[0])
    return X[perm], y


def shuffle_y(X, y):
    """Shuffle y"""
    perm = np.random.permutation(X.shape[0])
    return X, y[perm]


def shuffle_Xcols(X, y):
    """Shuffle Decorrelate X"""
    X = np.apply_along_axis(np.random.permutation, 0, X)
    return X, y


def roll_X(X, y):
    """Roll X"""
    X = np.roll(X, np.random.randint(1, len(X)), axis=0)
    return X, y


def shuffle_neither(X, y):
    """Shuffle neigher"""
    return X, y
