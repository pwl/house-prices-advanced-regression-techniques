import pandas as pd


def split_data(data, train_column="Train", target_column="SalePrice"):
    """Splits the data set into training and test data"""
    train = data[train_column]
    x = data.drop([train_column, target_column], axis=1)
    y = data[target_column]
    return x[train], y[train], x[~train]
