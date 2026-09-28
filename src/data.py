from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def load_and_split_data():
    wine = load_wine(as_frame=True)
    X = wine.data
    y = wine.target

    if X.isnull().sum().sum() > 0:
        raise ValueError("Data contains null values")

    if X.shape[1] != 13:
        raise ValueError("Feature count is not 13")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test