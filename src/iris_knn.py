"""Small helpers shared by the notebooks, so no code is copied between them."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "iris.data"
ASSETS = ROOT / "assets"

FEATURES = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
CLASSES = ["setosa", "versicolor", "virginica"]
# Colour-blind-safe colours (Okabe-Ito) and different markers, so the plots do not rely on colour only.
COLORS = ["#0072B2", "#E69F00", "#009E73"]
MARKERS = ["o", "s", "^"]


def load_iris_frame(path=DATA_PATH):
    """Read the UCI iris file into a DataFrame with a `species` column."""
    df = pd.read_csv(path, header=None, names=FEATURES + ["species"])
    df["species"] = df["species"].str.replace("Iris-", "", regex=False)
    return df


def split_data(df, test_size=0.2, seed=SEED):
    """Stratified train/test split: every species keeps the same share in both parts."""
    X = df[FEATURES]
    y = df["species"]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def save_figure(fig, name, dpi=100):
    """Save a figure into assets/ and return its path."""
    ASSETS.mkdir(exist_ok=True)
    path = ASSETS / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def scatter_species(ax, df, x, y, alpha=0.9):
    """Scatter plot of two features, one colour and marker per species."""
    for name, color, marker in zip(CLASSES, COLORS, MARKERS):
        part = df[df["species"] == name]
        ax.scatter(part[x], part[y], c=color, marker=marker, s=36, alpha=alpha,
                   edgecolor="white", linewidth=0.4, label=name)
    ax.set_xlabel(x)
    ax.set_ylabel(y)


def plot_decision_regions(ax, model, df, x, y, title=""):
    """Colour the plane by the class a fitted 2-feature model would predict."""
    x_min, x_max = df[x].min() - 0.5, df[x].max() + 0.5
    y_min, y_max = df[y].min() - 0.5, df[y].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = pd.DataFrame({x: xx.ravel(), y: yy.ravel()})
    pred = model.predict(grid)
    codes = pd.Categorical(pred, categories=CLASSES).codes.reshape(xx.shape)
    ax.pcolormesh(xx, yy, codes, cmap=ListedColormap(COLORS), alpha=0.18, shading="auto")
    scatter_species(ax, df, x, y)
    ax.set_title(title)


def make_model(k=3, metric="euclidean", scale=True):
    """KNN, optionally with standardisation inside a pipeline (as built step by step in notebook 02)."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    return make_pipeline(StandardScaler(), knn) if scale else knn


def repeated_cv(seed=SEED):
    """The cross-validation used in notebooks 02 and 03: 5 folds, repeated 10 times."""
    from sklearn.model_selection import RepeatedStratifiedKFold

    return RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
