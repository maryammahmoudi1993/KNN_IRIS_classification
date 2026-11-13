"""Fast smoke tests: the data loads and a KNN model trains and predicts."""

from sklearn.neighbors import KNeighborsClassifier

from src.iris_knn import CLASSES, FEATURES, load_iris_frame, split_data


def test_data_loads():
    df = load_iris_frame()
    assert df.shape == (150, 5)
    assert list(df.columns[:4]) == FEATURES
    assert sorted(df["species"].unique()) == sorted(CLASSES)
    assert df["species"].value_counts().tolist() == [50, 50, 50]


def test_split_is_stratified_and_repeatable():
    df = load_iris_frame()
    x_train, x_test, y_train, y_test = split_data(df, test_size=0.2, seed=42)
    assert len(x_train) == 120 and len(x_test) == 30
    assert y_test.value_counts().tolist() == [10, 10, 10]
    again = split_data(df, test_size=0.2, seed=42)
    assert again[1].index.equals(x_test.index)


def test_knn_trains_and_predicts():
    df = load_iris_frame()
    x_train, x_test, y_train, y_test = split_data(df)
    model = KNeighborsClassifier(n_neighbors=5).fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    assert accuracy > 0.8


def test_helpers_for_later_notebooks():
    from src.iris_knn import make_model, repeated_cv

    df = load_iris_frame()
    x_train, x_test, y_train, y_test = split_data(df)
    model = make_model(k=3).fit(x_train, y_train)
    assert model.score(x_test, y_test) > 0.8
    assert repeated_cv().get_n_splits() == 50
