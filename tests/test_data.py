from src.data_preprocessing import preprocess_data

def test_preprocess_data():
    X_train, X_test, y_train, y_test = preprocess_data()
    assert X_train.shape[0] == 120
    assert X_test.shape[0] == 30
    assert len(y_train) == 120
    assert len(y_test) == 30
