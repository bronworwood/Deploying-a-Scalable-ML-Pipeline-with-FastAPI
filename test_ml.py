import pytest
import numpy as np
import os
import pandas as pd
from ml.model import (train_model, compute_model_metrics, inference)
from sklearn.ensemble import RandomForestClassifier
# TODO: add necessary import

@pytest.fixture
def sample_data():
    """Fixture to provide sample training data."""
    # Create sample data
    X = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y = np.array([0, 1, 1, 0])
    return X, y

@pytest.fixture
def trained_model(sample_data):
    """Fixture to provide a trained model."""
    X, y = sample_data
    model = train_model(X, y)
    return model


# TODO: implement the first test. Change the function name and input as needed
def test_train_model_returns_randomforest(sample_data):
    """
    Ensure train_model returns a RandomForestClassifier.
    """
    X, y = sample_data
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "predict")

    pass


# TODO: implement the second test. Change the function name and input as needed
def test_inference_output_shape(trained_model, sample_data):
    """
    Check inference function returns numpy array with correct shape.
    """
    X, y = sample_data
    preds = inference(trained_model, X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == y.shape

    pass


# TODO: implement the third test. Change the function name and input as needed
def test_compute_model_metrics_values():
    """
    Verify precision, recall and f1 metrics are computed correctly.
    """
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    # Expected: precision=1.0, recall=0.5, f1=0.6667
    assert pytest.approx(precision, 0.01) == 1.0
    assert pytest.approx(recall, 0.01) == 0.5
    assert pytest.approx(fbeta, 0.01) == 0.6667

    pass
