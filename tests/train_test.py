# tests/test_training_script.py

import sys
import tempfile
import pandas as pd
import tensorflow as tf
import pytest
import types
import runpy


def test_training_script_runs(monkeypatch):
    """
    Smoke test: Ensure that the training script runs end-to-end
    with dummy data, builds the model, compiles, and calls fit().
    """

    # --- Create dummy CSV file ---
    tmp_csv = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    df = pd.DataFrame({"smiles": ["CCO", "CCN", "CCC"]})
    df.to_csv(tmp_csv.name, index=False)

    # --- Fake args ---
    test_args = [
        "script_name",
        "--batch_size", "2",
        "--df_path", tmp_csv.name,
        "--embedding_dim", "16",
        "--num_heads", "2",
        "--latent_dim", "32",
        "--max_seq_length", "10",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    # --- Monkeypatch model.fit to avoid real training ---
    fit_called = {}

    def fake_fit(self, *args, **kwargs):
        fit_called["called"] = True
        # Return a dummy history-like object
        return types.SimpleNamespace(history={})

    monkeypatch.setattr(tf.keras.Model, "fit", fake_fit)

    # --- Monkeypatch save/load ---
    monkeypatch.setattr(tf.keras.Model, "save", lambda self, path: None)
    monkeypatch.setattr(tf.keras.Model, "load_weights", lambda self, path: None)

    # --- Run the script as if `python -m src.train` was called ---
    runpy.run_module("src.train", run_name="__main__")

    # --- Assertions ---
    assert "called" in fit_called, "Model.fit was not called"
