"""Test SMS data loading and classifier predictions."""

import numpy as np

from messages import build_model, load_data


def test_load_seperated_messages(tmp_path):
    """Load labels and messages from a tab-separated file."""
    dataset = tmp_path / "messages.tsv"
    dataset.write_text(
        "ham\tSee you after class\n"
        "spam\tClaim your free prize now\n",
        encoding="utf-8",
    )

    data = load_data(dataset)

    assert list(data.columns) == ["label", "message"]
    assert data["label"].tolist() == ["ham", "spam"]
    assert len(data) == 2


def fits_and_returns():
    """Verify that predicted probabilities are finite and sum to one."""
    messages = [
        "meet me after class",
        "are we meeting after class",
        "call me when you get home",
        "see you after school",
        "claim your free prize now",
        "claim this free prize today",
        "win cash with free entry",
        "claim your cash prize now",
    ]
    labels = ["ham", "ham", "ham", "ham", "spam", "spam", "spam", "spam"]

    model = build_model()
    model.fit(messages, labels)

    probabilities = model.predict_proba(
        ["see you after class", "claim your free prize"]
    )

    assert probabilities.shape == (2, 2)
    assert np.all(np.isfinite(probabilities))
    np.testing.assert_allclose(probabilities.sum(axis=1), [1.0, 1.0])
