import numpy as np

from messages import build_model, load_data


def test_load_data_reads_tab_separated_messages(tmp_path):
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


def test_model_fits_and_returns_probabilities():
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

if __name__ == "__main__":
    main()

