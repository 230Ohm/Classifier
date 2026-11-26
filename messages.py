"""Train and evaluate a message classifier.""" 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import logging


def load_data(path="SMSSpamCollection"):
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "message"],
    )


def build_model():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2),
        SelectPercentile(score_func=chi2, percentile=20),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
        ),
    )


def main():
    data = load_data()

    train_messages, test_messages, train_labels, test_labels = train_test_split(
        data["message"],
        data["label"],
        test_size=0.2,
        random_state=42,
        stratify=data["label"],
    )

    model = build_model()
    model.fit(train_messages, train_labels)

    spam_index = list(model.classes_).index("spam")

    spam_probabilities = model.predict_proba(test_messages)[:, spam_index]

    THRESHOLD = 0.5
    predictions = [
        "spam" if probability >= THRESHOLD else "ham"
        for probability in spam_probabilities
    ]

    logging.info(classification_report(test_labels, predictions))
    logging.info(model.predict(["Congratulations, you won a prize!"]))


if __name__ == "__main__":
    main()