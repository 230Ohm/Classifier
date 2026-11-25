import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

data = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
)

train_messages, test_messages, train_labels, test_labels = train_test_split(
    data["message"],
    data["label"],
    test_size=0.2,
    random_state=42,
    stratify=data["label"],
)

model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=2),
    SelectPercentile(score_func=chi2, percentile=20),
    LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
    ),
)

model.fit(train_messages, train_labels)

spam_index = list(model.classes_).index("spam")
spam_probabilities = model.predict_proba(test_messages)[:, spam_index]

threshold = 0.5
predictions = [
    "spam" if probability >= threshold else "ham"
    for probability in spam_probabilities
]

print(classification_report(test_labels, predictions))

print(model.predict(["Congratulations, you won a prize!"]))