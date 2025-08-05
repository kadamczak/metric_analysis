from src.experiment.helpers.variables import dataset_root_dir

dataset_names = [
    "all-in-one_sentiment_imbalanced",
    "amazon-reviews-0.25_imbalanced",
    "ceas_imbalanced",
    "colon-0.5_imbalanced",
    "fake-news_imbalanced",
    "news-sarcasm_imbalanced",
    "philippine_imbalanced",
    "santander-customer-satisfaction_imbalanced",
    "spambase_imbalanced",
    "vehicle-sensit_imbalanced",
]

binary_imbalanced_datasets = {
    name: {
        "path": dataset_root_dir / "binary" / name,
        "classification_type": "binary",
        "class_balance": "imbalanced",
        "dataset_name": name,
    }
    for name in dataset_names
}