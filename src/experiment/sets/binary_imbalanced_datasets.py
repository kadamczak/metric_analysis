from src.experiment.helpers.variables import dataset_root_dir

dataset_names = [
    "colon-0.5_imbalanced",
    # "philippine_imbalanced",
    # "santander-customer-satisfaction_imbalanced",
    # "spambase_imbalanced",
    # "vehicle-sensit_imbalanced",
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