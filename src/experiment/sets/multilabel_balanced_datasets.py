from src.experiment.helpers.variables import dataset_root_dir

dataset_names = [
    "bibtex_balanced", "bookmarks_balanced", "cal500_balanced", "emotions_balanced"
]

multilabel_balanced_datasets = {
    name: {
        "path": dataset_root_dir / "multilabel" / name,
        "classification_type": "multilabel",
        "class_balance": "balanced",
        "dataset_name": name,
    }
    for name in dataset_names
}