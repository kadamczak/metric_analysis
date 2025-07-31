from src.experiment.helpers.variables import dataset_root_dir

dataset_names = [
    "gas-drift_balanced",
    "gtsrb-huelist_balanced",
    "usps_balanced",
    "volkert_balanced"
]

multiclass_balanced_datasets = {
    name: {
        "path": dataset_root_dir / "multiclass" / name,
        "classification_type": "multiclass",
        "class_balance": "balanced",
        "dataset_name": name,
    }
    for name in dataset_names
}