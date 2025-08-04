from src.experiment.helpers.variables import dataset_root_dir

dataset_names = [
    #"ag-news_imbalanced",
    #"dbpedia-ontology_imbalanced",
    #"gas-drift_imbalanced",
    #"gtsrb-huelist_imbalanced",
    "irish-times_imbalanced",
    #"mfeat-karhunen_imbalanced",
    #"news-category_imbalanced",
    #"nyt-comments-april17_imbalanced",
    #"usps_imbalanced",
    #"volkert_imbalanced"
]

multiclass_imbalanced_datasets = {
    name: {
        "path": dataset_root_dir / "multiclass" / name,
        "classification_type": "multiclass",
        "class_balance": "imbalanced",
        "dataset_name": name,
    }
    for name in dataset_names
}