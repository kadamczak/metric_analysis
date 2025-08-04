from src.experiment.helpers.variables import dataset_root_dir

dataset_names = [
    #"ag-news_balanced",
    #"dbpedia-ontology_balanced",
    #"gas-drift_balanced",
    #"gtsrb-huelist_balanced",
    #"mfeat-karhunen_balanced",
    #"news-category_balanced",
    #"nyt-comments-april17_balanced",
    "stocks_balanced",
    #"usps_balanced",
    #"volkert_balanced"
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