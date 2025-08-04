from src.experiment.helpers.variables import dataset_root_dir

dataset_names = [
    "bookmarks_balanced",
    "emotions_balanced",
    "imdb_balanced",
    "mediamill_balanced",
    "ng20_balanced",
    "nuswidevlad_balanced",
    "scene_balanced",
    "tmc2007500_balanced",
    "yeast_balanced",
    "yelp_balanced"
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