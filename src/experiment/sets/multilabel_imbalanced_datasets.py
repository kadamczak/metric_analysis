from src.experiment.helpers.variables import dataset_root_dir

dataset_names = [
    "bookmarks_imbalanced",
    "emotions_imbalanced",
    "imdb_imbalanced",
    "mediamill_imbalanced",
    "ng20_imbalanced",
    "nuswidevlad_imbalanced",
    "scene_imbalanced",
    "tmc2007500_imbalanced",
    "yeast_imbalanced",
    "yelp_imbalanced"
]

multilabel_imbalanced_datasets = {
    name: {
        "path": dataset_root_dir / "multilabel" / name,
        "classification_type": "multilabel",
        "class_balance": "imbalanced",
        "dataset_name": name,
    }
    for name in dataset_names
}