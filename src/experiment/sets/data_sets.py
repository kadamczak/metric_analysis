from src.experiment.helpers.variables import dataset_root_dir

multilabel_datasets = {
    "bibtex_trimmed": {
        "path": dataset_root_dir / "multilabel" / "bibtex_trimmed",
        "classification_type": "multilabel",
        "class_balance": "balanced",
        "dataset_name": "bibtex_trimmed",
    },
    # "bookmarks_trimmed": {
    #     "path": dataset_root_dir / "multilabel" / "bookmarks_trimmed",
    #     "classification_type": "multilabel",
    #     "class_balance": "balanced",
    #     "dataset_name": "bookmarks_trimmed",
    #},
     "emotions_trimmed": {
         "path": dataset_root_dir / "multilabel" / "emotions_trimmed",
         "classification_type": "multilabel",
         "class_balance": "balanced",
         "dataset_name": "emotions_trimmed",
     },
    # "scene_trimmed": {
    #     "path": dataset_root_dir / "multilabel" / "scene_trimmed",
    #     "classification_type": "multilabel",
    #     "class_balance": "balanced",
    #     "dataset_name": "scene_trimmed",
    # },
    # "yeast_trimmed": {
    #     "path": dataset_root_dir / "multilabel" / "yeast_trimmed",
    #     "classification_type": "multilabel",
    #     "class_balance": "balanced",
    #     "dataset_name": "yeast_trimmed",
    # },
}