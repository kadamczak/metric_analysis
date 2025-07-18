from src.experiment.helpers.variables import dataset_root_dir

multilabel_datasets = {
     "bibtex_trimmed": {
         "path": dataset_root_dir / "multilabel" / "bibtex_trimmed",
         "classification_type": "multilabel",
         "class_balance": "balanced",
         "dataset_name": "bibtex_trimmed",
     },
     "bookmarks_trimmed": {
         "path": dataset_root_dir / "multilabel" / "bookmarks_trimmed",
         "classification_type": "multilabel",
         "class_balance": "balanced",
         "dataset_name": "bookmarks_trimmed",
    },
     "corel16k009_trimmed": {
         "path": dataset_root_dir / "multilabel" / "corel16k009_trimmed",
         "classification_type": "multilabel",
         "class_balance": "balanced",
         "dataset_name": "corel16k009_trimmed",
     },
     "emotions_trimmed": {
         "path": dataset_root_dir / "multilabel" / "emotions_trimmed",
         "classification_type": "multilabel",
         "class_balance": "balanced",
         "dataset_name": "emotions_trimmed",
     },
     "mediamill_trimmed": {
         "path": dataset_root_dir / "multilabel" / "mediamill_trimmed",
         "classification_type": "multilabel",
         "class_balance": "balanced",
         "dataset_name": "mediamill_trimmed",
     },
    "scene_trimmed": {
        "path": dataset_root_dir / "multilabel" / "scene_trimmed",
        "classification_type": "multilabel",
        "class_balance": "balanced",
        "dataset_name": "scene_trimmed",
    },
    "slashdot_trimmed": {
         "path": dataset_root_dir / "multilabel" / "slashdot_trimmed",
         "classification_type": "multilabel",
         "class_balance": "balanced",
         "dataset_name": "slashdot_trimmed",
     },
    "stackexcs_trimmed": {
         "path": dataset_root_dir / "multilabel" / "stackexcs_trimmed",
         "classification_type": "multilabel",
         "class_balance": "balanced",
         "dataset_name": "stackexcs_trimmed",
     },
    "yeast_trimmed": {
        "path": dataset_root_dir / "multilabel" / "yeast_trimmed",
        "classification_type": "multilabel",
        "class_balance": "balanced",
        "dataset_name": "yeast_trimmed",
    },
}