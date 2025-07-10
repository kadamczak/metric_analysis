import json

from experiment.metric_processing.metric_calc import create_metric_dictionary
from experiment.metric_processing.metric_display import draw_metrics

# file creation order: 1.json, 2.json, 3.json, ...
def create_next_report_file_name(output_dir_path):
    existing_files = list(output_dir_path.glob("*.json"))
    if existing_files:
        existing_numbers = [int(f.stem) for f in existing_files if f.stem.isdigit()]
        next_number = max(existing_numbers) + 1
    else:
        next_number = 1
    return f"{next_number}.json"


def write_results_report_to_new_file(output_dir_path, experiment_info, epochs, results):
    output_file = output_dir_path / create_next_report_file_name(output_dir_path)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    full_dict = dict()
    full_dict.update(
        {
            "model name": experiment_info.model_name,
            "dataset name": experiment_info.dataset_name,
            "classification type": experiment_info.classification_type,
            "test set": experiment_info.test_set,
            "index": experiment_info.index,
            "cv fold": experiment_info.cv_fold,
            "epochs": epochs,
            "metrics": create_metric_dictionary(results, experiment_info.class_names),
        }
    )

    with open(output_file, "w") as f:
        f.write(json.dumps(full_dict, indent=4))
        draw_metrics(results, experiment_info.class_names, output=f)


class experiment_info:
    def __init__(
        self, model_name, dataset_name, classification_type, test_set, index, cv_fold, class_names
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.classification_type = classification_type
        self.test_set = test_set
        self.index = index
        self.cv_fold = cv_fold
        self.class_names = class_names
