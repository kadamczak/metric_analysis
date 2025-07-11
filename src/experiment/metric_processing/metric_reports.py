import json

from src.experiment.metric_processing.metric_calc import create_metric_dictionary
from src.experiment.metric_processing.metric_display import draw_metrics

# file creation order: 1.json, 2.json, 3.json, ...
# def create_next_report_file_name(output_dir_path):
#     existing_files = list(output_dir_path.glob("*.json"))
#     if existing_files:
#         existing_numbers = [int(f.stem) for f in existing_files if f.stem.isdigit()]
#         next_number = max(existing_numbers) + 1
#     else:
#         next_number = 1
#     return f"{next_number}.json"


def write_results_report_to_new_file(output_dir_path, experiment_info, fold_info, results):
    output_file = output_dir_path / "report.json"
    output_dir_path.mkdir(parents=True, exist_ok=True)

    full_dict = dict()
    full_dict.update(
        {
            "classification type": experiment_info.classification_type,
            "class balance": experiment_info.class_balance,
            
            "dataset name": experiment_info.dataset_name,
            "class names": experiment_info.class_names,
            
            "model name": experiment_info.model_name,
            
            "index": experiment_info.index,
            "cv fold": experiment_info.cv_fold,
            "fold info": {
                "train distribution": fold_info.train_distribution,
                "test distribution": fold_info.test_distribution,
                "train distribution pct": fold_info.train_distribution_pct,
                "test distribution pct": fold_info.test_distribution_pct,
            },
            
            "metrics": create_metric_dictionary(results, experiment_info.class_names),
        }
    )

    with open(output_file, "w") as f:
        f.write(json.dumps(full_dict, indent=4))
        draw_metrics(results, experiment_info.class_names, output=f)


class experiment_info:
    def __init__(
        self, model_name, dataset_name, classification_type, class_balance, index, cv_fold, class_names
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.classification_type = classification_type
        self.class_balance = class_balance
        self.index = index
        self.cv_fold = cv_fold
        self.class_names = class_names


class fold_info:
    def __init__(
        self, train_distribution, test_distribution, train_distribution_pct, test_distribution_pct
    ):
        self.train_distribution = train_distribution
        self.test_distribution = test_distribution
        self.train_distribution_pct = train_distribution_pct
        self.test_distribution_pct = test_distribution_pct
