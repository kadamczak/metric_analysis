import torch
import numpy as np
from src.experiment.metrics.rank.rocauc import drawn_ROC_list


def reset_metrics(metrics):
    [metric.reset() for metric in metrics.values()]


# numerical labels: 0, 1, 2, 3, ...
def update_metrics_using_logits(metrics, logits, numerical_labels):
    numerical_labels = numerical_labels.to(torch.int64)
    [metric.update(logits, numerical_labels) for metric in metrics.values()]
    
def update_metrics_using_probabilities(metrics, probabilities, labels):
    probas_tensor = torch.tensor(probabilities, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    [metric.update(probas_tensor, labels_tensor) for metric in metrics.values()]
    


def compute_metrics(metrics):
    return {name: metric.compute() for name, metric in metrics.items()}


def create_metric_dictionary(metrics, class_names):
    metric_dict = dict()
    for name, metric in metrics.items():
        if name in drawn_ROC_list or name == "confusion_matrix":
            continue

        if metric is np.nan:
            metric_dict[name] = np.nan
            continue
        
        value = metric.tolist()
        if isinstance(value, list):  # one number result for each class
            metric_dict[name] = {class_names[i]: value[i] for i in range(len(value))}
        else:  # one number result for all classes
            metric_dict[name] = value

    return metric_dict
