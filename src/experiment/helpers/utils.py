import torch
import torch.nn.functional as F
import pandas as pd

from src.experiment.helpers.task_type import TaskType

# BINARY CLASSIFICATION (1 output neuron):
# logits -> sigmoid -> round -> predicted class indexes
# MULTI-CLASS CLASSIFICATION (N output neurons):
# logits -> argmax -> predicted classes indexes (softmax is optional but not necessary)
def get_predicted_classes_from_logits(predictions, task_type):    
    if (task_type == TaskType.BINARY):
        return [torch.round(torch.sigmoid(pred)) for pred in predictions]
    if (task_type == TaskType.MULTICLASS):
        return [torch.argmax(pred) for pred in predictions]
    if (task_type == TaskType.MULTILABEL):
        result = (predictions > 0).int()
        return result

def get_predicted_classes_from_probabilities(predictions, task_type):
    if (task_type == TaskType.BINARY):
        return [torch.round(pred) for pred in predictions]
    if (task_type == TaskType.MULTICLASS):
        return [torch.argmax(pred) for pred in predictions]
    if (task_type == TaskType.MULTILABEL):
        result = (predictions > 0.5).int()
        return result

def get_predicted_probabilities(predictions, task_type):
    if (task_type == TaskType.BINARY):
        return [torch.sigmoid(pred) for pred in predictions]
    if (task_type == TaskType.MULTICLASS):
        return [F.softmax(pred, dim=0) for pred in predictions]
    if (task_type == TaskType.MULTILABEL):
        return [torch.sigmoid(pred) for pred in predictions]


def get_binary_labels_for_class(labels, class_index):
    return [1 if label == class_index else 0 for label in labels]


def get_sorted_class_percentages(y):
    y_numeric = y.apply(pd.to_numeric)
    class_percentages = y_numeric.sum() / len(y_numeric) * 100
    class_percentages_sorted = class_percentages.sort_values(ascending=False)
    return class_percentages_sorted

def get_cardinality(y):
    cardinality = y.apply(pd.to_numeric).sum(axis=1).mean()
    return cardinality