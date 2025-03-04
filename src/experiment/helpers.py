import torch
import torch.nn.functional as F


# BINARY CLASSIFICATION (1 output neuron):
# logits -> sigmoid -> round -> predicted class indexes
# MULTI-CLASS CLASSIFICATION (N output neurons):
# logits -> argmax -> predicted classes indexes (softmax is optional but not necessary)
def get_predicted_classes(predictions, is_binary):  # threshold for binary = 0.5 probability
    return (
        [torch.round(torch.sigmoid(pred)) for pred in predictions]
        if is_binary
        else [torch.argmax(pred) for pred in predictions]
    )


def get_predicted_probabilities(predictions, is_binary):
    return (
        [torch.sigmoid(pred) for pred in predictions]
        if is_binary
        else [F.softmax(pred, dim=0) for pred in predictions]
    )


def get_binary_labels_for_class(labels, class_index):
    return [1 if label == class_index else 0 for label in labels]
