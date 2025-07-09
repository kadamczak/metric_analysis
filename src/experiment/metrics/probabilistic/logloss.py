import torch
import torch.nn as nn
from torcheval.metrics.metric import Metric
import torch.nn.functional as F

from src.experiment.helpers.task_type import TaskType


# predicted: LOGITS
# true: NUMERICAL CLASS LABELS (0, 1, 2, 3...)
class LogLoss(Metric[torch.Tensor]):
    def __init__(self, num_classes, task_type, device=None) -> None:
        super().__init__(device=device)
        self.n_classes = num_classes
        self.task_type = task_type
        self._add_state("true_classes", torch.tensor([], device=self.device))
        self._add_state("predicted_logits", torch.tensor([], device=self.device))

    @torch.inference_mode()
    def update(self, logits, numerical_labels):
        self.true_classes = torch.cat((self.true_classes, numerical_labels))
        self.predicted_logits = torch.cat((self.predicted_logits, logits))
        return self

    @torch.inference_mode()
    def compute(self):
        if self.task_type == TaskType.BINARY or self.task_type == TaskType.MULTILABEL:
            loss = nn.BCEWithLogitsLoss()
            numerical_labels = self.true_classes
        else:
            loss = nn.CrossEntropyLoss(reduction="mean")
            numerical_labels = self.true_classes.type(torch.LongTensor).to(self.device)

        return loss(self.predicted_logits, numerical_labels)

    @torch.inference_mode()
    def merge_state(self, metrics):
        true_classes_2 = [
            self.true_classes,
        ]
        predicted_logits_2 = [
            self.predicted_logits,
        ]

        for metric in metrics:
            true_classes_2.append(metric.true_classes_2)
            predicted_logits_2.append(metric.predicted_logits_2)
            self.true_classes = torch.cat(true_classes_2)
            self.predicted_logits = torch.cat(predicted_logits_2)
        return self
