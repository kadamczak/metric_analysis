import torch
import torch.nn as nn
from torcheval.metrics.metric import Metric
import torch.nn.functional as F

from src.experiment.helpers.task_type import TaskType


# predicted: PROBABILITIES
# true: NUMERICAL CLASS LABELS (0, 1, 2...) or multi-label one-hot/binary
class LogLoss(Metric[torch.Tensor]):
    def __init__(self, num_classes, task_type, device=None) -> None:
        super().__init__(device=device)
        self.n_classes = num_classes
        self.task_type = task_type
        self._add_state("true_classes", torch.tensor([], device=self.device))
        self._add_state("predicted_probs", torch.tensor([], device=self.device))

    @torch.inference_mode()
    def update(self, probs, numerical_labels):
        self.true_classes = torch.cat((self.true_classes, numerical_labels))
        self.predicted_probs = torch.cat((self.predicted_probs, probs))
        return self

    @torch.inference_mode()
    def compute(self):
        if self.task_type in [TaskType.BINARY, TaskType.MULTILABEL]:
            loss_fn = nn.BCELoss()
            return loss_fn(self.predicted_probs, self.true_classes)

        elif self.task_type == TaskType.MULTICLASS:
            # Convert probabilities to log-probabilities
            log_probs = torch.log(self.predicted_probs + 1e-9)  # avoid log(0)
            numerical_labels = self.true_classes.long().to(self.device)
            loss_fn = nn.NLLLoss()
            return loss_fn(log_probs, numerical_labels)

        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    @torch.inference_mode()
    def merge_state(self, metrics):
        true_classes_list = [self.true_classes]
        predicted_probs_list = [self.predicted_probs]

        for metric in metrics:
            true_classes_list.append(metric.true_classes)
            predicted_probs_list.append(metric.predicted_probs)

        self.true_classes = torch.cat(true_classes_list)
        self.predicted_probs = torch.cat(predicted_probs_list)

        return self
