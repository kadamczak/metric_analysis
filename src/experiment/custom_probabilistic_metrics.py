import torch
import torch.nn as nn
from torcheval.metrics.metric import Metric
import torch.nn.functional as F
from torcheval.metrics.functional import mean_squared_error

from helpers import get_predicted_probabilities


# predicted: PROBABILITIES
# true: 0/1 if binary, one-hot encoded if multiclass
class MSE(Metric[torch.Tensor]):
    def __init__(self, num_classes, device=None) -> None:
        super().__init__(device=device)
        self.is_binary = num_classes == 2
        self.n_classes = num_classes
        self._add_state("true_classes", torch.tensor([], device=self.device))
        self._add_state("predicted_probabilities", torch.tensor([], device=self.device))

    @torch.inference_mode()
    def update(self, prediction_logits, labels):
        true = (
            labels.float()
            if self.is_binary
            else F.one_hot(labels, num_classes=self.n_classes).float()
        )
        probabilities = (
            torch.stack(get_predicted_probabilities(prediction_logits, self.is_binary))
            .clone()
            .detach()
        )

        self.true_classes = torch.cat((self.true_classes, true))
        self.predicted_probabilities = torch.cat((self.predicted_probabilities, probabilities))
        return self

    @torch.inference_mode()
    def compute(self):
        return mean_squared_error(self.predicted_probabilities, self.true_classes)

    @torch.inference_mode()
    def merge_state(self, metrics):
        true_classes_2 = [
            self.true_classes,
        ]
        predicted_probabilities_2 = [
            self.predicted_probabilities,
        ]

        for metric in metrics:
            true_classes_2.append(metric.true_classes_2)
            predicted_probabilities_2.append(metric.predicted_probabilities_2)
            self.true_classes = torch.cat(true_classes_2)
            self.predicted_probabilities = torch.cat(predicted_probabilities_2)
        return self


# predicted: LOGITS
# true: NUMERICAL CLASS LABELS (0, 1, 2, 3...)
class LogLoss(Metric[torch.Tensor]):
    def __init__(self, num_classes, device=None) -> None:
        super().__init__(device=device)
        self.is_binary = num_classes == 2
        self.n_classes = num_classes
        self._add_state("true_classes", torch.tensor([], device=self.device))
        self._add_state("predicted_logits", torch.tensor([], device=self.device))

    @torch.inference_mode()
    def update(self, logits, numerical_labels):
        self.true_classes = torch.cat((self.true_classes, numerical_labels))
        self.predicted_logits = torch.cat((self.predicted_logits, logits))
        return self

    @torch.inference_mode()
    def compute(self):
        if self.is_binary:
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
