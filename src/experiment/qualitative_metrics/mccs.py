import torch
from torcheval.metrics.metric import Metric
from sklearn.metrics import matthews_corrcoef

from helpers import get_predicted_classes

# predicted: NUMERICAL CLASS LABELS (0, 1, 2, 3...)
# true: NUMERICAL CLASS LABELS (0, 1, 2, 3...)
class MCC(Metric[torch.Tensor]):
    def __init__(self, is_binary, device=None) -> None:
        super().__init__(device=device)
        self.is_binary = is_binary
        self._add_state("true_classes", torch.tensor([], device=self.device))
        self._add_state("predicted_classes", torch.tensor([], device=self.device))

    @torch.inference_mode()
    def update(self, prediction_logits, labels):
        predicted = torch.tensor(
            get_predicted_classes(prediction_logits, self.is_binary), device=self.device
        )

        self.true_classes = torch.cat((self.true_classes, labels))
        self.predicted_classes = torch.cat((self.predicted_classes, predicted))
        return self

    @torch.inference_mode()
    def compute(self):
        return torch.tensor(
            matthews_corrcoef(
                self.true_classes.cpu().detach().numpy(),
                self.predicted_classes.cpu().detach().numpy(),
            )
        ).to(self.device)

    @torch.inference_mode()
    def merge_state(self, metrics):
        true_classes_2 = [
            self.true_classes,
        ]
        predicted_classes_2 = [
            self.predicted_classes,
        ]

        for metric in metrics:
            true_classes_2.append(metric.true_classes_2)
            predicted_classes_2.append(metric.predicted_classes_2)
            self.true_classes = torch.cat(true_classes_2)
            self.predicted_classes = torch.cat(predicted_classes_2)
        return self
