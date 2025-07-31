import torch
from torcheval.metrics.metric import Metric
import torch.nn.functional as F
from torcheval.metrics.functional import mean_squared_error

from src.experiment.helpers.utils import get_predicted_probabilities
from src.experiment.helpers.task_type import TaskType


# predicted: PROBABILITIES
# true: 0/1 if binary, one-hot encoded if multiclass
class MSE(Metric[torch.Tensor]):
    def __init__(self, num_classes, task_type, device=None) -> None:
        super().__init__(device=device)
        self.task_type = task_type
        self.n_classes = num_classes
        self._add_state("true_classes", torch.tensor([], device=self.device))
        self._add_state("predicted_probabilities", torch.tensor([], device=self.device))

    @torch.inference_mode()
    def update(self, probabilities, labels):
        
        # print(probabilities)
        # print(probabilities.dtype)
        
        # # convert labels to torch.int32
        # labels = labels.to(torch.int32)
        
        # print(labels)
        # print(labels.dtype)

        # print(self.n_classes)

        true = (
            labels.float()
            if self.task_type == TaskType.BINARY or self.task_type == TaskType.MULTILABEL
            else F.one_hot(labels.to(torch.long), num_classes=self.n_classes).float()
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

