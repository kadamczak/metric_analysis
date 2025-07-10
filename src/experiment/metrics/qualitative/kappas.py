import torch
from torcheval.metrics.metric import Metric
from src.experiment.helpers.task_type import TaskType
from src.experiment.metrics.qualitative.matrix_metric import MatrixMetric
from sklearn.metrics import cohen_kappa_score

from src.experiment.helpers.utils import get_predicted_classes_from_probabilities

# predicted: NUMERICAL CLASS LABELS (0, 1, 2, 3...)
# true: NUMERICAL CLASS LABELS (0, 1, 2, 3...)

# Both binary and multiclass kappa are np.nan when every entry in confusion matrix is in one diagonal cell
# Default scikit implementation of kappa also behaves this way
# Examples:
# 0  0  0          6  0    
# 0  6  0          0  0
# 0  0  0
class BinaryCohenKappa(Metric[torch.Tensor]):
    def __init__(self, device=None) -> None:
        super().__init__(device=device)
        self.task_type = TaskType.BINARY
        self._add_state("true_classes", torch.tensor([], device=self.device))
        self._add_state("predicted_classes", torch.tensor([], device=self.device))

    @torch.inference_mode()
    def update(self, predicted_probabilities, labels):
        predicted = torch.tensor(
            get_predicted_classes_from_probabilities(predicted_probabilities, self.task_type), device=self.device
        )

        self.true_classes = torch.cat((self.true_classes, labels))
        self.predicted_classes = torch.cat((self.predicted_classes, predicted))
        return self

    @torch.inference_mode()
    def compute(self):
        return torch.tensor(cohen_kappa_score(
            self.true_classes.cpu().detach().numpy(),
            self.predicted_classes.cpu().detach().numpy(),
        )).to(self.device)

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
    
 
     
class MulticlassCohenKappa(MatrixMetric):
    def __init__(self, num_classes, device=None) -> None:
        super().__init__(num_classes=num_classes, device=device, task_type=TaskType.MULTICLASS)

    @torch.inference_mode()
    def compute(self):
        matrix = self.calculate_matrix()

        n = matrix.sum()
        matrix_diag_sum = matrix.diag().sum()
        matrix_cols = [matrix[:, i].sum() for i in range(self.num_classes)]
        matrix_rows = [matrix[i, :].sum() for i in range(self.num_classes)]

        p0 = matrix_diag_sum / n
        pe = sum([matrix_cols[i] * matrix_rows[i] for i in range(self.num_classes)]) / (
            n * n
        )

        cohens_kappa = (p0 - pe) / (1 - pe)
        return torch.tensor(cohens_kappa).to(self.device)
