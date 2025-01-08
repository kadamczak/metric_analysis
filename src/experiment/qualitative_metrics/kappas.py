import torch
from torcheval.metrics.metric import Metric
from qualitative_metrics.matrix_metric import MatrixMetric
from sklearn.metrics import cohen_kappa_score

from helpers import get_predicted_classes

# predicted: NUMERICAL CLASS LABELS (0, 1, 2, 3...)
# true: NUMERICAL CLASS LABELS (0, 1, 2, 3...)

# Both binary and multiclass kappa are np.nan when every entry in confusion matrix is in one diagonal cell
# Examples:
# 0  0  0          6  0    
# 0  6  0          0  0
# 0  0  0
class BinaryCohenKappa(Metric[torch.Tensor]):
    def __init__(self, device=None) -> None:
        super().__init__(device=device)
        self.is_binary = True
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
        super().__init__(num_classes=num_classes, device=device)

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
