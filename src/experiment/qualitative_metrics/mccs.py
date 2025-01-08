import torch
from sklearn.metrics import matthews_corrcoef
import numpy as np

from qualitative_metrics.matrix_metric import MatrixMetric
from helpers import get_predicted_classes

# predicted: NUMERICAL CLASS LABELS (0, 1, 2, 3...)
# true: NUMERICAL CLASS LABELS (0, 1, 2, 3...)

# MCC is np.nan when all entries in confusion matrix are in one row or column
# Default scikit implementation returns 0 in this case
# Examples:
# 0  0  0       5  0  0
# 3  2  4       3  0  0
# 0  0  0       0  0  0
class MCC(MatrixMetric):
    def __init__(self, num_classes, device=None) -> None:
        super().__init__(device=device, num_classes=num_classes) 

    @torch.inference_mode()
    def compute(self):
        TPs, FPs, FNs, TNs = self.calculate_TPs_FPs_FNs_TNs_for_each_class()
        
        # Return np.nan if MCC can't be calculated
        for i in range(self.num_classes):
            if (TNs[i] + FPs[i] == 0) or (TNs[i] + FNs[i] == 0):
                return np.nan
        
        predicted_classes = torch.tensor(
            get_predicted_classes(self.predicted_logits, self.is_binary), device=self.device
        )
        
        return torch.tensor(
            matthews_corrcoef(
                self.true_classes.cpu().detach().numpy(),
                predicted_classes.cpu().detach().numpy(),
            )
        ).to(self.device)
        