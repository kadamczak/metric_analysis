import torch
import numpy as np
from src.experiment.metrics.qualitative.matrix_metric import MatrixMetric
from torcheval.metrics.metric import Metric
from sklearn.metrics import precision_score

from src.experiment.helpers.utils import get_predicted_classes_from_logits
from torcheval.metrics.functional import multiclass_precision


#===========
# TorchEval functional
#===========

# y_true - true numerical labels
# y_pred - predicted numerical labels

class PrecisionTorchEval(Metric[torch.Tensor]):
    def __init__(self, average, num_classes, device=None) -> None:
        super().__init__(device=device)
        self.average = average
        self.is_binary = num_classes == 2
        self.num_classes = num_classes
        self._add_state("true_classes", torch.tensor([], device=self.device))
        self._add_state("predicted_classes", torch.tensor([], device=self.device))

    @torch.inference_mode()
    def update(self, prediction_logits, labels):
        predicted = torch.tensor(
            get_predicted_classes_from_logits(prediction_logits, self.is_binary), device=self.device
        )

        self.true_classes = torch.cat((self.true_classes, labels))
        self.predicted_classes = torch.cat((self.predicted_classes, predicted))
        return self

    @torch.inference_mode()
    def compute(self):
        true_classes = self.true_classes.to(torch.int64)
        predicted_classes = self.predicted_classes.to(torch.int64)
        
        precision = multiclass_precision(
            predicted_classes,
            true_classes,
            average=self.average,
            num_classes=self.num_classes
        )
        
        return torch.tensor(precision).to(self.device)

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




#===========
# Sklearn
#===========

# y_true - true numerical labels
# y_pred - predicted numerical labels

class PrecisionSklearn(Metric[torch.Tensor]):
    def __init__(self, average, num_classes, zero_division, device=None) -> None:
        super().__init__(device=device)
        self.average = average
        self.is_binary = num_classes == 2
        self.zero_division = zero_division
        self._add_state("true_classes", torch.tensor([], device=self.device))
        self._add_state("predicted_classes", torch.tensor([], device=self.device))

    @torch.inference_mode()
    def update(self, prediction_logits, labels):
        predicted = torch.tensor(
            get_predicted_classes_from_logits(prediction_logits, self.is_binary), device=self.device
        )

        self.true_classes = torch.cat((self.true_classes, labels))
        self.predicted_classes = torch.cat((self.predicted_classes, predicted))
        return self

    @torch.inference_mode()
    def compute(self):
        return torch.tensor(precision_score(
            self.true_classes.cpu().detach().numpy(),
            self.predicted_classes.cpu().detach().numpy(),
            average=self.average,
            zero_division=self.zero_division
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



#===========
# Custom
#===========

class PrecisionMetric(MatrixMetric):
    def __init__(self, num_classes, task_type, device=None) -> None:
        super().__init__(num_classes=num_classes, task_type=task_type, device=device)
        
    @torch.inference_mode()
    def calculate_precision(self, TP, FP):   
        return TP / (TP + FP) if (TP + FP > 0) else np.nan

class MacroPrecision(PrecisionMetric):
    def __init__(self, num_classes, task_type, device=None) -> None:
        super().__init__(num_classes=num_classes, task_type=task_type, device=device)
        
    @torch.inference_mode()
    def compute(self):
        TPs, FPs, FNs, TNs = self.calculate_TPs_FPs_FNs_TNs_for_each_class()
        
        precisions = [self.calculate_precision(TPs[i], FPs[i]) for i in range(self.num_classes)]      
        calculable_precisions = [precision for precision in precisions if precision is not np.nan]
        
        if not calculable_precisions:
            return np.nan
        
        return sum(calculable_precisions) / len(calculable_precisions)



class MicroPrecision(PrecisionMetric):
    def __init__(self, num_classes, task_type, device=None) -> None:
        super().__init__(num_classes=num_classes, task_type=task_type, device=device)
        
    @torch.inference_mode()
    def compute(self):
        TPs, FPs, FNs, TNs = self.calculate_TPs_FPs_FNs_TNs_for_each_class()     
        TP_global = sum(TPs)
        FP_global = sum(FPs)
        
        return self.calculate_precision(TP_global, FP_global)

   

class PerClassPrecision(PrecisionMetric):
    def __init__(self, num_classes, task_type, device=None) -> None:
        super().__init__(num_classes=num_classes, task_type=task_type, device=device)
        
    @torch.inference_mode()
    def compute(self):
        TPs, FPs, FNs, TNs = self.calculate_TPs_FPs_FNs_TNs_for_each_class()      
        precisions = [self.calculate_precision(TPs[i], FPs[i]) for i in range(self.num_classes)]     
        return torch.tensor(precisions).to(self.device)