import torch
import numpy as np
from qualitative_metrics.matrix_metric import MatrixMetric
from torcheval.metrics.metric import Metric
from sklearn.metrics import f1_score

from torcheval.metrics.functional import multiclass_f1_score


from helpers import get_predicted_classes

#===========
# TorchEval functional
#===========

# y_true - true numerical labels
# y_pred - predicted numerical labels

class F1TorchEval(Metric[torch.Tensor]):
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
            get_predicted_classes(prediction_logits, self.is_binary), device=self.device
        )

        self.true_classes = torch.cat((self.true_classes, labels))
        self.predicted_classes = torch.cat((self.predicted_classes, predicted))
        return self

    @torch.inference_mode()
    def compute(self):
        true_classes = self.true_classes.to(torch.int64)
        predicted_classes = self.predicted_classes.to(torch.int64)
        
        precision = multiclass_f1_score(
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

class F1Sklearn(Metric[torch.Tensor]):
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
            get_predicted_classes(prediction_logits, self.is_binary), device=self.device
        )

        self.true_classes = torch.cat((self.true_classes, labels))
        self.predicted_classes = torch.cat((self.predicted_classes, predicted))
        return self

    @torch.inference_mode()
    def compute(self):
        return torch.tensor(f1_score(
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

# if recall N/A (TP + FN = 0) count F1 as np.nan (fault of test set)
# if precision N/A (TP + FP = 0) F1 equals 0 (fault of model)
# if both N/A count F1 np.nan (fault of test set)

class F1Metric(MatrixMetric):
    def __init__(self, num_classes, device=None) -> None:
        super().__init__(num_classes=num_classes, device=device)
        
    @torch.inference_mode()
    def calculate_F1(self, TP, FP, FN):   
        return (2 * TP) / (2 * TP + FP + FN) if (TP + FN > 0) else np.nan



class MacroF1(F1Metric):
    def __init__(self, num_classes, device=None) -> None:
        super().__init__(num_classes=num_classes, device=device)
        
    @torch.inference_mode()
    def compute(self):
        TPs, FPs, FNs, TNs = self.calculate_TPs_FPs_FNs_TNs_for_each_class()      
        f1s = [self.calculate_F1(TPs[i], FPs[i], FNs[i]) for i in range(self.num_classes)]
        calculable_f1s = [f1 for f1 in f1s if f1 is not np.nan]
        
        if not calculable_f1s:
            return np.nan
        
        return sum(calculable_f1s) / len(calculable_f1s)



class MicroF1(F1Metric):
    def __init__(self, num_classes, device=None) -> None:
        super().__init__(num_classes=num_classes, device=device)
        
    @torch.inference_mode()
    def compute(self):
        TPs, FPs, FNs, TNs = self.calculate_TPs_FPs_FNs_TNs_for_each_class()      
        global_TP = sum(TPs)
        global_FP = sum(FPs)
        global_FN = sum(FNs)
         
        return self.calculate_F1(global_TP, global_FP, global_FN)



class PerClassF1(F1Metric):
    def __init__(self, num_classes, device=None) -> None:
        super().__init__(num_classes=num_classes, device=device)
        
    @torch.inference_mode()
    def compute(self):
        TPs, FPs, FNs, TNs = self.calculate_TPs_FPs_FNs_TNs_for_each_class()      
        f1s = [self.calculate_F1(TPs[i], FPs[i], FNs[i]) for i in range(self.num_classes)]     
        return torch.tensor(f1s).to(self.device)