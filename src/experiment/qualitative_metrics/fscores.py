import torch
import numpy as np
from qualitative_metrics.matrix_metric import MatrixMetric

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