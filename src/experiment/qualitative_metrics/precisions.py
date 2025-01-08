import torch
import numpy as np
from qualitative_metrics.matrix_metric import MatrixMetric

# TorchEval MulticlassPrecision handles cases where TP + FP = 0 as 0
# custom MacroPrecision handles cases where TP + FP = 0 as np.nan

class PrecisionMetric(MatrixMetric):
    def __init__(self, num_classes, device=None) -> None:
        super().__init__(num_classes=num_classes, device=device)
        
    @torch.inference_mode()
    def calculate_precision(self, TP, FP):   
        return TP / (TP + FP) if (TP + FP > 0) else np.nan

class MacroPrecision(PrecisionMetric):
    def __init__(self, num_classes, device=None) -> None:
        super().__init__(num_classes=num_classes, device=device)
        
    @torch.inference_mode()
    def compute(self):
        TPs, FPs, FNs, TNs = self.calculate_TPs_FPs_FNs_TNs_for_each_class()
        
        precisions = [self.calculate_precision(TPs[i], FPs[i]) for i in range(self.num_classes)]      
        calculable_precisions = [precision for precision in precisions if precision is not np.nan]
        
        if not calculable_precisions:
            return np.nan
        
        return sum(calculable_precisions) / len(calculable_precisions)



class MicroPrecision(PrecisionMetric):
    def __init__(self, num_classes, device=None) -> None:
        super().__init__(num_classes=num_classes, device=device)
        
    @torch.inference_mode()
    def compute(self):
        TPs, FPs, FNs, TNs = self.calculate_TPs_FPs_FNs_TNs_for_each_class()     
        TP_global = sum(TPs)
        FP_global = sum(FPs)
        
        return self.calculate_precision(TP_global, FP_global)

   

class PerClassPrecision(PrecisionMetric):
    def __init__(self, num_classes, device=None) -> None:
        super().__init__(num_classes=num_classes, device=device)
        
    @torch.inference_mode()
    def compute(self):
        TPs, FPs, FNs, TNs = self.calculate_TPs_FPs_FNs_TNs_for_each_class()      
        precisions = [self.calculate_precision(TPs[i], FPs[i]) for i in range(self.num_classes)]     
        return torch.tensor(precisions).to(self.device)