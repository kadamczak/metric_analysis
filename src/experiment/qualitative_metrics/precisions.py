import torch
import numpy as np
from qualitative_metrics.matrix_metric import MatrixMetric


class MacroPrecision(MatrixMetric):
    def __init__(self, num_classes, device=None) -> None:
        super().__init__(num_classes=num_classes, device=device)
        
    @torch.inference_mode()
    def compute(self):
        TPs, FPs, FNs, TNs = self.calculate_TPs_FPs_FNs_TNs_for_each_class()
        
        precisions = [TPs[i] / (TPs[i] + FPs[i]) if (TPs[i] + FPs[i] > 0) else np.nan for i in range(self.num_classes)]      
        calculable_precisions = [precision for precision in precisions if precision is not np.nan]
        
        if not calculable_precisions:
            return np.nan
        
        return sum(calculable_precisions) / len(calculable_precisions)



class MicroPrecision(MatrixMetric):
    def __init__(self, num_classes, device=None) -> None:
        super().__init__(num_classes=num_classes, device=device)
        
    @torch.inference_mode()
    def compute(self):
        TPs, FPs, FNs, TNs = self.calculate_TPs_FPs_FNs_TNs_for_each_class()  
        
        TP_global = sum(TPs)
        FP_global = sum(FPs)
        
        if TP_global + FP_global == 0:
            return np.nan
            
        return TP_global / (TP_global + FP_global)

   

class PerClassPrecision(MatrixMetric):
    def __init__(self, num_classes, device=None) -> None:
        super().__init__(num_classes=num_classes, device=device)
        
    @torch.inference_mode()
    def compute(self):
        TPs, FPs, FNs, TNs = self.calculate_TPs_FPs_FNs_TNs_for_each_class()      
        precisions = [TPs[i] / (TPs[i] + FPs[i]) if (TPs[i] + FPs[i] > 0) else np.nan for i in range(self.num_classes)]     
        return torch.tensor(precisions).to(self.device)