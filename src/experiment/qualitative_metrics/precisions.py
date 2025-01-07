import torch
import numpy as np
from qualitative_metrics.matrix_metric import MatrixMetric

#    TP
# --------
# TP + FP
class NPV(MatrixMetric):
    def __init__(self, device=None) -> None:
        super().__init__(num_classes=2, device=device)
        
    @torch.inference_mode()
    def compute(self):  
        TP_neg, FP_neg, FN_neg, TN_neg = self.calculate_TPs_FPs_FNs_TNs_for_class(self.calculate_matrix(), 0)
        
        if (TP_neg + FP_neg == 0):
            return np.nan
        
        return TP_neg / (TP_neg + FP_neg)
    
    
# for models with 1 logit output
class MacroPrecisionForBinary(MatrixMetric):
    def __init__(self, device=None) -> None:
        super().__init__(num_classes=2, device=device)
        
    @torch.inference_mode()
    def compute(self):
        TPs, FPs, FNs, TNs = self.calculate_TPs_FPs_FNs_TNs_for_each_class()
        
        precision_for_positive = (TPs[1] / (TPs[1] + FPs[1])) if (TPs[1] + FPs[1] > 0) else np.nan
        precision_for_negative = (TPs[0] / (TPs[0] + FPs[0])) if (TPs[0] + FPs[0] > 0) else np.nan
        
        if (precision_for_positive is np.nan) and (precision_for_negative is np.nan):
            return np.nan
        elif precision_for_positive is np.nan:
            return precision_for_negative
        elif precision_for_negative is np.nan:
            return precision_for_positive
        else:
            return (precision_for_positive + precision_for_negative) / 2