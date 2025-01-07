import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/experiment')))
from torcheval.metrics import BinaryPrecision, MulticlassPrecision
from qualitative_metrics.precisions import NPV, MacroPrecisionForBinary

from metric_test_base import MetricTestBase
from sample_data import (
    multiclass_unbalanced_1,
    multiclass_balanced_2,
    multiclass_balanced_3,
    multiclass_balanced_4,
    multiclass_balanced_5,
    binary_unbalanced_6,
    binary_balanced_7,
    binary_8,
    binary_9,
    binary_10,
    binary_11,
    binary_12,
    binary_13
)

# TorchEval MulticlassPrecision handles cases where TP + FP = 0 as 0
# custom MacroPrecision handles cases where TP + FP = 0 as None

class TestMacroPrecision(MetricTestBase):
    def setUp(self):
        self.metric_name = "macro_precision"
        self.multiclass_metric_calculator = MulticlassPrecision(average="macro", num_classes=3)
        self.binary_metric_calculator = MacroPrecisionForBinary()
        
    def test_Compute_ShouldCalculate_WhenMulticlassUnbalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_unbalanced_1)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_balanced_2)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0TrueSamplesInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_balanced_3)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0PredictionsInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_balanced_4)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0TrueSamplesAndPredictionsInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_balanced_5)
    
        
        
    

class TestPerClassPrecision(MetricTestBase):
    def setUp(self):
        self.metric_name = "precision_per_class"
        self.multiclass_metric_calculator = MulticlassPrecision(average=None, num_classes=3)
        
    def test_Compute_ShouldCalculate_WhenMulticlassUnbalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_unbalanced_1)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_balanced_2)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0TrueSamplesInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_balanced_3)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0PredictionsInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_balanced_4)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0TrueSamplesAndPredictionsInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_balanced_5)


