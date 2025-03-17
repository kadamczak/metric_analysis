import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/experiment')))
from qualitative_metrics.precisions import MacroPrecision, MicroPrecision, PerClassPrecision

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

from torcheval.metrics import BinaryPrecision, MulticlassPrecision

#============
# TorchEval
#============

# MACRO
# no predicted samples (FP + TP = 0), but some true samples -> gets 0
# no no predicted samples, no true samples -> value IGNORED in macro calculation

# MICRO is okay

# PER CLASS
# no predicted samples (FP + TP = 0), but some true samples -> gets 0
# no no predicted samples, no true samples -> value NOT ignored and is displayed as 0 in the array


class TestMacroPrecision(MetricTestBase):
    def setUp(self):
        self.metric_name = "macro_precision"
        #self.multiclass_metric_calculator = MulticlassPrecision(average="macro", num_classes=3)
        #self.binary_metric_calculator = BinaryPrecision(threshold=0.0)
        self.multiclass_metric_calculator = MacroPrecision(num_classes=3)
        self.binary_metric_calculator = MacroPrecision(num_classes=2)
        
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
        
    def test_Compute_ShouldCalculate_WhenBinaryUnbalanced(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_unbalanced_6)

    def test_Compute_ShouldCalculate_WhenBinaryBalanced(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_balanced_7)

    def test_Compute_ShouldCalculate_WhenBinary_When0TrueSamplesInPositiveClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_8)
        
    def test_Compute_ShouldCalculate_WhenBinary_When0TrueSamplesInNegativeClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_9)

    def test_Compute_ShouldCalculate_WhenBinary_When0PredictionsInPositiveClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_10)

    def test_Compute_ShouldCalculate_WhenBinary_When0PredictionsInNegativeClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_11)
        
    def test_Compute_ShouldCalculate_WhenBinary_When0TrueSamplesAndPredictionsInPositiveClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_12)
        
    def test_Compute_ShouldCalculate_WhenBinary_When0TrueSamplesAndPredictionsInNegativeClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_13)
    


class TestMicroPrecision(MetricTestBase):
    def setUp(self):
        self.metric_name = "micro_precision"
        self.multiclass_metric_calculator = MulticlassPrecision(average="micro", num_classes=3)
        self.binary_metric_calculator = BinaryPrecision(threshold=0.0)
        #self.multiclass_metric_calculator = MicroPrecision(num_classes=3)
        #self.binary_metric_calculator = MicroPrecision(num_classes=2)
        
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
        
    def test_Compute_ShouldCalculate_WhenBinaryUnbalanced(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_unbalanced_6)

    def test_Compute_ShouldCalculate_WhenBinaryBalanced(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_balanced_7)

    def test_Compute_ShouldCalculate_WhenBinary_When0TrueSamplesInPositiveClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_8)
        
    def test_Compute_ShouldCalculate_WhenBinary_When0TrueSamplesInNegativeClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_9)

    def test_Compute_ShouldCalculate_WhenBinary_When0PredictionsInPositiveClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_10)

    def test_Compute_ShouldCalculate_WhenBinary_When0PredictionsInNegativeClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_11)
        
    def test_Compute_ShouldCalculate_WhenBinary_When0TrueSamplesAndPredictionsInPositiveClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_12)
        
    def test_Compute_ShouldCalculate_WhenBinary_When0TrueSamplesAndPredictionsInNegativeClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_13)
        
        

class TestPerClassPrecision(MetricTestBase):
    def setUp(self):
        self.metric_name = "precision_per_class"
        #self.multiclass_metric_calculator = MulticlassPrecision(average=None, num_classes=3)
        #self.binary_metric_calculator = BinaryPrecision(threshold=0.0)
        self.multiclass_metric_calculator = PerClassPrecision(num_classes=3)
        self.binary_metric_calculator = PerClassPrecision(num_classes=2)
    
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
        
    def test_Compute_ShouldCalculate_WhenBinaryUnbalanced(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_unbalanced_6)

    def test_Compute_ShouldCalculate_WhenBinaryBalanced(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_balanced_7)

    def test_Compute_ShouldCalculate_WhenBinary_When0TrueSamplesInPositiveClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_8)
        
    def test_Compute_ShouldCalculate_WhenBinary_When0TrueSamplesInNegativeClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_9)

    def test_Compute_ShouldCalculate_WhenBinary_When0PredictionsInPositiveClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_10)

    def test_Compute_ShouldCalculate_WhenBinary_When0PredictionsInNegativeClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_11)
        
    def test_Compute_ShouldCalculate_WhenBinary_When0TrueSamplesAndPredictionsInPositiveClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_12)
        
    def test_Compute_ShouldCalculate_WhenBinary_When0TrueSamplesAndPredictionsInNegativeClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_13)

