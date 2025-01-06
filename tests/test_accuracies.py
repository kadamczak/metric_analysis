import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/experiment')))
from torcheval.metrics import BinaryAccuracy
from qualitative_metrics.accuracies import MacroAccuracy, MicroAccuracy, AccuracyPerClass

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


class TestMacroAccuracy(MetricTestBase):
    def setUp(self):
        self.metric_name = "macro_accuracy"
        self.default_metric_calculator = MacroAccuracy(num_classes=3)
    
    def test_Compute_ShouldCalculate_WhenMulticlassUnbalanced(self):
        self.expected_matches_result(self.default_metric_calculator, multiclass_unbalanced_1)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced(self):
        self.expected_matches_result(self.default_metric_calculator, multiclass_balanced_2)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0TrueSamplesInClass(self):
        self.expected_matches_result(self.default_metric_calculator, multiclass_balanced_3)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0PredictionsInClass(self):
        self.expected_matches_result(self.default_metric_calculator, multiclass_balanced_4)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0TrueSamplesAndPredictionsInClass(self):
        self.expected_matches_result(self.default_metric_calculator, multiclass_balanced_5)


        
class TestMicroAccuracy(MetricTestBase):
    def setUp(self):
        self.metric_name = "micro_accuracy"
        self.default_metric_calculator = MicroAccuracy(num_classes=3)
    
    def test_Compute_ShouldCalculate_WhenMulticlassUnbalanced(self):
        self.expected_matches_result(self.default_metric_calculator, multiclass_unbalanced_1)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced(self):
        self.expected_matches_result(self.default_metric_calculator, multiclass_balanced_2)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0TrueSamplesInClass(self):
        self.expected_matches_result(self.default_metric_calculator, multiclass_balanced_3)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0PredictionsInClass(self):
        self.expected_matches_result(self.default_metric_calculator, multiclass_balanced_4)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0TrueSamplesAndPredictionsInClass(self):
        self.expected_matches_result(self.default_metric_calculator, multiclass_balanced_5)
    
        
        
class TestPerClassAccuracy(MetricTestBase):
    def setUp(self):
        self.metric_name = "accuracy_per_class"
        self.default_metric_calculator = AccuracyPerClass(num_classes=3)
        
    def test_Compute_ShouldCalculate_WhenMulticlassUnbalanced(self):
        self.expected_matches_result(self.default_metric_calculator, multiclass_unbalanced_1)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced(self):
        self.expected_matches_result(self.default_metric_calculator, multiclass_balanced_2)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0TrueSamplesInClass(self):
        self.expected_matches_result(self.default_metric_calculator, multiclass_balanced_3)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0PredictionsInClass(self):
        self.expected_matches_result(self.default_metric_calculator, multiclass_balanced_4)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0TrueSamplesAndPredictionsInClass(self):
        self.expected_matches_result(self.default_metric_calculator, multiclass_balanced_5)
       
 
        
class TestBinaryAccuracy(MetricTestBase):
    def setUp(self):
        self.metric_name = "macro_accuracy"
        self.default_metric_calculator = BinaryAccuracy(threshold=0)

    def test_Compute_ShouldCalculate_WhenBinaryUnbalanced(self):
        self.expected_matches_result(self.default_metric_calculator, binary_unbalanced_6)

    def test_Compute_ShouldCalculate_WhenBinaryBalanced(self):
        self.expected_matches_result(self.default_metric_calculator, binary_balanced_7)

    def test_Compute_ShouldCalculate_WhenBinary_When0TrueSamplesInPositiveClass(self):
        self.expected_matches_result(self.default_metric_calculator, binary_8)
        
    def test_Compute_ShouldCalculate_WhenBinary_When0TrueSamplesInNegativeClass(self):
        self.expected_matches_result(self.default_metric_calculator, binary_9)

    def test_Compute_ShouldCalculate_WhenBinary_When0PredictionsInPositiveClass(self):
        self.expected_matches_result(self.default_metric_calculator, binary_10)

    def test_Compute_ShouldCalculate_WhenBinary_When0PredictionsInNegativeClass(self):
        self.expected_matches_result(self.default_metric_calculator, binary_11)
        
    def test_Compute_ShouldCalculate_WhenBinary_When0TrueSamplesAndPredictionsInPositiveClass(self):
        self.expected_matches_result(self.default_metric_calculator, binary_12)
        
    def test_Compute_ShouldCalculate_WhenBinary_When0TrueSamplesAndPredictionsInNegativeClass(self):
        self.expected_matches_result(self.default_metric_calculator, binary_13)
        

