import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/experiment')))
from src.experiment.metrics.qualitative.accuracies import MacroAccuracy, MicroAccuracy, PerClassAccuracy
from src.experiment.helpers.task_type import TaskType

from helpers.metric_test_base import MetricTestBase
from helpers.sample_data import (
    multiclass_1,
    multiclass_2,
    multiclass_3,
    multiclass_4,
    multiclass_5,
    binary_6,
    binary_7,
    binary_8,
    binary_9,
    binary_10,
    binary_11,
    binary_12,
    binary_13,
    multilabel_14
)

#============
# TorchEval
#============

# MACRO
# wrongly calculates macrorecall instead

# MICRO 
# wrongly calculates microrecall instead

# PER CLASS
# wrongly calculates per class recall instead


#============
# Sklearn
#============

# does not work in multiclass context

class TestMacroAccuracy(MetricTestBase):
    def setUp(self):
        self.metric_name = "macro_accuracy"
        #self.multiclass_metric_calculator = MulticlassAccuracy(average="macro", num_classes=3)
        self.multiclass_metric_calculator = MacroAccuracy(num_classes=3, task_type=TaskType.MULTICLASS)
        self.binary_metric_calculator = MacroAccuracy(num_classes=2, task_type=TaskType.BINARY)
        self.multilabel_metric_calculator = MacroAccuracy(num_classes=3, task_type=TaskType.MULTILABEL)
    
    def test_Compute_ShouldCalculate_WhenMulticlassImbalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_1)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_2)
        
    def test_Compute_ShouldCalculate_WhenMulticlassImbalanced_When0TrueSamplesInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_3)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0PredictionsInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_4)
        
    def test_Compute_ShouldCalculate_WhenMulticlassImbalanced_When0TrueSamplesAndPredictionsInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_5)
        
    def test_Compute_ShouldCalculate_WhenBinaryImbalanced(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_6)

    def test_Compute_ShouldCalculate_WhenBinaryBalanced(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_7)

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
        
    def test_Compute_ShouldCalculate_WhenMultilabel1(self):
        self.expected_matches_result(self.multilabel_metric_calculator, multilabel_14)
        
        
class TestMicroAccuracy(MetricTestBase):
    def setUp(self):
        self.metric_name = "micro_accuracy"   
        self.multiclass_metric_calculator = MicroAccuracy(num_classes=3, task_type=TaskType.MULTICLASS)
        self.binary_metric_calculator = MicroAccuracy(num_classes=2, task_type=TaskType.BINARY)
        self.multilabel_metric_calculator = MicroAccuracy(num_classes=3, task_type=TaskType.MULTILABEL)
    
    def test_Compute_ShouldCalculate_WhenMulticlassImbalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_1)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_2)
        
    def test_Compute_ShouldCalculate_WhenMulticlassImbalanced_When0TrueSamplesInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_3)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0PredictionsInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_4)
        
    def test_Compute_ShouldCalculate_WhenMulticlassImbalanced_When0TrueSamplesAndPredictionsInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_5)
        
    def test_Compute_ShouldCalculate_WhenBinaryImbalanced(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_6)

    def test_Compute_ShouldCalculate_WhenBinaryBalanced(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_7)

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
        
    def test_Compute_ShouldCalculate_WhenMultilabel1(self):
        self.expected_matches_result(self.multilabel_metric_calculator, multilabel_14)
    
        
        
class TestPerClassAccuracy(MetricTestBase):
    def setUp(self):
        self.metric_name = "accuracy_per_class"
        self.multiclass_metric_calculator = PerClassAccuracy(num_classes=3, task_type=TaskType.MULTICLASS)
        self.binary_metric_calculator = PerClassAccuracy(num_classes=2, task_type=TaskType.BINARY)
        self.multilabel_metric_calculator = PerClassAccuracy(num_classes=3, task_type=TaskType.MULTILABEL)
        
    def test_Compute_ShouldCalculate_WhenMulticlassImbalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_1)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_2)
        
    def test_Compute_ShouldCalculate_WhenMulticlassImbalanced_When0TrueSamplesInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_3)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0PredictionsInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_4)
        
    def test_Compute_ShouldCalculate_WhenMulticlassImbalanced_When0TrueSamplesAndPredictionsInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_5)
        
    def test_Compute_ShouldCalculate_WhenBinaryImbalanced(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_6)

    def test_Compute_ShouldCalculate_WhenBinaryBalanced(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_7)

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
        
    def test_Compute_ShouldCalculate_WhenMultilabel1(self):
        self.expected_matches_result(self.multilabel_metric_calculator, multilabel_14)

       

