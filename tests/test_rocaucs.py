import sys
import os

from src.experiment.helpers.task_type import TaskType

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/experiment')))
from src.experiment.metrics.rank.rocauc import ROCAUC

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

class TestBinaryROC(MetricTestBase):
    def setUp(self):
        self.metric_name = "roc"
        self.binary_metric_calculator = ROCAUC(comparison_method="raise", average="macro", task_type=TaskType.BINARY)
        
    def test_ShouldCalculate_WhenBinaryImbalanced(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_6)

    def test_ShouldCalculate_WhenBinaryBalanced(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_7)

    def test_ShouldCalculate_WhenBinary_When0TrueSamplesInPositiveClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_8)
        
    def test_ShouldCalculate_WhenBinary_When0TrueSamplesInNegativeClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_9)

    def test_ShouldCalculate_WhenBinary_When0PredictionsInPositiveClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_10)

    def test_ShouldCalculate_WhenBinary_When0PredictionsInNegativeClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_11)
        
    def test_ShouldCalculate_WhenBinary_When0TrueSamplesAndPredictionsInPositiveClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_12)
        
    def test_ShouldCalculate_WhenBinary_When0TrueSamplesAndPredictionsInNegativeClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_13)

    
    
        
class TestAUNU(MetricTestBase):
    def setUp(self):
        self.metric_name = "aunu"
        self.multiclass_metric_calculator = ROCAUC(comparison_method="ovr", average="macro", task_type=TaskType.MULTICLASS)
        self.multilabel_metric_calculator = ROCAUC(comparison_method="ovr", average="macro", task_type=TaskType.MULTILABEL)
        
    def test_ShouldCalculate_WhenMulticlassImbalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_1)
        
    def test_ShouldCalculate_WhenMulticlassBalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_2)
        
    def test_ShouldCalculate_WhenMulticlassImbalanced_When0TrueSamplesInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_3)
        
    def test_ShouldCalculate_WhenMulticlassBalanced_When0PredictionsInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_4)
        
    def test_ShouldCalculate_WhenMulticlassImbalanced_When0TrueSamplesAndPredictionsInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_5)
    
    def test_ShouldCalculate_WhenMultilabel1(self):
        self.expected_matches_result(self.multilabel_metric_calculator, multilabel_14)


class TestAUNP(MetricTestBase):
    def setUp(self):
        self.metric_name = "aunp"
        self.multiclass_metric_calculator = ROCAUC(comparison_method="ovr", average="weighted", task_type=TaskType.MULTICLASS)
        self.multilabel_metric_calculator = ROCAUC(comparison_method="ovr", average="weighted", task_type=TaskType.MULTILABEL)
        
    def test_ShouldCalculate_WhenMulticlassImbalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_1)
        
    def test_ShouldCalculate_WhenMulticlassBalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_2)
        
    def test_ShouldCalculate_WhenMulticlassImbalanced_When0TrueSamplesInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_3)
        
    def test_ShouldCalculate_WhenMulticlassBalanced_When0PredictionsInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_4)
        
    def test_ShouldCalculate_WhenMulticlassImbalanced_When0TrueSamplesAndPredictionsInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_5)
        
    def test_ShouldCalculate_WhenMultilabel1(self):
        self.expected_matches_result(self.multilabel_metric_calculator, multilabel_14)
        

class TestAU1U(MetricTestBase):
    def setUp(self):
        self.metric_name = "au1u"
        self.multiclass_metric_calculator = ROCAUC(comparison_method="ovo", average="macro", task_type=TaskType.MULTICLASS)
        
    def test_ShouldCalculate_WhenMulticlassImbalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_1)
        
    def test_ShouldCalculate_WhenMulticlassBalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_2)
        
    def test_ShouldCalculate_WhenMulticlassImbalanced_When0TrueSamplesInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_3)
        
    def test_ShouldCalculate_WhenMulticlassBalanced_When0PredictionsInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_4)
        
    def test_ShouldCalculate_WhenMulticlassImbalanced_When0TrueSamplesAndPredictionsInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_5)
        
        
        
class TestAU1P(MetricTestBase):
    def setUp(self):
        self.metric_name = "au1p"
        self.multiclass_metric_calculator= ROCAUC(comparison_method="ovo", average="weighted", task_type=TaskType.MULTICLASS)
        
    def test_ShouldCalculate_WhenMulticlassImbalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_1)
        
    def test_ShouldCalculate_WhenMulticlassBalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_2)
        
    def test_ShouldCalculate_WhenMulticlassImbalanced_When0TrueSamplesInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_3)
        
    def test_ShouldCalculate_WhenMulticlassBalanced_When0PredictionsInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_4)
        
    def test_ShouldCalculate_WhenMulticlassImbalanced_When0TrueSamplesAndPredictionsInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_5)
        
    
        

class TestPerClassVsRest(MetricTestBase):
    def setUp(self):
        self.metric_name = "per_class_vs_rest"
        self.multiclass_metric_calculator = ROCAUC(comparison_method="ovr", average=None, task_type=TaskType.MULTICLASS)
        self.multilabel_metric_calculator = ROCAUC(comparison_method="ovr", average=None, task_type=TaskType.MULTILABEL)
        
    def test_ShouldCalculate_WhenMulticlassImbalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_1)
        
    def test_ShouldCalculate_WhenMulticlassBalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_2)
        
    def test_ShouldCalculate_WhenMulticlassImbalanced_When0TrueSamplesInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_3)
        
    def test_ShouldCalculate_WhenMulticlassBalanced_When0PredictionsInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_4)
        
    def test_ShouldCalculate_WhenMulticlassImbalanced_When0TrueSamplesAndPredictionsInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_5)
        
    
    def test_ShouldCalculate_WhenMultilabel1(self):
        self.expected_matches_result(self.multilabel_metric_calculator, multilabel_14)


class TestMicroROCAUC(MetricTestBase):
    def setUp(self):
        self.metric_name = "micro_rocauc"
        self.binary_metric_calculator = ROCAUC(comparison_method="ovr", average="micro", task_type=TaskType.BINARY)
        self.multiclass_metric_calculator = ROCAUC(comparison_method="ovr", average="micro", task_type=TaskType.MULTICLASS)
        self.multilabel_metric_calculator = ROCAUC(comparison_method="ovr", average="micro", task_type=TaskType.MULTILABEL)
        
    def test_ShouldCalculate_WhenMulticlassImbalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_1)
        
    def test_ShouldCalculate_WhenMulticlassBalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_2)
        
    def test_ShouldCalculate_WhenMulticlassImbalanced_When0TrueSamplesInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_3)
        
    def test_ShouldCalculate_WhenBinaryImbalanced(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_6)
        
    def test_ShouldCalculate_WhenBinaryBalanced(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_7)
        
    def test_ShouldCalculate_WhenBinary_When0TrueSamplesInPositiveClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_8)
        
    def test_ShouldCalculate_WhenBinary_When0TrueSamplesInNegativeClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_9)

    def test_ShouldCalculate_WhenBinary_When0PredictionsInPositiveClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_10)

    def test_ShouldCalculate_WhenBinary_When0PredictionsInNegativeClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_11)
        
    def test_ShouldCalculate_WhenBinary_When0TrueSamplesAndPredictionsInPositiveClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_12)
        
    def test_ShouldCalculate_WhenBinary_When0TrueSamplesAndPredictionsInNegativeClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_13)

    def test_ShouldCalculate_WhenMultilabel1(self):
        self.expected_matches_result(self.multilabel_metric_calculator, multilabel_14)


    

        