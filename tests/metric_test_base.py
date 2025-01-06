import unittest
import torch

class MetricTestBase(unittest.TestCase):
    def _init_(self, metric_name, metric_calculator):
        self.metric_name = metric_name
        self.default_metric_calculator = metric_calculator
        
    def get_data(self, sample):
        return (sample.logits,
                sample.true_numerical_labels,
                getattr(sample, self.metric_name))
        
    def calculate_result(self, metric_calculator, logits, true_numerical_labels):
        metric_calculator.update(logits, true_numerical_labels)
        result = metric_calculator.compute()
        metric_calculator.reset()
        
        if result is None:
            return None
        
        value = result.tolist()
        
        if isinstance(value, list):
            return [round(item, 4) if item is not None else None for item in value]
          
        return round(value, 4)
    
    def expected_matches_result(self, metric_calculator, sample):
        logits, true_labels, expected = self.get_data(sample)       
        result = self.calculate_result(metric_calculator, logits, true_labels)    
        assert expected == result