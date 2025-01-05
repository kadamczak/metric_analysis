import unittest

class MetricTestBase(unittest.TestCase):
    def _init_(self, metric_name, metric_calculator):
        self.metric_name = metric_name
        self.metric_calculator = metric_calculator
        
    def get_data(self, sample):
        return (sample.logits,
                sample.true_numerical_labels,
                getattr(sample, self.metric_name))
        
    def calculate_result(self, logits, true_numerical_labels):
        self.metric_calculator.update(logits, true_numerical_labels)
        result = self.metric_calculator.compute()
        self.metric_calculator.reset()
          
        return round(result.item(), 4)
    
    def expected_matches_result(self, sample):
        logits, true_labels, expected = self.get_data(sample)       
        result = self.calculate_result(logits, true_labels)    
        assert expected == result