import torch
import numpy as np

class SampleData:
    def get_macro_perclass_micro(self, metric):
        return metric['macro'], metric['per_class'], metric['micro']
    
    def __init__(self, logits, true_numerical_labels, accuracies, precisions, recalls, f1s, kappa, mcc, mse, logloss, roc):
          self.logits = logits
          self.true_numerical_labels = true_numerical_labels
          
          self.macro_accuracy, self.accuracy_per_class, self.micro_accuracy = self.get_macro_perclass_micro(accuracies)
          self.macro_precision, self.precision_per_class, self.micro_precision = self.get_macro_perclass_micro(precisions)
          self.macro_recall, self.recall_per_class, self.micro_recall = self.get_macro_perclass_micro(recalls)
          self.macro_f1, self.f1_per_class, self.micro_f1 = self.get_macro_perclass_micro(f1s)
          
          self.kappa = kappa
          self.mcc = mcc
          self.mse = mse
          self.logloss = logloss
          self.roc = roc
          
  
########################################################
## 1 MULTICLASS, UNBALANCED
########################################################
logits1 = torch.tensor([[ 3.0,  2.0,  1.0],      #0
                        [ 2.0,  1.0,  1.0],      #1
                        [-2.0,  1.0,  2.0],      #2
                        [ 0.2,  0.3,  0.4],      #3
                        [ 0.5,  0.2, -1.0],      #4
                        [ 5.0,  2.0,  3.0],      #5
                        [ 3.0,  4.0,  6.0],      #6
                        [ 5.0, -2.0,  4.0],      #7
                        [ 3.0, -3.0,  1.0],      #8
                        [ 0.0,  2.0,  0.2]])     #9

labels1 = torch.tensor([0, 1, 0, 0, 2, 1, 2, 0, 2, 1])

multiclass_unbalanced_1 = SampleData(logits=logits1,
                                     true_numerical_labels=labels1,
                                     accuracies={'micro': 0.6,
                                                 'per_class': [0.4, 0.8, 0.6],
                                                 'macro': 0.6},
                                     precisions={'micro': 0.4,
                                                 'per_class': [0.3333, 1.0, 0.3333],
                                                 'macro': 0.5556},
                                     recalls={'micro': 0.4,
                                              'per_class': [0.5, 0.3333, 0.3333],
                                              'macro': 0.3889},
                                     f1s={'micro': 0.4,
                                          'per_class': [0.4, 0.5, 0.3333],
                                          'macro': 0.4111},
                                     kappa=0.0625,
                                     mcc=0.0670,
                                     mse=0.2707,
                                     logloss=1.5708,
                                     roc={"aunu": 0.4881,
                                          "aunp": 0.4810,
                                          "au1u": 0.5093,    
                                          "au1p": 0.5})


########################################################
## 2 MULTICLASS, BALANCED
########################################################
logits2 = torch.tensor([[ 3.0,  2.0,  1.0],      #0
                        [ 2.0,  1.0,  1.0],      #1
                        [-2.0,  1.0,  2.0],      #2
                        [ 0.2,  0.3,  0.4],      #3
                        [ 0.5,  0.2, -1.0],      #4
                        [ 5.0,  2.0,  3.0],      #5
                        [ 3.0,  4.0,  6.0],      #6
                        [ 5.0, -2.0,  4.0],      #7
                        [ 3.0, -3.0,  1.0],      #8
                        [ 0.0,  2.0,  0.2],      #9
                        [ 0.1,  2.0,  1.6],      #10
                        [ 2.0,  5.0,  1.0]])     #11

labels2 = torch.tensor([0, 1, 0, 0, 2, 1, 2, 0, 2, 1, 2, 1])

multiclass_balanced_2 = SampleData(logits=logits2,
                                   true_numerical_labels=labels2,
                                   accuracies={'micro': 0.6111,
                                               'per_class': [0.5, 0.75, 0.5833],
                                               'macro': 0.6111},
                                   precisions={'micro': 0.4167,
                                               'per_class': [0.3333, 0.6667, 0.3333],
                                               'macro': 0.4444},
                                   recalls={'micro': 0.4167,
                                            'per_class': [0.5, 0.5, 0.25],
                                            'macro': 0.4167},
                                   f1s={'micro': 0.4167,
                                        'per_class': [0.4, 0.5714, 0.2857],
                                        'macro': 0.4190},
                                   kappa=0.125,
                                   mcc=0.1291,
                                   mse=0.2454,
                                   logloss=1.3977,
                                   roc={"aunu": 0.5938,
                                        "aunp": 0.5938,
                                        "au1u": 0.5938,    
                                        "au1p": 0.5938})


########################################################
## 3 MULTICLASS, BALANCED, 0 TRUE SAMPLES IN CLASS 1
########################################################
logits3 = torch.tensor([[ 3.0,  2.0,  1.0],      #0
                        [ 2.0,  1.0,  1.0],      #1
                        [-2.0,  1.0,  2.0],      #2
                        [ 0.2,  0.3,  0.4],      #3
                        [ 0.5,  0.2, -1.0],      #4
                        [ 5.0,  2.0,  3.0],      #5
                        [ 3.0,  4.0,  6.0],      #6
                        [ 5.0, -2.0,  4.0],      #7
                        [ 3.0, -3.0,  1.0],      #8
                        [ 0.0,  2.0,  4.0],      #9
                        [ 0.1,  2.0,  1.6],      #10
                        [ 2.0,  5.0,  8.0]])     #11

labels3 = torch.tensor([0, 0, 0, 0, 2, 0, 2, 0, 2, 2, 2, 2])

multiclass_balanced_3 = SampleData(logits=logits3,
                                   true_numerical_labels=labels3,
                                   accuracies={'micro': 0.7222,
                                               'per_class': [0.6667, 0.9167, 0.5833],
                                               'macro': 0.7222},
                                   precisions={'micro': 0.5833,
                                               'per_class': [0.6667, 0.0, 0.6],
                                               'macro': 0.4222},
                                   recalls={'micro': 0.5833,
                                            'per_class': [0.6667, np.nan, 0.5],
                                            'macro': 0.5833},
                                   f1s={'micro': 0.5833,
                                        'per_class': [0.6667, np.nan, 0.5455],
                                        'macro': 0.6061},
                                   kappa=0.2308,
                                   mcc=0.2343,
                                   mse=0.1792,
                                   logloss=1.0532,
                                   roc={"aunu": 0.7083,
                                        "aunp": 0.7083,
                                        "au1u": 0.7083,    
                                        "au1p": 0.7083})


########################################################
## 4 MULTICLASS, BALANCED, 0 PREDICTIONS IN CLASS 1
########################################################
logits4 = torch.tensor([[ 3.0,  2.0,  1.0],      #0
                        [ 2.0,  1.0,  1.0],      #1
                        [-2.0,  1.0,  2.0],      #2
                        [ 0.2,  0.3,  0.4],      #3
                        [ 0.5,  0.2, -1.0],      #4
                        [ 5.0,  2.0,  3.0],      #5
                        [ 3.0,  4.0,  6.0],      #6
                        [ 5.0, -2.0,  4.0],      #7
                        [ 3.0, -3.0,  1.0],      #8
                        [ 3.0,  2.0,  0.2],      #9
                        [ 0.1,  2.0,  2.3],      #10
                        [ 6.0,  5.0,  1.0]])     #11

labels4 = torch.tensor([0, 1, 0, 0, 2, 1, 2, 0, 2, 1, 2, 1])

multiclass_balanced_4 = SampleData(logits=logits4,
                                   true_numerical_labels=labels4,
                                   accuracies={'micro': 0.5556,
                                               'per_class': [0.3333, 0.6667, 0.6667],
                                               'macro': 0.5556},
                                   precisions={'micro': 0.3333,
                                               'per_class': [0.25, np.nan, 0.5],
                                               'macro': 0.375},
                                   recalls={'micro': 0.3333,
                                            'per_class': [0.5, 0.0, 0.5],
                                            'macro': 0.3333},
                                   f1s={'micro': 0.3333,
                                        'per_class': [0.3333, 0.0, 0.5],
                                        'macro': 0.2778},
                                   kappa=0.0,
                                   mcc=0.0,
                                   mse=0.2923,
                                   logloss=1.5614,
                                   roc={"aunu": 0.5,
                                        "aunp": 0.5,
                                        "au1u": 0.5,    
                                        "au1p": 0.5})


########################################################
## 5 MULTICLASS, BALANCED, 0 TRUE SAMPLES AND PREDICTIONS IN CLASS 1
########################################################
logits5 = torch.tensor([[ 3.0,  2.0,  1.0],      #0
                        [ 2.0,  1.0,  1.0],      #1
                        [-2.0,  1.0,  2.0],      #2
                        [ 0.2,  0.3,  0.4],      #3
                        [ 0.5,  0.2, -1.0],      #4
                        [ 5.0,  2.0,  3.0],      #5
                        [ 3.0,  4.0,  6.0],      #6
                        [ 5.0, -2.0,  4.0],      #7
                        [ 3.0, -3.0,  1.0],      #8
                        [ 3.0,  2.0,  0.2],      #9
                        [ 0.1,  2.0,  2.3],      #10
                        [ 6.0,  5.0,  1.0]])     #11

labels5 = torch.tensor([0, 0, 0, 0, 2, 0, 2, 0, 2, 2, 2, 2])

multiclass_balanced_5 = SampleData(logits=logits5,
                                   true_numerical_labels=labels5,
                                   accuracies={'micro': 0.6667,
                                               'per_class': [0.5, 1.0, 0.5],
                                               'macro': 0.6667},
                                   precisions={'micro': 0.5,
                                               'per_class': [0.5, np.nan, 0.5],
                                               'macro': 0.5},
                                   recalls={'micro': 0.5,
                                            'per_class': [0.6667, np.nan, 0.3333],
                                            'macro': 0.5},
                                   f1s={'micro': 0.5,
                                        'per_class': [0.5714, np.nan, 0.4],
                                        'macro': 0.4857},
                                   kappa=0.0,
                                   mcc=0.0,
                                   mse=0.2541,
                                   logloss=1.7114,
                                   roc={"aunu": 0.4444,
                                        "aunp": 0.4444,
                                        "au1u": 0.4444,    
                                        "au1p": 0.4444})


########################################################
## 6 BINARY, UNBALANCED
########################################################
logits6 = torch.tensor([-3.0,     #0
                        -2.4,     #1
                        -1.0,     #2
                        -0.5,     #3
                         3.2,     #4
                        -0.8,     #5
                        -4.0])    #6

labels6 = torch.tensor([0, 0, 0, 0, 1, 1, 1])

binary_unbalanced_6 = SampleData(logits=logits6,
                                 true_numerical_labels=labels6,
                                 accuracies={'micro': 0.7143,
                                             'per_class': [0.7143, 0.7143],
                                             'macro': 0.7143},
                                 precisions={'micro': 0.7143,
                                             'per_class': [0.6667, 1.0],
                                             'macro': 0.8333},
                                 recalls={'micro': 0.7143,
                                          'per_class': [1.0, 0.3333],
                                          'macro': 0.6667},
                                 f1s={'micro': 0.7143,
                                      'per_class': [0.8, 0.5],
                                      'macro': 0.65},
                                 kappa=0.3636,
                                 mcc=0.4714,
                                 mse=0.2380,
                                 logloss=0.8789,
                                 roc=0.9167)


########################################################
## 7 BINARY, BALANCED
########################################################
logits7 = torch.tensor([-3.0,     #0
                        -2.4,     #1
                        -1.0,     #2
                        -0.2,     #3
                         3.2,     #4
                        -0.8,     #5
                        -2.8,     #6
                         2.1])    #7

labels7 = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

binary_balanced_7 = SampleData(logits=logits7,
                               true_numerical_labels=labels7,
                               accuracies={'micro': 0.75,
                                           'per_class': [0.75, 0.75],
                                           'macro': 0.75},
                               precisions={'micro': 0.75,
                                           'per_class': [0.6667, 1.0],
                                           'macro': 0.8333},
                               recalls={'micro': 0.75,
                                        'per_class': [1.0, 0.5],
                                        'macro': 0.75},
                               f1s={'micro': 0.75,
                                    'per_class': [0.8, 0.6667],
                                    'macro': 0.7333},
                               kappa=0.5,
                               mcc=0.5774,
                               mse=0.2078,
                               logloss=0.6541,
                               roc=0.75)


########################################################
## 8 BINARY, 0 TRUE SAMPLES IN POSITIVE CLASS 1
########################################################
logits8 = torch.tensor([-3.0,     #0
                        -2.4,     #1
                        -1.0,     #2
                        -0.2,     #3
                         3.2,     #4
                        -0.8,     #5
                        -2.8,     #6
                         2.1])    #7

labels8 = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0])

binary_8 = SampleData(logits=logits8,
                      true_numerical_labels=labels8,
                      accuracies={'micro': 0.75,
                                  'per_class': [0.75, 0.75],
                                  'macro': 0.75},
                      precisions={'micro': 0.75,
                                  'per_class': [1.0, 0.0],
                                  'macro': 0.5},
                      recalls={'micro': 0.75,
                               'per_class': [0.75, np.nan],
                               'macro': 0.75},
                      f1s={'micro': 0.75,
                           'per_class': [0.8571, np.nan],
                           'macro': 0.8571},
                      kappa=0.0,
                      mcc=np.nan,
                      mse=0.2626,
                      logloss=0.8666,
                      roc=np.nan)



########################################################
## 9 BINARY, 0 TRUE SAMPLES IN NEGATIVE CLASS 0
########################################################
logits9 = torch.tensor([ 2.0,     #0
                         1.2,     #1
                         0.6,     #2
                         0.4,     #3
                         0.9,     #4
                         1.1,     #5
                        -0.9,     #6
                        -0.7])    #7

labels9 = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1])

binary_9 = SampleData(logits=logits9,
                      true_numerical_labels=labels9,
                      accuracies={'micro': 0.75,
                                  'per_class': [0.75, 0.75],
                                  'macro': 0.75},
                      precisions={'micro': 0.75,
                                  'per_class': [0.0, 1.0],
                                  'macro': 0.5},
                      recalls={'micro': 0.75,
                               'per_class': [np.nan, 0.75],
                               'macro': 0.75},
                      f1s={'micro': 0.75,
                           'per_class': [np.nan, 0.8571],
                           'macro': 0.8571},
                      kappa=0.0,
                      mcc=np.nan,
                      mse=0.1815,
                      logloss=0.5392,
                      roc=np.nan)


########################################################
## 10 BINARY, 0 PREDICTIONS IN POSITIVE CLASS 1
########################################################
logits10 = torch.tensor([-2.0,     #0
                         -1.2,     #1
                         -0.6,     #2
                         -0.4,     #3
                         -0.9,     #4
                         -1.1,     #5
                         -0.9,     #6
                         -0.7])    #7

labels10 = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1])

binary_10 = SampleData(logits=logits10,
                       true_numerical_labels=labels10,
                       accuracies={'micro': 0.75,
                                   'per_class': [0.75, 0.75],
                                   'macro': 0.75},
                       precisions={'micro': 0.75,
                                   'per_class': [0.75, np.nan],
                                   'macro': 0.75},
                       recalls={'micro': 0.75,
                                'per_class': [1.0, 0.0],
                                'macro': 0.5},
                       f1s={'micro': 0.75,
                            'per_class': [0.8571, 0.0],
                            'macro': 0.4286},
                       kappa=0.0,
                       mcc=np.nan,
                       mse=0.1815,
                       logloss=0.5392,
                       roc=0.6250)


########################################################
## 11 BINARY, 0 PREDICTIONS IN NEGATIVE CLASS 0
########################################################
logits11 = torch.tensor([ 2.0,     #0
                          1.2,     #1
                          0.6,     #2
                          0.4,     #3
                          0.9,     #4
                          1.1,     #5
                          0.9,     #6
                          0.7])    #7

labels11 = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1])

binary_11 = SampleData(logits=logits11,
                       true_numerical_labels=labels11,
                       accuracies={'micro': 0.25,
                                   'per_class': [0.25, 0.25],
                                   'macro': 0.25},
                       precisions={'micro': 0.25,
                                   'per_class': [np.nan, 0.25],
                                   'macro': 0.25},
                       recalls={'micro': 0.25,
                                'per_class': [0.0, 1.0],
                                'macro': 0.5},
                       f1s={'micro': 0.25,
                            'per_class': [0.0, 0.4],
                            'macro': 0.2},
                       kappa=0.0,
                       mcc=np.nan,
                       mse=0.4255,
                       logloss=1.1142,
                       roc=0.6250)


########################################################
## 12 BINARY, 0 PREDICTIONS AND SAMPLES IN POSITIVE CLASS 1
########################################################
logits12 = torch.tensor([-2.0,     #0
                         -1.2,     #1
                         -0.6,     #2
                         -0.4,     #3
                         -0.9,     #4
                         -1.1,     #5
                         -0.9,     #6
                         -0.7])    #7

labels12 = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0])

binary_12 = SampleData(logits=logits12,
                       true_numerical_labels=labels12,
                       accuracies={'micro': 1.0,
                                   'per_class': [1.0, 1.0],
                                   'macro': 1.0},
                       precisions={'micro': 1.0,
                                   'per_class': [1.0, np.nan],
                                   'macro': 1.0},
                       recalls={'micro': 1.0,
                                'per_class': [1.0, np.nan],
                                'macro': 1.0},
                       f1s={'micro': 1.0,
                            'per_class': [1.0, np.nan],
                            'macro': 1.0},
                       kappa=np.nan,
                       mcc=np.nan,
                       mse=0.0867,
                       logloss=0.3392,
                       roc=np.nan)


########################################################
## 13 BINARY, 0 PREDICTIONS AND SAMPLES IN NEGATIVE CLASS 0
########################################################
logits13 = torch.tensor([ 2.0,     #0
                          1.2,     #1
                          0.6,     #2
                          0.4,     #3
                          0.9,     #4
                          1.1,     #5
                          0.9,     #6
                          0.7])    #7

labels13 = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1])

binary_13 = SampleData(logits=logits13,
                       true_numerical_labels=labels13,
                       accuracies={'micro': 1.0,
                                   'per_class': [1.0, 1.0],
                                   'macro': 1.0},
                       precisions={'micro': 1.0,
                                   'per_class': [np.nan, 1.0],
                                   'macro': 1.0},
                       recalls={'micro': 1.0,
                               'per_class': [np.nan, 1.0],
                               'macro': 1.0},
                       f1s={'micro': 1.0,
                            'per_class': [np.nan, 1.0],
                            'macro': 1.0},
                       kappa=np.nan,
                       mcc=np.nan,
                       mse=0.0867,
                       logloss=0.3392,
                       roc=np.nan)