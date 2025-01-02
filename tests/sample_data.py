import torch

class SampleData:
    def __init__(self, logits, true_numerical_labels, accuracies, precisions):
        self.logits = logits
        self.true_numerical_labels = true_numerical_labels
        
        self.micro_accuracy = accuracies['micro']
        self.accuracy_per_class = accuracies['per_class']
        self.macro_accuracy = accuracies['macro']
        
        self.macro_precision = precisions['macro']
        self.precision_per_class = precisions['per_class']
        self.micro_precision = precisions['micro']
        
  
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
                                                 'macro': 0.5556})


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
                                               'macro': 0.4444})


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
                                               'macro': 0.4222}) # 0.0s are counted


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
                                               'per_class': [0.25, None, 0.5],
                                               'macro': 0.375})


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
                                               'per_class': [0.5, None, 0.5],
                                               'macro': 0.5})


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
                                             'per_class': [1.0, 0.6667],
                                             'macro': 0.8333})


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
                                           'per_class': [1.0, 0.6667],
                                           'macro': 0.8333})


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
                                  'per_class': [0.0, 1.0],
                                  'macro': 0.5})



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
                                  'per_class': [1.0, 0.0],
                                  'macro': 0.5})


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
                                   'per_class': [None, 0.75],
                                   'macro': 0.75})


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
                                   'per_class': [0.25, None],
                                   'macro': 0.25})


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
                                   'per_class': [None, 1.0],
                                   'macro': 1.0})


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
                                   'per_class': [1.0, None],
                                   'macro': 1.0})