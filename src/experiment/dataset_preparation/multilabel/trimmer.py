import pandas as pd
import numpy as np

def trim_class(X, y, class_name, step, protected_classes):
    y_numeric = y.apply(pd.to_numeric)

    relaxing_indices = y_numeric.index[
        (y_numeric[class_name] == 1) &
        (y_numeric[protected_classes].sum(axis=1) == 0)
    ].tolist()


    np.random.seed(47)
    remove_indices = np.random.choice(relaxing_indices, step, replace=False)

    X_trimmed = X.drop(remove_indices).reset_index(drop=True)
    y_trimmed = y.drop(remove_indices).reset_index(drop=True)
    
    #y_numeric_trimmed = y_numeric.drop(remove_indices).reset_index(drop=True)
    # class_percentages_trimmed = y_numeric_trimmed.sum() / len(y_numeric_trimmed) * 100
    # print(class_percentages_trimmed)
    
    return X_trimmed, y_trimmed

