import numpy as np
from collections import Counter


def retrieval_recall(pred, true, method="binary", zero_denom=1):
    pred_resources = set(pred)
    
    # No true resources, no predicted resources: Perfect recall
    if len(true) == 0 and len(pred) == 0:
        return 1.0
    
    # No true resources, yes predicted resources: Remove from calculation (return NaN)
    if len(true) == 0 and len(pred) > 0:
        return np.nan
    
    # Yes true resources, no predicted resources: Recall = 0
    if len(true) > 0 and len(pred) == 0:
        return 0.0
        
    # Standard case: yes true resources, yes predicted resources
    if method == "continuous":
        return np.mean([true_rsc in pred_resources for true_rsc in true])
    elif method == "binary":
        return int(all([true_rsc in pred_resources for true_rsc in true]))


def retrieval_precision(pred, true, method="continuous", zero_denom=0):
    # No true resources, no predicted resources: Perfect precision
    if len(true) == 0 and len(pred) == 0:
        return 1.0
    
    # No true resources, yes predicted resources: Precision = 0 (all false positives)
    if len(true) == 0 and len(pred) > 0:
        return 0.0
    
    # Yes true resources, no predicted resources: Remove from calculation (return NaN)
    if len(true) > 0 and len(pred) == 0:
        return np.nan
    
    # Standard case: yes true resources, yes predicted resources
    if method == "continuous":
        return np.mean([pred_rsc in true for pred_rsc in pred])
    elif method == "binary":
        return all([pred_rsc in true for pred_rsc in pred])