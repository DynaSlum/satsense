# -*- coding: utf-8 -*-
"""
jaccrad_similarity

functions to calculate the Jaccard similarity index between 2 binary masks or multipolygonal shapes.
"""

# imports
from sklearn.metrics import jaccard_similarity_score as jss

# JI between 2 binary masks
def jaccard_index_binary_masks(truth_mask, predicted_mask):
    return jss(truth_mask, predicted_mask)

