"""
Jaccard similarity module.

Contains functions to calculate the Jaccard similarity index between 2 binary
masks or multipolygonal shapes.
"""

from sklearn.metrics import jaccard_similarity_score as jss


# JI between 2 binary masks
def jaccard_index_binary_masks(truth_mask, predicted_mask):
    return jss(truth_mask, predicted_mask, normalize=True)


# JI between 2 multipolygons
def jaccard_index_multipolygons(truth_multi, predicted_multi):
    if not (truth_multi.is_valid):
        raise ('The truth multipolygon is not valid!')
    if not (predicted_multi.is_valid):
        raise ('The predicted multipolygon is not valid!')

    # intersection
    intersec = truth_multi.intersection(predicted_multi).area
    # union
    union = truth_multi.union(predicted_multi).area

    # Jaccard index is intersection over union
    return intersec / union
