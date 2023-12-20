"""
The structure of our data pipeline and modelling approach has been based off of
the work by UKPLab based in Germany. Their repository may be found here: 
https://github.com/UKPLab/sentence-transformers. We have adapted code from
this repository to be used for our project.

In this script, we define the loss functions for our triplet model. This script has been adapted for 
our project as well.
"""

import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
import torch.nn.functional as F
from enum import Enum
from sentence_transformers import SentenceTransformer

class TripletDistanceMetric(Enum):
    """
    The metric for the triplet loss
    """
    COSINE = lambda x, y: 1 - F.relu(F.cosine_similarity(x, y))
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)

class TripletLoss(nn.Module):
    """
    This class implements triplet loss. Given a triplet of (anchor, positive, negative),
    the loss minimizes the distance between anchor and positive while it maximizes the distance
    between anchor and negative. It compute the following loss function:

    loss = max(||anchor - positive|| - ||anchor - negative|| + margin, 0).

    Margin is an important hyperparameter and needs to be tuned respectively.

    :param model: SentenceTransformerModel
    :param distance_metric: Function to compute distance between two embeddings. The class TripletDistanceMetric contains common distance metrices that can be used.
    :param triplet_margin: The negative should be at least this much further away from the anchor than the positive.
    """
    def __init__(self, model: SentenceTransformer, metric='cosine', triplet_margin: float = 5):
        super(TripletLoss, self).__init__()
        self.model = model

        if metric == 'cosine':
            self.distance_metric=TripletDistanceMetric.COSINE
        elif metric == 'euclidian':
            self.distance_metric=TripletDistanceMetric.EUCLIDEAN
        else:
            break

        self.distance_metric = distance_metric
        self.triplet_margin = triplet_margin


    def get_config_dict(self):

        distance_metric_name = self.distance_metric.__name__
        for name, value in vars(TripletDistanceMetric).items():
            if value == self.distance_metric:
                distance_metric_name = "TripletDistanceMetric.{}".format(name)
                break

        return {'distance_metric': distance_metric_name, 'triplet_margin': self.triplet_margin}

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]

        rep_anchor, rep_pos, rep_neg = reps
        distance_pos = self.distance_metric(rep_anchor, rep_pos)
        distance_neg = self.distance_metric(rep_anchor, rep_neg)

        losses = F.relu(distance_pos - distance_neg + self.triplet_margin)
        return losses.mean()
