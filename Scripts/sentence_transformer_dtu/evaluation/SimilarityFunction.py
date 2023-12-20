"""
The structure of our data pipeline and modelling approach has been based off of
the work by UKPLab based in Germany. Their repository may be found here: 
https://github.com/UKPLab/sentence-transformers. We have adapted code from
this repository to be used for our project.

This script was created by UKPLab.
"""

from enum import Enum

class SimilarityFunction(Enum):
    COSINE = 0
    EUCLIDEAN = 1
    MANHATTAN = 2
    DOT_PRODUCT = 3

