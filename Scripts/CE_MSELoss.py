from sentence_transformer_dtu import util
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Iterable, Dict

class CE_MSELoss(nn.Module):
    """
    Compute the MSE loss between the |sim(Query, Pos)| and |gold_sim(Q, Pos)|.
    By default, sim() is the dot-product. Here we use cosine similarity.
    For more details, please refer to https://arxiv.org/abs/2010.02666.
    """
    def __init__(self, model, metric = "euclidian"): #util.pairwise_cos_sim
        """
        :param model: SentenceTransformerModel
        :param similarity_fct:  Which similarity function to use.
        """
        super(CE_MSELoss, self).__init__()
        self.model = model
        self.similarity_fct = metric
        self.loss_fct = nn.MSELoss()
        
    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        # sentence_features: query, positive passage, negative passage
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_query = reps[0]
        embeddings_pos = reps[1]
        #embeddings_neg = reps[2]
        if self.similarity_fct == "euclidian":
            scores_pos = F.pairwise_distance(embeddings_query, embeddings_pos, p=2)
            labels_euclid = 1 - labels
            return self.loss_fct(scores_pos, labels_euclid)
        elif self.similarity_fct == "cosine":
            scores_pos = util.pairwise_cos_sim(embeddings_query, embeddings_pos)
            return self.loss_fct(scores_pos, labels)
        else:
            break
        #scores_neg = self.similarity_fct(embeddings_query, embeddings_neg)
        #margin_pred = scores_pos - scores_neg

         # here we could add the scores_neg and score against labels[1] if these were a list
