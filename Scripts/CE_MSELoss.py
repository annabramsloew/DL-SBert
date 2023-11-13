from sentence_transformers import util
from torch import nn, Tensor
from typing import Iterable, Dict

class CE_MSELoss(nn.Module):
    """
    Compute the MSE loss between the |sim(Query, Pos)| and |gold_sim(Q, Pos)|.
    By default, sim() is the dot-product. Here we use cosine similarity.
    For more details, please refer to https://arxiv.org/abs/2010.02666.
    """
    def __init__(self, model, similarity_fct = util.cos_sim):
        """
        :param model: SentenceTransformerModel
        :param similarity_fct:  Which similarity function to use.
        """
        super(CE_MSELoss, self).__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.loss_fct = nn.MSELoss()

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        # sentence_features: query, positive passage, negative passage
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_query = reps[0]
        embeddings_pos = reps[1]
        embeddings_neg = reps[2]

        scores_pos = self.similarity_fct(embeddings_query, embeddings_pos) 
        #scores_neg = self.similarity_fct(embeddings_query, embeddings_neg)
        #margin_pred = scores_pos - scores_neg

        return self.loss_fct(scores_pos, labels) # here we could add the scores_neg and score against labels[1] if these were a list
