# Siamese vs. Triplet SBERT Networks for Information Retrieval

credits til andre

guide til directory

In today's society with increasing amounts of data, efficient information retrieval demands advanced NLP models. While BERT models excel in NLP tasks, its direct use for information retrieval faces limitations due to scalability issues of pairwise comparisons and poor embedding abilities. To harness BERT's NLP capabilities for better sentence embedding, we employ a Siamese and Triplet network structure, yielding a Sentence-BERT (SBERT) model. This is trained to produce sentence level embeddings enabling information retrieval using cheap distance/similarity measures.

![Network_architecture (1)](https://github.com/annabramsloew/DL-SBert/assets/80269825/5c626d14-da95-4360-b459-0dccdda0f624)


Our study compares Siamese and Triplet SBERT models for information retrieval using cosine and euclidean similarity/distance measures. Notably, the Siamese model prefers training with cosine similarity, while Triplet strongly prefers training with euclidean distance. Furthermore, a qualitative visual inspection of embedding space was performed using PCA. This revealed a grouping effect in the Triplet cosine model, possibly explaining the poor performance of this model. Overall, the Siamese cosine model has the shortest training time of the two best performing models.
