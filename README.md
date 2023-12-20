# Siamese vs. Triplet SBERT Networks for Information Retrieval
## Repository Details

The structure of our data pipeline and modelling approach has been based off of the work by UKPLab based in Germany. Their repository may be found here: https://github.com/UKPLab/sentence-transformers. We have adapted code from this repository to be used for our project.

In this repository, the structure is as follows:
- 


guide til directory


## Project Summary

In today's society with increasing amounts of data, efficient information retrieval demands advanced NLP models. While BERT models excel in NLP tasks, its direct use for information retrieval faces limitations due to scalability issues of pairwise comparisons and poor embedding abilities. To harness BERT's NLP capabilities for better sentence embedding, we employ a Siamese and Triplet network structure, yielding a Sentence-BERT (SBERT) model. This is trained to produce sentence level embeddings enabling information retrieval using cheap distance/similarity measures.

Our study compares Siamese and Triplet SBERT models for information retrieval using cosine and euclidean similarity/distance measures. Our model architechture can be seen below:

![Network_architecture (1)](https://github.com/annabramsloew/DL-SBert/assets/80269825/5c626d14-da95-4360-b459-0dccdda0f624)

The project aims to understand the similarities and differences between the two model structures when using cosine similarity and euclidean distance to compare embedding vectors. Specifically, we train four different SBERT models using the publicly available MS Marco data set. These four models are evaluated on their performance in information retrieval across four different metrics; precision, accuracy, Mean Reciprocal Rank (MRR) and Normalised Discounted Cumulative Gain (NDCG).

Notably, the Siamese model prefers training with cosine similarity, while Triplet strongly prefers training with euclidean distance. Furthermore, a qualitative visual inspection of embedding space was performed using PCA. This revealed a grouping effect in the Triplet cosine model, possibly explaining the poor performance of this model. Overall, the Siamese cosine model has the shortest training time of the two best performing models.

An excerpt of our results can be viwed below:

![PCA_and_results](https://github.com/annabramsloew/DL-SBert/assets/80269825/df5bb271-8917-469a-b4e1-204ebc20c1ee)
