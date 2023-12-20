"""
The structure of our data pipeline and modelling approach has been based off of
the work by UKPLab based in Germany. Their repository may be found here: 
https://github.com/UKPLab/sentence-transformers. We have adapted code from
this repository to be used for our project.

In this script, we evaluate our Siamese model trained with cosine similarity. This script has been adapted for 
our project as well.
"""

import sentence_transformers
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import logging
import requests
import csv
import gzip
from io import BytesIO
import tarfile
import os
import numpy as np

from sentence_transformer_dtu import SentenceTransformer, util
import numpy as np


#logging information
logging.basicConfig(filename='Cos_Siamese_Eval_IR.log', filemode='w', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)


logging.info("Loading model")
# Specify the path to the model
print(os.getcwd())
model_path = "/zhome/d2/a/175738/ML_BERT/DL-SBert/Scripts/siamese_cosine"
#check if path exists
print(os.path.exists(model_path))
logging.info(model_path)
# Load the model
model = SentenceTransformer(model_path)
logging.info("Model loaded!")

#downoad the test queries
logging.info("Download 200 evaluation queries")
# URL of the test query file
url = "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz"
# Download the file

data_folder = 'msmarco-data'
test_queries_filepath = os.path.join(data_folder, 'test_queries.tsv')
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Open the downloaded file using gzip and read its content
    with gzip.open(BytesIO(response.content), 'rt', encoding='utf-8') as file:
        # Create an empty dictionary to store data
        test_queries_dict = {}
        # Parse the TSV file
        tsv_reader = csv.reader(file, delimiter='\t')

        # Store data in the dictionary
        for row in tsv_reader:
            query_id, query_text = row[0], row[1]
            test_queries_dict[query_id] = query_text
    
    # Now data_dict contains the content of the file in a dictionary

    logging.info("Data downloaded and stored test queries in dictionary successfully.")
else:
    logging.info("Failed to download the file.")


#Load the Queries to Relevant passages mapping for the Information Retrieval evaluation
# URL of the file
url = "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-passagetest2019-top1000.tsv.gz"

# Download the file
response = requests.get(url)
logging.info("Starting download of Query to relevant corpus mapping.")
# Check if the request was successful
if response.status_code == 200:
    # Open the downloaded file using gzip and read its content
    with gzip.open(BytesIO(response.content), 'rt', encoding='utf-8') as file:
        # Create an empty dictionary to store data
        query_to_corpus = {}
        # Parse the TSV file
        tsv_reader = csv.reader(file, delimiter='\t')

        # Store data in the dictionary
        for row in tsv_reader:
            query_id, relevant_passage_id = row[0], row[1]
            if query_id in query_to_corpus:
                query_to_corpus[query_id].append(relevant_passage_id)
            else:
                query_to_corpus[query_id] = [relevant_passage_id]
    
        
    
    # Now  query_to_corpus contains all the relevant passages for each query
    logging.info("Query to relevant corpus mapping downloaded and stored in dictionary successfully.")
else:
    logging.info("Failed to download the file.")


# Download Corpus

### Now we read the MS Marco dataset
data_folder = 'msmarco-data'

logging.info("Start downloading large passage corpus")

#### Read the corpus files, that contain all the passages. Store them in the corpus dict
corpus = {}         #dict in the format: passage_id -> passage. Stores all existent passages
collection_filepath = os.path.join(data_folder, 'collection.tsv')
if not os.path.exists(collection_filepath):
    tar_filepath = os.path.join(data_folder, 'collection.tar.gz')
    if not os.path.exists(tar_filepath):
        logging.info("Download collection.tar.gz")
        util.http_get('https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz', tar_filepath)

    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)

logging.info("Read corpus: collection.tsv")
with open(collection_filepath, 'r', encoding='utf8') as fIn:
    for i,line in enumerate(fIn): # now it can be stopped with less than the full corpus!
        if True:
            pid, passage = line.strip().split("\t")
            pid = pid
            corpus[pid] = passage
        else: 
            break

logging.info("Starting Evaluation of the model")

print("lentht is " + str(len(corpus.keys())))
# Evaluate the model

'''This evaluation function should be fed with a large corpus (the one used for training)
in addition we need a list of queries and a list of relevant documents for each query
this gives us a score in three different metrics: Mean Reciprocal Rank (MRR), Recall@k, and Normalized Discounted Cumulative Gain'''


evaluator = InformationRetrievalEvaluator(queries=test_queries_dict, corpus=corpus, relevant_docs=query_to_corpus,show_progress_bar=True,mrr_at_k=[1,5,10],ndcg_at_k= [1,5,10], accuracy_at_k = [1, 5, 10], precision_recall_at_k = [1, 5, 10])
logging.info("Evaluator made")

metrics=evaluator.compute_metrices(model)

for key in metrics.keys():
    logging.info(f"The results for {key} is {metrics[key]}")
