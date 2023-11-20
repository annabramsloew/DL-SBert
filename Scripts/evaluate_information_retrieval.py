
# Evaluate the model
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from io import BytesIO
import requests
import csv
import gzip
import os
import logging
from sentence_transformers import SentenceTransformer, LoggingHandler, util, models, evaluation, losses, InputExample
from torch.utils.data import DataLoader


'''this evaluation function should be fed with a large corpus (the one used for training)
in addition we need a list of queries and a list of relevant documents for each query
this gives us a score in three different metrics: Mean Reciprocal Rank (MRR), Recall@k, and Normalized Discounted Cumulative Gain'''

#logging information
logging.basicConfig(filename='train_siamese.log', filemode='w', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)

### Now we read the MS Marco dataset
data_folder = 'msmarco-data'


logging.info("Download Large Corpus")
#### Read the corpus files, that contain all the passages. Store them in the corpus dict
corpus = {}         #dict in the format: passage_id -> passage. Stores all existent passages
collection_filepath = os.path.join(data_folder, 'collection.tsv')
if not os.path.exists(collection_filepath):
    tar_filepath = os.path.join(data_folder, 'collection.tar.gz')
    if not os.path.exists(tar_filepath):
        logging.info("Download collection.tar.gz")
        util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz', tar_filepath)

    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)

logging.info("Read corpus: collection.tsv")
with open(collection_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        pid = int(pid)
        corpus[pid] = passage

logging.info("Download evaluation queries")

# URL of the file
url = "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz"

# Download the file
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

    logging.info("Data downloaded and stored in dictionary successfully.")
else:
    logging.info("Failed to download the file.")


#load the queries to relevant passages mapping for the Information Retrieval evaluation

# URL of the file
url = "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-passagetest2019-top1000.tsv.gz"
# Download the file
response = requests.get(url)
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


# Some code that loads a trained model

model=#code here

evaluator = InformationRetrievalEvaluator(queries=test_queries_dict, corpus=corpus, relevant_docs=query_to_corpus)

model.evaluate(evaluator)
