"""
The structure of our data pipeline and modelling approach has been based off of
the work by UKPLab based in Germany. Their repository may be found here: 
https://github.com/UKPLab/sentence-transformers. We have adapted code from
this repository to be used for our project.

In this script, we train our siamese model. This script has been adapted for 
our project as well - it is based on the triplet model script.
"""

import sys
import json
from torch.utils.data import DataLoader
from sentence_transformer_dtu import SentenceTransformer, util, models, InputExample
import logging
from datetime import datetime
import gzip
import os
import tarfile
import tqdm
from torch.utils.data import Dataset
import random
from shutil import copyfile
import pickle
import argparse
from CE_MSELoss import CE_MSELoss
import numpy as np


#### Just some code to print debug information to stdout
logging.basicConfig(filename='train_siamese.log', filemode='w', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
#### /print debug information to stdout


parser = argparse.ArgumentParser()
parser.add_argument("--train_batch_size", default=64, type=int)
parser.add_argument("--max_seq_length", default=300, type=int)
parser.add_argument("--model_name", default='distilbert-base-uncased')
parser.add_argument("--max_passages", default=0, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--pooling", default="mean")
parser.add_argument("--negs_to_use", default=None, help="From which systems should negatives be used? Multiple systems seperated by comma. None = all")
parser.add_argument("--warmup_steps", default=0, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--num_negs_per_system", default=5, type=int)
parser.add_argument("--use_pre_trained_model", default=False, action="store_true")
parser.add_argument("--use_all_queries", default=False, action="store_true")
parser.add_argument("--ce_score_margin", default=3.0, type=float)
parser.add_argument("--evaluation_steps", default=10000, type=int)
parser.add_argument("--metric", default='cosine', type=str) #either cosine or euclidean
args = parser.parse_args()

print(args)
logging.info(str(args))


# The  model we want to fine-tune
train_batch_size = args.train_batch_size          #Increasing the train batch size improves the model performance, but requires more GPU memory
model_name = args.model_name
max_passages = args.max_passages
ce_score_margin = args.ce_score_margin
max_seq_length = args.max_seq_length            #Max length for passages. Increasing it, requires more GPU memory

num_negs_per_system = args.num_negs_per_system  # We used different systems to mine hard negatives. Number of hard negatives to add from each system
num_epochs = args.epochs         # Number of epochs we want to train

# Load our embedding model
if args.use_pre_trained_model:
    logging.info("use pretrained SBERT model")
    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_length
else:
    logging.info("Create new SBERT model")
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), args.pooling)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

model_save_path = f'output/train_siamese_sbert-{model_name.replace("/", "-")}-batch_size_{train_batch_size}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'


# Write self to path
os.makedirs(model_save_path, exist_ok=True)

train_script_path = os.path.join(model_save_path, 'train_script.py')
copyfile(__file__, train_script_path)
with open(train_script_path, 'a') as fOut:
    fOut.write("\n\n# Script was called via:\n#python " + " ".join(sys.argv))


### Now we read the MS Marco dataset
data_folder = 'msmarco-data'

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


### Read the train queries, store in queries dict
queries = {}        #dict in the format: query_id -> query. Stores all training queries
queries_filepath = os.path.join(data_folder, 'queries.train.tsv')
if not os.path.exists(queries_filepath):
    tar_filepath = os.path.join(data_folder, 'queries.tar.gz')
    if not os.path.exists(tar_filepath):
        logging.info("Download queries.tar.gz")
        util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz', tar_filepath)

    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)


with open(queries_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        qid = int(qid)
        queries[qid] = query


# Load a dict (qid, pid) -> ce_score that maps query-ids (qid) and paragraph-ids (pid)
# to the CrossEncoder score computed by the cross-encoder/ms-marco-MiniLM-L-6-v2 model
ce_scores_file = os.path.join(data_folder, 'cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz')
if not os.path.exists(ce_scores_file):
    logging.info("Download cross-encoder scores file")
    util.http_get('https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz', ce_scores_file)

logging.info("Load CrossEncoder scores dict")
with gzip.open(ce_scores_file, 'rb') as fIn:
    ce_scores = pickle.load(fIn)

#added sigmoid function
def sigmoid_np(x):
    return 1 / (1 + np.exp(-np.array(x)))

#data pipeline created for Siamese model
logging.info("Read hard negatives train file")
hard_negatives_filepath = os.path.join(data_folder, 'msmarco-hard-negatives.jsonl.gz')
train_queries = {}
negs_to_use = None
with gzip.open(hard_negatives_filepath, 'rt') as fIn:
    for line in tqdm.tqdm(fIn):
        data = json.loads(line)

        #Get the positive passage ids

        qid = data['qid']
        pos_pids = data['pos']

        #if qid not in unique_qids and pos_pids not in unique_pids:
          #continue

        if len(pos_pids) == 0:  #Skip entries without positives passages
            continue

        pos_min_ce_score = min([ce_scores[qid][pid] for pid in data['pos']])
        ce_score_threshold = pos_min_ce_score - ce_score_margin

        #Get the hard negatives
        neg_pids = set()
        if negs_to_use is None:
            if args.negs_to_use is not None:    #Use specific system for negatives
                negs_to_use = args.negs_to_use.split(",")
            else:   #Use all systems
                negs_to_use = list(data['neg'].keys())
            logging.info("Using negatives from the following systems: {}".format(", ".join(negs_to_use)))

        for system_name in negs_to_use:
            if system_name not in data['neg']:
                continue

            system_negs = data['neg'][system_name]
            negs_added = 0
            for pid in system_negs:
                #if pid not in unique_pids:
                 #   continue

                if ce_scores[qid][pid] > ce_score_threshold:
                    continue

                if pid not in neg_pids:
                    neg_pids.add(pid)
                    negs_added += 1
                    if negs_added >= num_negs_per_system:
                        break
        #print("Got here")
        if args.use_all_queries or (len(pos_pids) > 0 and len(neg_pids) > 0):
            train_queries[data['qid']] = {'qid': data['qid'], 'query': queries[data['qid']], 'pos': pos_pids, 'neg': neg_pids}

#del ce_scores

logging.info("Train queries: {}".format(len(train_queries)))


# We create a custom MSMARCO dataset that returns triplets (query, positive, negative)
# on-the-fly based on the information from the mined-hard-negatives jsonl file.
class MSMARCODataset(Dataset):
    def __init__(self, queries, corpus, ce_scores):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus
        
        # make list to store the 
        self.anchors = []
        self.passages = []
        
        #added query limit
        MAX_QUERIES = 125000 #125000
        for i, qid in enumerate(self.queries):
            if i >= MAX_QUERIES:
                continue
            # add the one positive example ((in some cases there are two, but these are disregarded))
            pos_id = self.queries[qid]['pos'][0]
            self.anchors.append(qid)
            self.passages.append(pos_id)

            # add the many negatives
            negs = self.queries[qid]['neg']
            neg_count = len(negs)
            
            # append same length of anchors and negative passages
            self.anchors += [qid]*neg_count
            self.passages += negs
                
        assert len(self.passages) == len(self.anchors)
        logging.info(f"Total examples: {len(self.anchors)}")
        print(f"Total examples: {len(self.anchors)}")
        ce_scores_list = [ce_scores[qid][pid] for qid,pid in zip(self.anchors, self.passages)]
        self.ce_scores = [float(number) for number in sigmoid_np(ce_scores_list)]

    def __getitem__(self, item):
        query = self.queries[self.anchors[item]]
        query_text = query['query']

        passage_id = self.passages[item]  
        passage_text = self.corpus[passage_id]

        score = self.ce_scores[item]

        return InputExample(texts=[query_text, passage_text], label=score)

    def __len__(self):
        return len(self.anchors)



# For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
train_dataset = MSMARCODataset(queries=train_queries, corpus=corpus, ce_scores=ce_scores)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = CE_MSELoss(model=model, metric=args.metric)

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          warmup_steps=args.warmup_steps,
          use_amp=True,
          checkpoint_path=model_save_path,
          checkpoint_save_steps=10000,
          optimizer_params = {'lr': args.lr},
          evaluation_steps = args.evaluation_steps,
          checkpoint_save_total_limit = 5
          )

# save latest model
model.save(model_save_path)

