import os
import tarfile
import sys
import json
from torch.utils.data import DataLoader
from sentence_transformer_dtu import SentenceTransformer, util, models, InputExample
import logging
from datetime import datetime
import gzip
import os
import tarfile
from collections import defaultdict
from torch.utils.data import IterableDataset
import tqdm
from torch.utils.data import Dataset
import random
import pickle
import argparse
import numpy as np
import joblib
from torch.utils.data import DataLoader

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--train_batch_size", default=64, type=int)
parser.add_argument("--max_seq_length", default=300, type=int)
parser.add_argument("--model_name",default="bert-base-uncased")
parser.add_argument("--max_passages", default=0, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--pooling", default="mean")
parser.add_argument("--negs_to_use", default=None, help="From which systems should negatives be used? Multiple systems seperated by comma. None = all")
parser.add_argument("--warmup_steps", default=1000, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--num_negs_per_system", default=5, type=int)
parser.add_argument("--use_pre_trained_model", default=False, action="store_true")
parser.add_argument("--use_all_queries", default=False, action="store_true")
parser.add_argument("--ce_score_margin", default=3.0, type=float)
parser.add_argument("--evaluation_steps", default=10000, type=int)
args = parser.parse_args()

#logging information
logging.basicConfig(filename='PCA_cos_siamese.log', filemode='w', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)



# First we want to load the corpus, queries and CE scores

### Now we read the MS Marco dataset
data_folder = "/zhome/d2/a/175738/ML_BERT/DL-SBert/msmarco-data"

#### Read the corpus files, that contain all the passages. Store them in the corpus dict
corpus = {}         #dict in the format: passage_id -> passage. Stores all existent passages
corpus_list=[]
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
    for i, line in enumerate(fIn):
        #if i <= 100:
        pid, passage = line.strip().split("\t")
        pid = int(pid)
        corpus[pid] = passage
        corpus_list.append(passage)


### Read the train queries, store in queries dict
queries = {}        #dict in the format: query_id -> query. Stores all training queries
query_list=[]
queries_filepath = os.path.join(data_folder, 'queries.train.tsv')
if not os.path.exists(queries_filepath):
    tar_filepath = os.path.join(data_folder, 'queries.tar.gz')
    if not os.path.exists(tar_filepath):
        logging.info("Download queries.tar.gz")
        util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz', tar_filepath)

    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)


with open(queries_filepath, 'r', encoding='utf8') as fIn:
    for i, line in enumerate(fIn):
        #if i <= 100:
        qid, query = line.strip().split("\t")
        qid = int(qid)
        queries[qid] = query
        query_list.append(query)

#unique_qids = set(queries.keys())
#unique_pids = set(corpus.keys())

# Load a dict (qid, pid) -> ce_score that maps query-ids (qid) and paragraph-ids (pid)
# to the CrossEncoder score computed by the cross-encoder/ms-marco-MiniLM-L-6-v2 model
ce_scores_file = os.path.join(data_folder, 'cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz')
if not os.path.exists(ce_scores_file):
    logging.info("Download cross-encoder scores file")
    util.http_get('https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz', ce_scores_file)

logging.info("Load CrossEncoder scores dict")
with gzip.open(ce_scores_file, 'rb') as fIn:
      ce_scores = pickle.load(fIn)

# As training data we use hard-negatives that have been mined using various systems
hard_negatives_filepath = os.path.join(data_folder, 'msmarco-hard-negatives.jsonl.gz')
if not os.path.exists(hard_negatives_filepath):
    logging.info("Download cross-encoder scores file")
    util.http_get('https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/msmarco-hard-negatives.jsonl.gz', hard_negatives_filepath)

logging.info("Read hard negatives train file")
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
        ce_score_threshold = pos_min_ce_score - args.ce_score_margin

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
                    if negs_added >= args.num_negs_per_system:
                        break
        #print("Got here")
        if args.use_all_queries or (len(pos_pids) > 0 and len(neg_pids) > 0):
            train_queries[data['qid']] = {'qid': data['qid'], 'query': queries[data['qid']], 'pos': pos_pids, 'neg': neg_pids}



logging.info("Train queries: {}".format(len(train_queries)))



# We create a custom MSMARCO dataset that returns triplets (query, positive, negative)
# on-the-fly based on the information from the mined-hard-negatives jsonl file.
class MSMARCODataset(Dataset):
    def __init__(self, queries, corpus):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        #self.pct5_threshold = int(query_total*0.05)
        self.corpus = corpus
        
        # make list to store the 
        self.anchors = []
        self.positives = []
        self.negatives = []

        for qid in self.queries:

            # store the negatives for the positive examples
            negs = self.queries[qid]['neg']
            neg_count = len(negs)
            
            # only take the first positive (in some cases there are two, but these are disregarded)
            pos_id = self.queries[qid]['pos'][0]
            
            # append same length of anchors, negatives and positives
            self.anchors += [qid]*neg_count
            self.positives += [pos_id]*neg_count
            self.negatives += negs
                
            #self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
            #self.queries[qid]['neg'] = list(self.queries[qid]['neg'])
            #random.shuffle(self.queries[qid]['neg'])
        assert len(self.negatives) == len(self.positives)
        assert len(self.negatives) == len(self.anchors)
        logging.info(f"Total examples: {len(self.anchors)}")

    def __getitem__(self, item):
        query = self.queries[self.anchors[item]]
        query_text = query['query']

        pos_id = self.positives[item]  
        pos_text = self.corpus[pos_id]

        neg_id = self.negatives[item]  
        neg_text = self.corpus[neg_id]

        return InputExample(texts=[query_text, pos_text, neg_text])

    def __len__(self):
        return len(self.anchors)

    

# Now we load the model that we have trained


logging.info("Loading model")
# Specify the path to the model
print(os.getcwd())
#Absolute path to model
model_path = "/zhome/d2/a/175738/ML_BERT/DL-SBert/Scripts/siamese_cosine"
#check if path exists
logging.info(os.path.exists(model_path))
logging.info(model_path)
# Load the model
model = SentenceTransformer(model_path)
logging.info("Model loaded!")


#now we encode the whole corpus
corpus_query=corpus_list+query_list
logging.info(f"Proceeding to encode {len(corpus_query)} queries and passages")
logging.info(f'We have {len(corpus_list)} passages in corpus and {len(query_list)} queries')

#try a small subset
train_dataset = MSMARCODataset(train_queries, corpus=corpus)
corp_quer=[]
for idx in range(70000):
    query = train_dataset.queries[train_dataset.anchors[idx]]
    query_text = query['query']
    
    pos_id = train_dataset.positives[idx]  
    pos_text = train_dataset.corpus[pos_id]

    neg_id = train_dataset.negatives[idx]  
    neg_text = train_dataset.corpus[neg_id]

    corp_quer.append(query_text)
    corp_quer.append(pos_text)
    corp_quer.append(neg_text)

logging.info(f"encoding {len(corp_quer)} texts")
encodings=model.encode(corp_quer,show_progress_bar=True,normalize_embeddings=True) #corpus_query[:2000]

#Doing the PCA
pca=PCA(n_components=2)
data_2d=encodings

#Normalize the data 
mean_data_set=np.mean(data_2d, axis=0)
std_data_set=np.std(data_2d, axis=0)

data_normalized = (data_2d - mean_data_set) / std_data_set

pca.fit(data_normalized) # used to be data_normalized
logging.info(f"PCA fit got {pca.explained_variance_ratio_}")


# Transform the data to reduced dimensions
query_encode=model.encode(query_list[:100])
query_encode_normalised=(query_encode-mean_data_set)/std_data_set

corpus_encode=model.encode(corpus_list[:100])
corpus_encode_normalised=(corpus_encode-mean_data_set)/std_data_set

queries_reduced = pca.transform(query_encode_normalised)
corpus_reduced = pca.transform(corpus_encode_normalised)

#Hardcode a triplets

# For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
train_dataset = MSMARCODataset(train_queries, corpus=corpus)
indexes=[0,10000,30000,50000,60000]
triplets=[]
for idx in indexes:
    query = train_dataset.queries[train_dataset.anchors[idx]]
    query_text = query['query']

    pos_id = train_dataset.positives[idx]  
    pos_text = train_dataset.corpus[pos_id]

    neg_id = train_dataset.negatives[idx]  
    neg_text = train_dataset.corpus[neg_id]
    triplets.append([query_text,pos_text,neg_text])
logging.info([x for x in triplets[:5]])
triplets_encoded_norm=[]
triplets_reduced=[]
for triplet in triplets:
    
    temp_encode=model.encode(triplet,normalize_embeddings=True) # maybe we are double normalising...
    temp_encode_norm=(temp_encode-mean_data_set)/std_data_set
    triplets_reduced.append(pca.transform(temp_encode_norm)) # used to be temp_encode_norm

    #calculate the pairwise cos similiarity
    quer_pos=np.dot(temp_encode[0],temp_encode[1])/(np.linalg.norm(temp_encode[0])*np.linalg.norm(temp_encode[1]))
    quer_neg=np.dot(temp_encode[0],temp_encode[2])/(np.linalg.norm(temp_encode[0])*np.linalg.norm(temp_encode[2]))
    logging.info(f"The triplet {triplet} has the cos similarity of query to pos of {quer_pos} and query to neg of {quer_neg}")

# Plotting the embeddings in 2D
plt.figure(figsize=(8, 6))
#plt.scatter(queries_reduced[:, 0], queries_reduced[:, 1], alpha=0.2,label="queries")
#plt.scatter(corpus_reduced[:, 0], corpus_reduced[:, 1], alpha=0.2,label="corpus",color="red",marker="x")
label_dict={0:"Query",1:"Pos",2:"Neg"}
anot_dict={0:"Q",1:"P",2:"N"}
marker_dict={0:"x",1:".",2:"v"}
colors_dict={0:"green",1:"teal",2:"olive",3:"orangered",4:"indigo",5:"darkorchid",6:"red"}
for x,triplet_reduced in enumerate(triplets_reduced):
    for i,sent in enumerate(triplet_reduced):
        plt.scatter(sent[0],sent[1],color=colors_dict[x],label=label_dict[i],marker=marker_dict[i],s=200)
        #plt.annotate(anot_dict[i]+str(x+1),(sent[0],sent[1]))
#plt.legend()
plt.title('2D Embeddings after PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
#plt.grid(True)
plt.show()
plt.savefig('pca_queries_and_corpus_cos_siamese.png')

joblib.dump(pca, 'pca_model_cos_siamese.pkl') 