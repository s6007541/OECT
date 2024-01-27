import logging
import os
import random
import re
import string
import uuid

from argparse import ArgumentParser
from typing import Sequence

import nltk
nltk.download("punkt")
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm

import pathlib

from filelock import FileLock
from sib import SIB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# import clusters algorithms
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, DBSCAN, OPTICS

STOP_WORDS_FILE = './stop_words.txt'
# OUT_DIR = './output'
# tf.config.run_functions_eagerly(True)

def remove_stop_words_and_punctuation(texts):
    with open(STOP_WORDS_FILE, 'r') as f:
        stop_words = [line.strip() for line in f if line.strip()]
    escaped_stop_words = [re.escape(stop_word) for stop_word in stop_words]
    regex_pattern = r"\b(" + "|".join(escaped_stop_words) + r")\b"
    # remove stop words
    texts = [re.sub(r" +", r" ", re.sub(regex_pattern, "", str(text).lower())).strip() for text in texts]
    # remove punctuation
    texts = [t.translate(t.maketrans(string.punctuation, ' ' * len(string.punctuation))) for t in texts]
    return [' '.join(t.split()) for t in texts]

def stem(texts):
    stemmer = nltk.SnowballStemmer("english", ignore_stopwords=False)
    return [" ".join(stemmer.stem(word).lower().strip()
            for word in nltk.word_tokenize(text)) for text in texts]

def get_embeddings(texts):
    # apply text processing to prepare the texts
    texts = remove_stop_words_and_punctuation(texts)
    texts = stem(texts)

    # create the vectorizer and transform data to vectors
    vectorizer = TfidfVectorizer(max_df=1.0, 
                                 min_df=1, 
                                 max_features=10000, 
                                 stop_words=None, 
                                 use_idf=False, 
                                 norm=None)
    vectors = vectorizer.fit_transform(texts)
    return vectors

def get_cluster_labels_SECF(texts, 
                            n_clusters, 
                            clustering_algo, 
                            model_name_or_path="bert-base-uncased", 
                            num_labels=200, 
                            infer_batch_size = 32):
    embedding_vectors = get_embeddings(texts)
    # logging.info('Finished generating embedding vectors')
    
    # feature extraction first'  
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    logging.info(f'running inference using model {model_name_or_path} on {len(texts)} texts')
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_labels)
    embeddings = []
    for x in tqdm.tqdm(range(0, len(texts), infer_batch_size)):
        batch_texts = texts[x:x + infer_batch_size]
        batch_input = tokenizer(batch_texts, 
                                padding=True, 
                                truncation=True, 
                                return_tensors="tf")
        batch_res = model(batch_input).logits.numpy().tolist()
        embeddings.extend(batch_res)

    embeddings = np.array(embeddings)
    print("extraction done!")

    if clustering_algo == "sib":
        algorithm = SIB(n_clusters=n_clusters, 
                        n_init=10, 
                        n_jobs=-1, 
                        max_iter=15, 
                        random_state=1024, 
                        tol=0.02)
        
    elif clustering_algo == "kmeans":
        algorithm = KMeans(n_clusters=n_clusters, 
                           init='k-means++', 
                           n_init = 3, 
                           max_iter=300, 
                           tol=0.0001, 
                           verbose=0, 
                           random_state=None, 
                           copy_x=True, 
                           algorithm='auto')
        
    elif clustering_algo == "affinity":
        algorithm = AffinityPropagation(damping=0.5, 
                                        max_iter=200, 
                                        convergence_iter=15, 
                                        copy=True, 
                                        preference=None, 
                                        affinity='euclidean', 
                                        verbose=False, 
                                        random_state=None)
        
    elif clustering_algo == "meanshift":
        algorithm = MeanShift(bandwidth=None, 
                              seeds=None, 
                              bin_seeding=False, 
                              min_bin_freq=1, 
                              cluster_all=True, 
                              n_jobs=None, 
                              max_iter=300)
        
    elif clustering_algo == "OPTICS":
        algorithm = OPTICS(min_samples=5 , 
                           metric='minkowski', 
                           p=2, 
                           metric_params=None, 
                           cluster_method='xi', 
                           eps=None, xi=0.05, 
                           predecessor_correction=True, 
                           min_cluster_size=None, 
                           algorithm='auto', 
                           leaf_size=30, 
                           memory=None, 
                           n_jobs=1)
        
    elif clustering_algo == "DBSCAN":
        algorithm = DBSCAN(eps=11, 
                           min_samples=5, 
                           metric='euclidean',
                           metric_params=None, 
                           algorithm='auto', 
                           leaf_size=30, 
                           p=None, 
                           n_jobs=1)
        
    else:
        raise NotImplementedError
    
    clustering_model = algorithm.fit(embeddings)
    
    logging.info(f'Finished clustering embeddings for {len(texts)} texts into {n_clusters} clusters')
    cluster_labels = clustering_model.labels_.tolist()
    
    if args.soft_label:
        soft_labels = []
        centroids = clustering_model.cluster_centers_
        for emb in embedding_vectors:
            soft_label = []
            for centroid_i in range(len(centroids)):
                
                soft_label += [1/np.linalg.norm(emb -centroids[centroid_i].reshape(1, -1))]
        
            soft_labels += [soft_label]
            
        soft_labels = np.array(soft_labels)
        for slb in soft_labels:
            slb[:] = 0
            slb[slb.argmax()] = 1
        soft_labels = soft_labels/soft_labels.sum(axis = 1, keepdims = 1)
        
        
        
        return soft_labels
    else:
        return cluster_labels
    
def get_cluster_labels(texts, n_clusters, clustering_algo):
    embedding_vectors = get_embeddings(texts)
    logging.info('Finished generating embedding vectors')
    # print(embedding_vectors)
    # decide clustering algorithms
    print(embedding_vectors.toarray().shape)
    
    if clustering_algo == "sib":
        algorithm = SIB(n_clusters=n_clusters, 
                        n_init=10, 
                        n_jobs=-1, 
                        max_iter=15, 
                        random_state=1024, 
                        tol=0.02)
        clustering_model = algorithm.fit(embedding_vectors)
    elif clustering_algo == "kmeans":
        algorithm = KMeans(n_clusters=n_clusters,
                           init='k-means++',
                           n_init = 3,
                           max_iter=300,
                           tol=0.0001,
                           verbose=0,
                           random_state=None,
                           copy_x=True,
                           algorithm='auto')
        clustering_model = algorithm.fit(embedding_vectors.toarray())
    elif clustering_algo == "affinity":
        algorithm = AffinityPropagation(damping=0.5,
                                        max_iter=200,
                                        convergence_iter=15,
                                        copy=True,
                                        preference=None,
                                        affinity='euclidean',
                                        verbose=False,
                                        random_state=None)
        clustering_model = algorithm.fit(embedding_vectors.toarray())
    elif clustering_algo == "meanshift":
        algorithm = MeanShift(bandwidth=None,
                              seeds=None,
                              bin_seeding=False,
                              min_bin_freq=1,
                              cluster_all=True,
                              n_jobs=None,
                              max_iter=300)
        clustering_model = algorithm.fit(embedding_vectors.toarray())
    elif clustering_algo == "OPTICS":
        algorithm = OPTICS(min_samples=5 ,
                           metric='minkowski',
                           p=2,
                           metric_params=None,
                           cluster_method='xi',
                           eps=None,
                           xi=0.05,
                           predecessor_correction=True,
                           min_cluster_size=None,
                           algorithm='auto',
                           leaf_size=30,
                           memory=None,
                           n_jobs=-1)
        clustering_model = algorithm.fit(embedding_vectors.toarray())
    elif clustering_algo == "DBSCAN":
        algorithm = DBSCAN(eps=11,
                           min_samples=5,
                           metric='euclidean',
                           metric_params=None,
                           algorithm='auto',
                           leaf_size=30,
                           p=None,
                           n_jobs=-1)
        clustering_model = algorithm.fit(embedding_vectors)
    else:
        raise NotImplementedError
    
    logging.info(f'Finished clustering embeddings for {len(texts)} texts into {n_clusters} clusters')
    cluster_labels = clustering_model.labels_.tolist()

    if args.soft_label:
        soft_labels = []
        centroids = clustering_model.cluster_centers_
        for emb in embedding_vectors:
            soft_label = []
            for centroid_i in range(len(centroids)):
                
                soft_label += [1/np.linalg.norm(emb -centroids[centroid_i].reshape(1, -1))]
        
            soft_labels += [soft_label]
            
        soft_labels = np.array(soft_labels)
        for slb in soft_labels:
            slb[:] = 0
            slb[slb.argmax()] = 1
        soft_labels = soft_labels/soft_labels.sum(axis = 1, keepdims = 1)
        
        
        
        return soft_labels
    else:
        return cluster_labels

def cluster_and_ratio_loss(clustering_algo, n_clusters):
    def cluster_and_ratio_loss_(y_true, embedding_vectors):
        normalized_embeddings = tf.divide(tf.subtract(embedding_vectors, 
                                                      tf.reduce_min(embedding_vectors)), 
                                          tf.subtract(tf.reduce_max(embedding_vectors), 
                                                      tf.reduce_min(embedding_vectors)))
        if clustering_algo == "sib":
            algorithm = SIB(n_clusters=n_clusters,
                            n_init=10,
                            n_jobs=-1,
                            max_iter=15,
                            random_state=1024,
                            tol=0.02)
        elif clustering_algo == "kmeans":
            algorithm = KMeans(n_clusters=n_clusters,
                               init='k-means++',
            n_init='warn',
            max_iter=300,
            tol=0.0001,
            verbose=0,
            random_state=None,
            copy_x=True,
            algorithm='lloyd')
        elif clustering_algo == "affinity":
            algorithm = AffinityPropagation(damping=0.5,
                                            max_iter=200,
                                            convergence_iter=15,
                                            copy=True,
                                            preference=None,
                                            affinity='euclidean',
                                            verbose=False,
                                            random_state=None)
        elif clustering_algo == "meanshift":
            algorithm = MeanShift(bandwidth=None,
                                  seeds=None,
                                  bin_seeding=False,
                                  min_bin_freq=1,
                                  cluster_all=True,
                                  n_jobs=None,
                                  max_iter=300)
        elif clustering_algo == "DBSCAN":
            algorithm = DBSCAN(eps=0.5,
                               min_samples=5,
                               metric='euclidean',
                               metric_params=None,
                               algorithm='auto',
                               leaf_size=30,
                               p=None,
                               n_jobs=None)
        elif clustering_algo == "OPTICS":
            algorithm = OPTICS(min_samples=5 ,
                               metric='minkowski',
                               p=2,
                               metric_params=None,
                               cluster_method='xi',
                               eps=None,
                               xi=0.05,
                               predecessor_correction=True,
                               min_cluster_size=None,
                               algorithm='auto',
                               leaf_size=30,
                               memory=None,
                               n_jobs=-1)
        else:
            raise NotImplementedError

        norm_embedding = tf.identity(embedding_vectors).numpy()
        norm_embedding = (norm_embedding - norm_embedding.min())/(norm_embedding.max() - norm_embedding.min())
        clustering_model = algorithm.fit(norm_embedding)
        cluster_labels = clustering_model.labels_.tolist()
        
        for emb in embedding_vectors:
            tf.norm
        embedding_vectors
        logging.info(f'Finished clustering embeddings for {len(embedding_vectors)} texts into {len(cluster_labels)} clusters')
        
        

        # calculate centroids
        centroids_ = {}
        for e_i, embedding_vector in enumerate(embedding_vectors):
            embedding_vector = tf.reshape(embedding_vector, [1, -1])
            print(embedding_vector.shape)
            if cluster_labels[e_i] not in centroids_:
                centroids_[cluster_labels[e_i]] = [embedding_vector]
            else:
                centroids_[cluster_labels[e_i]] +=  [embedding_vector]
        
        centroids = []
        for label in centroids_.keys():
            temp = tf.concat(centroids_[label], 0)
            print(temp.shape)
            centroids += [tf.reduce_mean(temp, 0)]
            # print(tf.reduce_mean(centroids_[label], 0).shape)

        # calculate intra-distance for each cluster 
        # for computational efficiency, 
        # we use average distance between each data and centroid of that cluster
        # intra_distance = clustering_model.inertia_'
        intra_distance = 0
        for e_i, embedding_vector in enumerate(embedding_vectors):
            intra_distance += tf.norm(embedding_vector-centroids[cluster_labels[e_i]], ord='euclidean')

        # calculate inter-distance between each clusters # we use centroid as the representation of each cluster
        inter_distance = 0
        for c_i in range(len(centroids)):
            for c_j in range(c_i + 1, len(centroids)):
                inter_distance += tf.norm(centroids[c_i]-centroids[c_j], ord='euclidean') 
        
        # final ratio loss
        ratio_loss = intra_distance/inter_distance
        
        return ratio_loss
    
    return cluster_and_ratio_loss_
    
def EntropyLoss(y_true, y_pred):
    temp_factor = 1.0
    softmax = tf.nn.softmax(y_pred / temp_factor, axis=1)
    entropy = -softmax * tf.math.log(softmax + 1e-6)
    b = tf.math.reduce_mean(entropy)
    return b

def newLoss(y_true, y_pred):
    return EntropyLoss(y_true, y_pred) + tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    
def record_list_of_results(res_file_name, agg_file_name, list_of_dicts):
    if len(list_of_dicts) == 0:
        return
    lock_path = os.path.abspath(os.path.join(res_file_name, os.pardir, 'result_csv_files.lock'))
    with FileLock(lock_path):
        logging.debug("Inside lock")
        if os.path.isfile(res_file_name):
            orig_df = pd.read_csv(res_file_name)
            df = pd.concat([orig_df, pd.DataFrame(list_of_dicts)])
                    
            df_agg = df.groupby(by=['setting_name', 'eval_file', 'labeling_budget', 'pipeline', 'lr', 'bs', 'num_clusters']).mean()\
                .sort_values(by=['eval_file', 'labeling_budget', 'setting_name', 'pipeline', 'lr', 'bs', 'num_clusters'])
            df_agg.to_csv(agg_file_name)
        else:
            df = pd.DataFrame(list_of_dicts)
        df.to_csv(res_file_name, index=False)
        logging.debug("Releasing lock")
        
# trained from original pretrained model as the default
def train(texts, labels, is_cluster_labels, keep_classification_layer=True, model_name_or_path="bert-base-uncased",
          batch_size=64, learning_rate=3e-5, num_epochs=10):
    
    model_id = str(uuid.uuid1())
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenized_inputs = tokenizer(texts, add_special_tokens=True, max_length=128, padding=True, truncation=True)
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(tokenized_inputs), labels))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-06)
    if args.soft_label and is_cluster_labels:
        num_labels = labels.shape[1]
    else:
        num_labels = len(set(labels))
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_labels)
    model_path = os.path.join(OUT_DIR, model_id)
    os.makedirs(model_path)
    logging.info(f'training model {model_id} with {num_labels} output classes, '
                 f'starting from base model {model_name_or_path}')
    if args.soft_label and is_cluster_labels:
        model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))
    else:
        model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.fit(x=train_dataset.shuffle(1000).batch(batch_size), validation_data=None, epochs=num_epochs)
    if not keep_classification_layer:
        model.classifier._name = "dummy"  # change classifier layer name so it will not be reused
    model.save_pretrained(model_path)
    model.config.__class__.from_pretrained(model_name_or_path).save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    return model_path

# trained from original pretrained model as the default
def train_embedding(texts, clustering_algo, n_clusters, keep_classification_layer=True, model_name_or_path="bert-base-uncased", 
          batch_size=64, learning_rate=3e-5, num_epochs=10):
    
    model_id = str(uuid.uuid1())
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenized_inputs = tokenizer(texts, add_special_tokens=True, max_length=128, padding=True, truncation=True)
    train_dataset = tf.data.Dataset.from_tensor_slices(dict(tokenized_inputs))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-06)
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels = 50)
    model_path = os.path.join(OUT_DIR, model_id)
    os.makedirs(model_path)
    logging.info(f'training model {model_id} with {None} dimension of embedding, '
                 f'starting from base model {model_name_or_path}')
    model.compile(optimizer=optimizer, loss=cluster_and_ratio_loss(clustering_algo, n_clusters))    

    model.fit(x=train_dataset.shuffle(1000).batch(batch_size), validation_data=None, epochs=num_epochs)
        
    if not keep_classification_layer:
        model.classifier._name = "dummy"  # change classifier layer name so it will not be reused
    model.save_pretrained(model_path)
    model.config.__class__.from_pretrained(model_name_or_path).save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    return model_path

# trained from original pretrained model as the default
def train_entropy(texts, num_labels=50, keep_classification_layer=True, model_name_or_path="bert-base-uncased", 
          batch_size=64, learning_rate=3e-5, num_epochs=10):
    
    model_id = str(uuid.uuid1())
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenized_inputs = tokenizer(texts, add_special_tokens=True, max_length=128, padding=True, truncation=True)
    train_dataset = tf.data.Dataset.from_tensor_slices(dict(tokenized_inputs))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-06)
    # num_labels = len(set(labels))
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_labels)
    model_path = os.path.join(OUT_DIR, model_id)
    os.makedirs(model_path)
    logging.info(f'training model {model_id} with {num_labels} output classes, '
                 f'starting from base model {model_name_or_path}')
    model.compile(optimizer=optimizer, loss=EntropyLoss)
    model.fit(x=train_dataset.shuffle(1000).batch(batch_size), validation_data=None, epochs=num_epochs)
    if not keep_classification_layer:
        model.classifier._name = "dummy"  # change classifier layer name so it will not be reused
    model.save_pretrained(model_path)
    model.config.__class__.from_pretrained(model_name_or_path).save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    return model_path

# trained from original pretrained model as the default
def train_combine(texts, labels, keep_classification_layer=True, model_name_or_path="bert-base-uncased", 
          batch_size=64, learning_rate=3e-5, num_epochs=10):
    

    model_id = str(uuid.uuid1())
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenized_inputs = tokenizer(texts, add_special_tokens=True, max_length=128, padding=True, truncation=True)
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(tokenized_inputs), labels))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-06)
    num_labels = len(set(labels))
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_labels)
    model_path = os.path.join(OUT_DIR, model_id)
    os.makedirs(model_path)
    logging.info(f'training model {model_id} with {num_labels} output classes, '
                 f'starting from base model {model_name_or_path}')
    model.compile(optimizer=optimizer, loss=newLoss)
    model.fit(x=train_dataset.shuffle(1000).batch(batch_size), validation_data=None, epochs=num_epochs)
    if not keep_classification_layer:
        model.classifier._name = "dummy"  # change classifier layer name so it will not be reused
    model.save_pretrained(model_path)
    model.config.__class__.from_pretrained(model_name_or_path).save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    return model_path


    
def infer(texts, model_name_or_path, num_labels, infer_batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    logging.info(f'running inference using model {model_name_or_path} on {len(texts)} texts')
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_labels)
    predictions = []
    for x in tqdm.tqdm(range(0, len(texts), infer_batch_size)):
        batch_texts = texts[x:x + infer_batch_size]
        batch_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="tf")
        batch_res = model(batch_input).logits.numpy().tolist()
        predictions.extend(batch_res)

    return np.argmax(predictions, axis=1)

def evaluate(eval_texts, eval_labels, model_path, label_encoder):
    eval_predictions = infer(eval_texts, model_path, num_labels=len(label_encoder.classes_))
    eval_predictions = label_encoder.inverse_transform(eval_predictions)
    accuracy = np.mean([gold_label == prediction for gold_label, prediction in zip(eval_labels, eval_predictions)])
    return accuracy


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    parser = ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--eval_file', type=str, required=True)
    parser.add_argument('--labeling_budget', type=int, required=True)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--inter_training_epochs', type=int, default=1)
    parser.add_argument('--finetuning_epochs', type=int, default=10)
    parser.add_argument('--num_clusters', type=int, default=50)
    
    parser.add_argument('--clustering_algo', type=str, required=True, help="choose between [sib, kmeans, affinity, meanshift, DBSCAN]")
    parser.add_argument('--run_baseline', action='store_true', default=False, help="evaluate baseline (slower)")
    parser.add_argument('--pipeline', type=str, required=True, help="choose between [original, embedding, entropy]")
    parser.add_argument('--cuda', type=str, default="0,1,2,3,4,5,6,7,8", help="set gpu value")
    parser.add_argument('--lr', type=float, default=3e-5, help="learning rate for intermidiate task")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size of intermidiate task")
    parser.add_argument('--soft_label', action='store_true', default=False, help="activate soft labels in intermidate process")

    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda
    global OUT_DIR
    OUT_DIR = f"./{args.pipeline}_output_seed{args.random_seed}"
    logging.info(args)
    
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True) 


    # set random seed
    random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)
    np.random.seed(args.random_seed)

    
    unlabeled_texts = pd.read_csv(args.train_file)['text'].tolist()
    logging.info(f'Clustering {len(unlabeled_texts)} unlabeled texts into {args.num_clusters} clusters')
    # get_cluster_labels(unlabeled_texts, args.num_clusters, args.clustering_algo)
    if args.pipeline == "original":
        # generate pseudo-labels for unlabeled texts
        clustering_pseudo_labels = get_cluster_labels(unlabeled_texts, 
                                                      args.num_clusters, 
                                                      args.clustering_algo)

        # inter-train model on clustering pseudo-labels
        inter_trained_model_path = train(unlabeled_texts, 
                                         clustering_pseudo_labels, 
                                         is_cluster_labels = True, 
                                         keep_classification_layer=False, 
                                         num_epochs=args.inter_training_epochs)
        
    elif args.pipeline == "embedding":
        # extract features, cluster the embeddings, train the ratio distance loss (intra_distance/inter_distance)
        inter_trained_model_path = train_embedding(unlabeled_texts, 
                                                   args.clustering_algo, 
                                                   args.num_clusters, 
                                                   keep_classification_layer=False, 
                                                   num_epochs=args.inter_training_epochs)
    
    elif args.pipeline == "entropy":
        # entropy loss update using unlabeled datastream
        inter_trained_model_path = train_entropy(unlabeled_texts, 
                                                 num_labels=args.num_clusters, 
                                                 keep_classification_layer=False, 
                                                 num_epochs=args.inter_training_epochs, 
                                                 learning_rate = args.lr, 
                                                 batch_size = args.batch_size)
    
    elif args.pipeline == "combine":
        # generate pseudo-labels for unlabeled texts
        clustering_pseudo_labels = get_cluster_labels(unlabeled_texts, 
                                                      args.num_clusters, 
                                                      args.clustering_algo)

        # inter-train model on clustering pseudo-labels
        inter_trained_model_path = train(unlabeled_texts, 
                                         clustering_pseudo_labels, 
                                         is_cluster_labels = True, 
                                         keep_classification_layer=False, 
                                         num_epochs=args.inter_training_epochs)
    
    elif args.pipeline == "SECF":
        # generate pseudo-labels for unlabeled texts
        clustering_pseudo_labels = get_cluster_labels_SECF(unlabeled_texts, 
                                                           args.num_clusters, 
                                                           args.clustering_algo)

        # inter-train model on clustering pseudo-labels
        inter_trained_model_path = train(unlabeled_texts, 
                                         clustering_pseudo_labels, 
                                         is_cluster_labels = True, 
                                         keep_classification_layer=False, 
                                         num_epochs=args.inter_training_epochs)
        
    else:
        raise NotImplementedError
    
    # sample *labeling_budget* examples with their gold labels from the train file for fine-tuning
    labeled_data_sample = pd.read_csv(args.train_file).sample(n=args.labeling_budget, 
                                                              random_state=args.random_seed)
    label_encoder = LabelEncoder()
    sample_labels = label_encoder.fit_transform(labeled_data_sample['label'].tolist())
    sample_texts = labeled_data_sample['text'].tolist()

    # fine-tune over the pretrained model using the given sample (BASELINE)
    if args.run_baseline:
        model_finetuned_over_base_path = train(sample_texts, 
                                               sample_labels, 
                                               is_cluster_labels = False, 
                                               keep_classification_layer=True, 
                                               num_epochs=args.finetuning_epochs)

    # fine-tune over the inter-trained model using the given sample
    model_finetuned_over_intermediate_path = train(sample_texts, 
                                                   sample_labels, 
                                                   is_cluster_labels = False, 
                                                   keep_classification_layer=True, 
                                                   num_epochs=args.finetuning_epochs, 
                                                   model_name_or_path=inter_trained_model_path)

    # evaluate classification accuracy over the *eval_file*
    eval_df = pd.read_csv(args.eval_file)
    eval_texts = eval_df['text'].tolist()
    eval_labels = eval_df['label'].tolist()
    
    # take 10-15 mins
    if args.run_baseline:
        model_finetuned_over_base_accuracy = evaluate(eval_texts, 
                                                      eval_labels, 
                                                      model_finetuned_over_base_path, 
                                                      label_encoder)
    else:
        model_finetuned_over_base_accuracy = None

    model_finetuned_over_intermediate_accuracy = evaluate(eval_texts, 
                                                          eval_labels, 
                                                          model_finetuned_over_intermediate_path, 
                                                          label_encoder)

    if args.run_baseline:
        results = [{'eval_file': args.eval_file, 
                    'setting_name': 'base', 
                    'labeling_budget': args.labeling_budget, 
                    'pipeline' : "None", 
                    'lr' : args.lr, 
                    'bs' : args.batch_size, 
                    'num_clusters' : args.num_clusters,
                    'accuracy': model_finetuned_over_base_accuracy},
                   {'eval_file': args.eval_file, 
                    'setting_name': 'intermediate', 
                    'labeling_budget': args.labeling_budget, 
                    'pipeline' : args.pipeline, 
                    'lr' : args.lr, 
                    'bs' : args.batch_size, 
                    'num_clusters' : args.num_clusters,
                    'accuracy': model_finetuned_over_intermediate_accuracy}]
    else:
        results = [{'eval_file': args.eval_file, 
                    'setting_name': 'intermediate', 
                    'labeling_budget': args.labeling_budget, 
                    'pipeline' : args.pipeline, 
                    'lr' : args.lr, 
                    'bs' : args.batch_size, 
                    'num_clusters' : args.num_clusters,
                    'accuracy': model_finetuned_over_intermediate_accuracy}]
        
    # save evaluation results to csv files
    record_list_of_results(res_file_name=f'{OUT_DIR}/results.csv', 
                           agg_file_name=f'{OUT_DIR}/aggregated_results.csv',
                           list_of_dicts=results)
    if args.run_baseline:
        logging.info(f'Fine-tuned over base:\neval_file: {args.eval_file}, model: {model_finetuned_over_base_path}, '
                    f'accuracy: {model_finetuned_over_base_accuracy}')
    logging.info(f'Fine-tuned over intermediate:\neval_file: {args.eval_file}, '
                 f'model: {model_finetuned_over_intermediate_path}, '
                 f'accuracy: {model_finetuned_over_intermediate_accuracy}')
