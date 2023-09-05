import os
import random
import numpy as np
import pickle as pkl
# import networkx as nx
import scipy.sparse as sp
from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from transformers import BertTokenizer

import sys
sys.path.append('../')
from utils.utils import loadWord2Vec, clean_str



if len(sys.argv) != 2:
	sys.exit("Use: python build_graph.py <dataset>")

datasets = ['SentNOB', 'mr']
dataset = sys.argv[1]

if dataset not in datasets:
	sys.exit("wrong dataset name")

word_embeddings_dim = 300
word_vector_map = {}


doc_name_list = []
doc_train_list = []
doc_test_list = []

with open('../data/' + dataset + '.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        doc_name_list.append(line.strip())
        temp = line.split("\t")
        if temp[1].find('test') != -1:
            doc_test_list.append(line.strip())
        elif temp[1].find('train') != -1:
            doc_train_list.append(line.strip())


doc_content_list = []
with open('../data/corpus/' + dataset + '.clean.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        doc_content_list.append(line.strip())


print(len(doc_name_list))
print(len(doc_content_list))

train_ids = []
for train_name in doc_train_list:
    train_id = doc_name_list.index(train_name)
    train_ids.append(train_id)
print(train_ids)
random.shuffle(train_ids)


train_ids_str = '\n'.join(str(index) for index in train_ids)
with open('../data/' + dataset + '.train.index', 'w') as f:
    f.write(train_ids_str)


test_ids = []
for test_name in doc_test_list:
    test_id = doc_name_list.index(test_name)
    test_ids.append(test_id)
print(test_ids)
random.shuffle(test_ids)

test_ids_str = '\n'.join(str(index) for index in test_ids)
with open('../data/' + dataset + '.test.index', 'w') as f:
    f.write(test_ids_str)


ids = (train_ids + test_ids)
print(ids)
print(len(ids))

shuffle_doc_name_list = []
shuffle_doc_words_list = []
for id in ids:
    shuffle_doc_name_list.append(doc_name_list[int(id)])
    shuffle_doc_words_list.append(doc_content_list[int(id)])
shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list)
shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)


with open('../data/' + dataset + '_shuffle.txt', 'w') as f:
    f.write(shuffle_doc_name_str)

with open('../data/corpus/' + dataset + '_shuffle.txt', 'w') as f:
    f.write(shuffle_doc_words_str)



word_freq = {}
word_set = set()
for doc_words in shuffle_doc_words_list:
    words = doc_words.split()
    for word in words:
        word_set.add(word)
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

vocab = list(word_set)
vocab_size = len(vocab)

word_doc_list = {}

for i in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    appeared = set()
    for word in words:
        if word in appeared:
            continue
        if word in word_doc_list:
            doc_list = word_doc_list[word]
            doc_list.append(i)
            word_doc_list[word] = doc_list
        else:
            word_doc_list[word] = [i]
        appeared.add(word)

word_doc_freq = {}
for word, doc_list in word_doc_list.items():
    word_doc_freq[word] = len(doc_list)

word_id_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i

vocab_str = '\n'.join(vocab)

with open('../data/corpus/' + dataset + '_vocab.txt', 'w') as f:
    f.write(vocab_str)

label_set = set()
for doc_meta in shuffle_doc_name_list:
    temp = doc_meta.split('\t')
    label_set.add(temp[2])
label_list = list(label_set)

label_list_str = '\n'.join(label_list)
with open('../data/corpus/' + dataset + '_labels.txt', 'w') as f:
    f.write(label_list_str)


train_size = len(train_ids)
val_size = int(0.1 * train_size)
real_train_size = train_size - val_size  # - int(0.5 * train_size)

real_train_doc_names = shuffle_doc_name_list[:real_train_size]
real_train_doc_names_str = '\n'.join(real_train_doc_names)

with open('../data/' + dataset + '.real_train.name', 'w') as f:
    f.write(real_train_doc_names_str)


row_x = []
col_x = []
data_x = []
for i in range(real_train_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            # print(doc_vec)
            # print(np.array(word_vector))
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_x.append(i)
        col_x.append(j)
        # np.random.uniform(-0.25, 0.25)
        # data_x.append(doc_vec[j] / doc_len)  # doc_vec[j]/ doc_len
        if doc_len != 0 and not np.isnan(doc_len):
            data_x.append(doc_vec[j] / doc_len)
        else:
            # Handle the case where division is not possible
            data_x.append(0.0)  # You can choose an appropriate default value


x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
    real_train_size, word_embeddings_dim))

# y = []
# for i in range(real_train_size):
#     doc_meta = shuffle_doc_name_list[i]
#     temp = doc_meta.split('\t')
#     label = temp[2]
#     one_hot = [0 for l in range(len(label_list))]
#     label_index = label_list.index(label)
#     one_hot[label_index] = 1
#     y.append(one_hot)
# y = np.array(y)
# print(y)


# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('csebuetnlp/banglabert')  # You can choose a different BERT model if needed

# Assuming y is your one-hot encoded labels
y = []

for i in range(len(y)):
    label_indices = [idx for idx, val in enumerate(y[i]) if val == 1]
    label_text = ' '.join([label_list[idx] for idx in label_indices])

    # Tokenize the label text using the BERT tokenizer
    tokenized_label = tokenizer.tokenize(label_text)

    # Convert tokens to IDs
    label_ids = tokenizer.convert_tokens_to_ids(tokenized_label)

    y.append(label_ids)

print(y)


test_size = len(test_ids)

row_tx = []
col_tx = []
data_tx = []
for i in range(test_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i + train_size]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_tx.append(i)
        col_tx.append(j)
        # np.random.uniform(-0.25, 0.25)
        # data_tx.append(doc_vec[j] / doc_len)  # doc_vec[j] / doc_len
        if doc_len != 0 and not np.isnan(doc_len):
            data_tx.append(doc_vec[j] / doc_len)
        else:
            # Handle the case where division is not possible
            data_tx.append(0.0)  # You can choose an appropriate default value

# tx = sp.csr_matrix((test_size, word_embeddings_dim), dtype=np.float32)
tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                   shape=(test_size, word_embeddings_dim))

# ty = []
# for i in range(test_size):
#     doc_meta = shuffle_doc_name_list[i + train_size]
#     temp = doc_meta.split('\t')
#     label = temp[2]
#     one_hot = [0 for l in range(len(label_list))]
#     label_index = label_list.index(label)
#     one_hot[label_index] = 1
#     ty.append(one_hot)
# ty = np.array(ty)
# print(ty)


# Assuming ty is your one-hot encoded test labels
ty = []

for i in range(len(ty)):
    label_indices = [idx for idx, val in enumerate(ty[i]) if val == 1]
    label_text = ' '.join([label_list[idx] for idx in label_indices])

    # Tokenize the label text using the BERT tokenizer
    tokenized_label = tokenizer.tokenize(label_text)

    # Convert tokens to IDs
    label_ids = tokenizer.convert_tokens_to_ids(tokenized_label)

    ty.append(label_ids)

print(ty)



word_vectors = np.random.uniform(-0.01, 0.01,
                                 (vocab_size, word_embeddings_dim))

for i in range(len(vocab)):
    word = vocab[i]
    if word in word_vector_map:
        vector = word_vector_map[word]
        word_vectors[i] = vector

row_allx = []
col_allx = []
data_allx = []

for i in range(train_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_allx.append(int(i))
        col_allx.append(j)
        # np.random.uniform(-0.25, 0.25)
        # data_allx.append(doc_vec[j] / doc_len)  # doc_vec[j]/doc_len
        if doc_len != 0 and not np.isnan(doc_len):
            data_allx.append(doc_vec[j] / doc_len)
        else:
            # Handle the case where division is not possible
            data_allx.append(0.0)  # You can choose an appropriate default value

for i in range(vocab_size):
    for j in range(word_embeddings_dim):
        row_allx.append(int(i + train_size))
        col_allx.append(j)
        data_allx.append(word_vectors.item((i, j)))


row_allx = np.array(row_allx)
col_allx = np.array(col_allx)
data_allx = np.array(data_allx)

allx = sp.csr_matrix(
    (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))

# ally = []
# for i in range(train_size):
#     doc_meta = shuffle_doc_name_list[i]
#     temp = doc_meta.split('\t')
#     label = temp[2]
#     one_hot = [0 for l in range(len(label_list))]
#     label_index = label_list.index(label)
#     one_hot[label_index] = 1
#     ally.append(one_hot)

# for i in range(vocab_size):
#     one_hot = [0 for l in range(len(label_list))]
#     ally.append(one_hot)

# ally = np.array(ally)


# Assuming ally is your combined one-hot encoded labels
ally = []

# Convert one-hot encoded labels for training data
for i in range(train_size):
    label_indices = [idx for idx, val in enumerate(ally[i]) if val == 1]
    label_text = ' '.join([label_list[idx] for idx in label_indices])

    # Tokenize the label text using the BERT tokenizer
    tokenized_label = tokenizer.tokenize(label_text)

    # Convert tokens to IDs
    label_ids = tokenizer.convert_tokens_to_ids(tokenized_label)

    ally.append(label_ids)

# Add one-hot encoded vectors for vocabulary
for i in range(vocab_size):
    one_hot_indices = [idx for idx, val in enumerate(ally[i + train_size]) if val == 1]
    one_hot_text = ' '.join([label_list[idx] for idx in one_hot_indices])

    # Tokenize the one-hot text using the BERT tokenizer
    tokenized_one_hot = tokenizer.tokenize(one_hot_text)

    # Convert tokens to IDs
    one_hot_ids = tokenizer.convert_tokens_to_ids(tokenized_one_hot)

    ally.append(one_hot_ids)

ally = np.array(ally)


print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

'''
Doc word heterogeneous graph
'''

# word co-occurence with context windows
window_size = 20
windows = []

for doc_words in shuffle_doc_words_list:
    words = doc_words.split()
    length = len(words)
    if length <= window_size:
        windows.append(words)
    else:
        # print(length, length - window_size + 1)
        for j in range(length - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)
            # print(window)


word_window_freq = {}
for window in windows:
    appeared = set()
    for i in range(len(window)):
        if window[i] in appeared:
            continue
        if window[i] in word_window_freq:
            word_window_freq[window[i]] += 1
        else:
            word_window_freq[window[i]] = 1
        appeared.add(window[i])

word_pair_count = {}
for window in windows:
    for i in range(1, len(window)):
        for j in range(0, i):
            word_i = window[i]
            word_i_id = word_id_map[word_i]
            word_j = window[j]
            word_j_id = word_id_map[word_j]
            if word_i_id == word_j_id:
                continue
            word_pair_str = str(word_i_id) + ',' + str(word_j_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            # two orders
            word_pair_str = str(word_j_id) + ',' + str(word_i_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1

row = []
col = []
weight = []

# pmi as weights

num_window = len(windows)

for key in word_pair_count:
    temp = key.split(',')
    i = int(temp[0])
    j = int(temp[1])
    count = word_pair_count[key]
    word_freq_i = word_window_freq[vocab[i]]
    word_freq_j = word_window_freq[vocab[j]]
    pmi = log((1.0 * count / num_window) /
              (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
    if pmi <= 0:
        continue
    row.append(train_size + i)
    col.append(train_size + j)
    weight.append(pmi)

# word vector cosine similarity as weights

# doc word frequency
doc_word_freq = {}

for doc_id in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[doc_id]
    words = doc_words.split()
    for word in words:
        word_id = word_id_map[word]
        doc_word_str = str(doc_id) + ',' + str(word_id)
        if doc_word_str in doc_word_freq:
            doc_word_freq[doc_word_str] += 1
        else:
            doc_word_freq[doc_word_str] = 1

for i in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_word_set = set()
    for word in words:
        if word in doc_word_set:
            continue
        j = word_id_map[word]
        key = str(i) + ',' + str(j)
        freq = doc_word_freq[key]
        if i < train_size:
            row.append(i)
        else:
            row.append(i + vocab_size)
        col.append(train_size + j)
        idf = log(1.0 * len(shuffle_doc_words_list) /
                  word_doc_freq[vocab[j]])
        weight.append(freq * idf)
        doc_word_set.add(word)

node_size = train_size + vocab_size + test_size
adj = sp.csr_matrix(
    (weight, (row, col)), shape=(node_size, node_size))

# dump objects
with open("../data/ind.{}.x".format(dataset), 'wb') as f:
    pkl.dump(x, f)

with open("../data/ind.{}.y".format(dataset), 'wb') as f:
    pkl.dump(y, f)

with open("../data/ind.{}.tx".format(dataset), 'wb') as f:
    pkl.dump(tx, f)

with open("../data/ind.{}.ty".format(dataset), 'wb') as f:
    pkl.dump(ty, f)

with open("../data/ind.{}.allx".format(dataset), 'wb') as f:
    pkl.dump(allx, f)

with open("../data/ind.{}.ally".format(dataset), 'wb') as f:
    pkl.dump(ally, f)

with open("../data/ind.{}.adj".format(dataset), 'wb') as f:
    pkl.dump(adj, f)







