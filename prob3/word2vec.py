import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import random
import csv
import collections
import os
from sklearn.manifold import TSNE
from sklearn import preprocessing
def vec_dist(vec1,vec2):
    return np.sqrt(np.sum((vec1-vec2)**2))

def nearest_word(word_idx, vector_list):
    min_dist = 50000
    min_idx = -1
    ask_vec = vector_list[word_idx]
    for idx,vec in enumerate(vector_list):
        if(vec_dist(vec,ask_vec) < min_dist and not np.array_equal(vec,ask_vec)):
            min_dist = vec_dist(vec,ask_vec)
            min_idx=idx
    return min_idx


def raw_string_to_dicts(raw_string):
    wordlist = []
    for word in raw_string.split():
        if word != '.':
            wordlist.append(word)
    word_set = set(wordlist)
    w2i = collections.OrderedDict()
    i2w = collections.OrderedDict()
    for i,w in enumerate(word_set):
        w2i[w] = i
        i2w[i] = w
    return w2i,i2w

def sentence_list(raw_string):
    sents = []
    for sentence in raw_string.split('.'):
        sents.append(sentence.split())
    return sents

def create_word_pairs(sentences, win_size=2):
    data = []
    for sent in sentences: # loop thru sentences
        for idx,w in enumerate(sent): # loop thru words in sentence
            for sub_word in sent[max(idx-win_size,0):min(idx+win_size,len(sent))+1]:
                if sub_word!=w:
                    data.append([w,sub_word])
    return data

def convert_to_sparse(index, size):
    temp = np.zeros(size)
    temp[index]=1
    return temp

neg_names = os.listdir('../data/txt_sentoken/neg/')
pos_names = os.listdir('../data/txt_sentoken/pos/')

file_amt = 3
neg_names = neg_names[0:file_amt]
pos_names = pos_names[0:file_amt]

neg_data= ''
pos_data= ''
for i in range(0,file_amt):
    with open('../data/txt_sentoken/pos/'+pos_names[i],"r+") as pos_f:
        pos_data+=pos_f.read()
    with open('../data/txt_sentoken/neg/'+neg_names[i], "r+") as neg_f:
        neg_data += neg_f.read()

# now pos/neg data are both long strings of data
# pos is from positive reviews, neg is from negative reviews

w2i_pos,i2w_pos = raw_string_to_dicts(pos_data)

w2i_neg,i2w_neg = raw_string_to_dicts(neg_data)

pos_sents = sentence_list(pos_data)
neg_sents = sentence_list(neg_data)
#print(pos_sents)
#print(neg_sents)

pos_data=(create_word_pairs(pos_sents))
neg_data=(create_word_pairs(neg_sents))

pos_x_train = []
pos_y_train = []

neg_x_train = []
neg_y_train = []

for dw in pos_data:
    pos_x_train.append(convert_to_sparse(w2i_pos[dw[0]], len(w2i_pos)))
    pos_y_train.append(convert_to_sparse(w2i_pos[dw[1]], len(w2i_pos)))
for dw in neg_data:
    neg_x_train.append(convert_to_sparse(w2i_neg[dw[0]], len(w2i_neg)))
    neg_y_train.append(convert_to_sparse(w2i_neg[dw[1]], len(w2i_neg)))

pos_x_train = np.asarray(pos_x_train)
pos_y_train = np.asarray(pos_x_train)
neg_x_train = np.asarray(neg_x_train)
neg_y_train = np.asarray(neg_x_train)

pos_x = tf.placeholder(tf.float32,shape=(None,len(w2i_pos)))
pos_y_label = tf.placeholder(tf.float32,shape=(None,len(w2i_pos)))

neg_x = tf.placeholder(tf.float32,shape=(None,len(w2i_neg)))
neg_y_label = tf.placeholder(tf.float32,shape=(None,len(w2i_neg)))

embed_dim = 5

posW1 = tf.Variable(tf.random_normal([len(w2i_pos),embed_dim]))
posb1 = tf.Variable(tf.random_normal([embed_dim]))

hidden_rep_pos = tf.add(tf.matmul(pos_x,posW1),posb1)

negW1 = tf.Variable(tf.random_normal([len(w2i_neg),embed_dim]))
negb1 = tf.Variable(tf.random_normal([embed_dim]))

hidden_rep_neg = tf.add(tf.matmul(pos_x,posW1),posb1)

posW2 = tf.Variable(tf.random_normal([embed_dim,len(w2i_pos)]))
posb2 = tf.Variable(tf.random_normal([len(w2i_pos)]))


negW2 = tf.Variable(tf.random_normal([embed_dim,len(w2i_neg)]))
negb2 = tf.Variable(tf.random_normal([len(w2i_neg)]))


pos_pred = tf.nn.softmax(tf.add(tf.matmul(hidden_rep_pos,posW2),posb2))

neg_pred = tf.nn.softmax(tf.add(tf.matmul(hidden_rep_neg,negW2),negb2))


session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)

# loss funcs
cross_entropy_loss_pos = tf.reduce_mean(-tf.reduce_sum(pos_y_label*tf.log(pos_pred),reduction_indices=[1]))
cross_entropy_loss_neg = tf.reduce_mean(-tf.reduce_sum(neg_y_label*tf.log(neg_pred),reduction_indices=[1]))

# train steps
pos_train_step =tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss_pos)
neg_train_step =tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss_neg)

max_iter = 2000

for i in range(max_iter):
    print(i)
    session.run(pos_train_step,feed_dict={pos_x: pos_x_train, pos_y_label: pos_y_train})

# for i in range(max_iter):
#     session.run(neg_train_step, feed_dict={neg_x: neg_x_train, neg_y_label: neg_y_train})




vectors = session.run(posW1+posb1)

model = TSNE(n_components=2)
vectors = model.fit_transform(vectors)

norm = preprocessing.Normalizer()
vectors = norm.fit_transform(vectors,'l2')

fig,ax = plt.subplots()
for word in w2i_pos.keys():
    ax.annotate(word,(vectors[w2i_pos[word]][0],vectors[w2i_pos[word]][1]))

plt.show()