import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import random
import csv
import collections

random.seed(12345)
np.random.seed(12345)

def write_results(info,dataname):
    with open('results/'+dataname+'_.csv','w') as csvfile:
        writer = csv.writer(csvfile,delimiter=',')
        for it in info:
            for i in it.items():

                writer.writerow(i)
            writer.writerow((' '))

def get_folds(examples, numfolds, labels=None, numclasses=10):
    if labels is None:
        # Can't straitify since we don't know class labels
        folded_examples = [[] for _ in range(numfolds)]
        next_fold = 0
        for example in examples:
            folded_examples[next_fold].append(example)
            next_fold = (next_fold + 1) % numfolds
        return folded_examples
    else:
        assert (len(examples) == len(labels))
        folded_examples = [[] for _ in range(numfolds)]
        folded_labels = [[] for _ in range(numfolds)]

        if numfolds <= numclasses:
            offset = 1
        else:
            offset = numfolds // numclasses

        next_fold = []
        for y in range(numclasses):
            next_fold.append((y * offset) % numfolds)

        for i in range(len(examples)):
            y = labels[i]
            folded_examples[next_fold[y]].append(examples[i])
            folded_labels[next_fold[y]].append(y)
            next_fold[y] = (next_fold[y] + 1) % numfolds

        return folded_examples, folded_labels


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# preprocessing
train_images = train_images/255.0
test_images=test_images/255.0


dropout_vals = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
accs = collections.OrderedDict()
losses = collections.OrderedDict()

for val in dropout_vals:
    accs[val] = 0
    losses[val]=0

print(len(train_images))
print(len(train_labels))
print(len(test_images))
print(len(test_labels))
train_images, train_labels = get_folds(train_images,len(dropout_vals)*3,train_labels)
test_images, test_labels = get_folds(test_images,len(dropout_vals)*3,test_labels)
print(len(train_images))
print(len(train_labels))
print(len(test_images))
print(len(test_labels))


for i in range(0,3):


    for idx,val in enumerate(dropout_vals):

        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28,28)), # 784 inputs
            #keras.layers.Dropout(val),
            keras.layers.Dense(50,activation=tf.nn.relu),
            keras.layers.Dropout(val),
            keras.layers.Dense(100, activation=tf.nn.relu),
            keras.layers.Dropout(val),

            keras.layers.Dense(100, activation=tf.nn.relu),
            keras.layers.Dropout(val),
            keras.layers.Dense(100, activation=tf.nn.relu),
            keras.layers.Dropout(val),
            keras.layers.Dense(10,activation=tf.nn.softmax) # 10 classes
        ])
        model.compile(optimizer=tf.train.AdamOptimizer(),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        curr_train_set = np.array(train_images[idx+i*len(dropout_vals)])
        curr_train_labels = np.array(train_labels[idx+i*len(dropout_vals)])
        curr_test_set = np.array(test_images[idx+i*len(dropout_vals)])
        curr_test_labels = np.array(test_labels[idx+i*len(dropout_vals)])


        model.fit(curr_train_set,curr_train_labels, epochs=30)
        test_loss,test_acc = model.evaluate(curr_test_set,curr_test_labels)
        print(test_loss)
        print(test_acc)
        accs[val] += test_acc
        losses[val] += test_loss



for val in dropout_vals:
    accs[val]=accs[val]/3
    losses[val]=losses[val]/3

write_results([accs,losses],"dropout_mk3")
print(accs)
print(losses)