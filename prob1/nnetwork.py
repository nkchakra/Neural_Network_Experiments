from mnist import MNIST
import numpy as np
import random
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from scipy.misc import imresize
from skimage.color import rgb2gray
import scipy.io
from sklearn.preprocessing import StandardScaler
import csv
import collections
import sys
scaler = StandardScaler()

#

def get_data(foldername,num_examples, test=False,labeled=True):
    if foldername == "fashion_mnist" or foldername == 'mnist':

        print('reg mnist')
        fm = MNIST('../data/'+foldername)
        sample_examples = [];
        sample_labels = [];
        count = 0
        if test:
            images, labels = fm.load_testing()
        else:
            images, labels = fm.load_training()
        for i in range(10):
            idx = 0
            while (count < num_examples / 10):
                if labels[idx] == i:
                    sample_examples.append(images[idx]);
                    sample_labels.append(labels[idx])
                    count += 1
                idx += 1
            count = 0
        if labeled:
            print(np.array(sample_examples).shape)
            return np.array(sample_examples), np.array(sample_labels)

        else:
            return np.array(sample_examples)

    else:
        print('house_number')
        if test:
            data = scipy.io.loadmat("../data/house_number/train_32x32.mat")
        else:
            data = scipy.io.loadmat("../data/house_number/test_32x32.mat")
        images = []
        for i in range(data['X'].shape[3]):
            images.append(imresize(data['X'][:, :, :, i], (28, 28)))
        labels = data['y']
        for i in range(len(labels)):
            if labels[i] == 10:
                labels[i] = 0
        for i in range(len(images)):
            images[i] = (rgb2gray(images[i]) * 256).ravel()
        rand = np.random.randint(0, len(images), num_examples)
        sample_examples = [];
        sample_labels = []
        for i in rand:
            sample_examples.append(images[i]);
            sample_labels.append(labels[i][0])
        if labeled:
            print(np.array(sample_examples).shape)
            return np.array(sample_examples), sample_labels
        else:
            return np.array(sample_examples)


def create_network(train_set,label_set,prev_network=None, layer_tuple=(8), warm=True):
    if prev_network == None:
        network = MLPClassifier(solver='sgd',
                                hidden_layer_sizes=layer_tuple,
                                max_iter=30000,
                                learning_rate_init=0.0001,
                                n_iter_no_change=1,
                                random_state=12345,
                                warm_start=warm,
                                alpha=0.1,
                                verbose=False,
                                tol=0.0001)
        network.fit(train_set,label_set)
        print(network.n_iter_)
        return network
    else:
        maxiter = 30000
        iter = 0
        prev_loss = 100
        losses = []
        while (iter < maxiter):
            if iter >= 50 and abs((losses[0]) - (prev_network.loss_)) < 0.0001:
                break
            iter +=1
            prev_loss = prev_network.loss_
            losses.append(prev_loss)
            if iter > 50:
                losses.pop(0)

            prev_network.fit(train_set,label_set)
            #print('loss diff: '+str(prev_loss - prev_network.loss_))
        print('warm start used '+str(iter)+" iterations to converge")
        prev_network.n_iter_=iter

        return prev_network

        return prev_network








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


'''
shift digits in feature vector by shift
'''
def treat_data(dataset,shift=0):
    np.roll(dataset,shift,axis=1)



def main():
    print('start')
    dataname = 'mnist'
    if len(sys.argv)>1:
        dataname=sys.argv[1]
    experiment_count = 1
    info = collections.OrderedDict()
    info['base_acc_reg']=0
    info['warm_acc_reg']=0
    info['treat_acc_reg']=0
    info['base_acc_treat']=0
    info['warm_acc_treat']=0
    info['treat_acc_treat']=0
    info['base_iter']=0
    info['warm_iter']=0
    info['treat_iter']=0
    train_size = 20000
    test_size = 800
    shift_amt = 100
    for i in range(experiment_count):





        data=(get_data(dataname,train_size))
        folds = get_folds(data[0],2,labels=data[1])
        train_set = folds[0][0]
        train_labels = folds[1][0]

        treat_set = folds[0][1]
        treat_labels = folds[1][1]

        #print(train_set)

        treat_data(treat_set, shift_amt)
        #print(treated_set)

        scaler.fit(train_set)

        train_set = scaler.transform(train_set)
        treat_set = scaler.transform(treat_set)

        test = get_data(dataname,test_size,test=True);


        folds_test = get_folds(test[0],2,labels=test[1])
        test_set = folds_test[0][0]
        test_labels = folds_test[1][0]


        treat_test_set = folds_test[0][1]
        treat_data(treat_test_set,shift_amt)
        treat_test_labels = folds_test[1][1]


        test_set = scaler.transform(test_set)
        treat_test_set = scaler.transform(treat_test_set)
        print('base')

        network = create_network(train_set,train_labels)

        print(network.n_iter_)


        info['base_iter'] += network.n_iter_
        info['base_acc_reg'] +=  network.score(test_set,test_labels)
        info['base_acc_treat'] += network.score(treat_test_set,treat_test_labels)

        print('train acc: '+str(network.score(train_set,train_labels)))

        print('test acc: '+str(network.score(test_set,test_labels)))

        print('treat acc: '+str(network.score(treat_set,treat_labels)))

        print('treat_test acc: '+str(network.score(treat_test_set, treat_test_labels)))

        #input('')

        print('\nwarm start for treated')
        network = create_network(treat_set, treat_labels, prev_network=network)
        print(network.n_iter_)

        info['warm_iter'] += network.n_iter_
        info['warm_acc_reg'] +=  network.score(test_set,test_labels)
        info['warm_acc_treat'] += network.score(treat_test_set,treat_test_labels)
        print('train acc: '+str(network.score(train_set,train_labels)))

        print('test acc: '+str(network.score(test_set,test_labels)))

        print('treat acc: '+str(network.score(treat_set,treat_labels)))

        print('treat_test acc: '+str(network.score(treat_test_set, treat_test_labels)))

        #input('')

        print('\nnn from scratch for treated')
        network = create_network(treat_set, treat_labels, prev_network=None,warm=False)
        print(network.n_iter_)

        info['treat_iter'] += network.n_iter_
        info['treat_acc_reg'] +=  network.score(test_set,test_labels)
        info['treat_acc_treat'] += network.score(treat_test_set,treat_test_labels)

        print('train acc: ' + str(network.score(train_set, train_labels)))

        print('test acc: ' + str(network.score(test_set, test_labels)))

        print('treat acc: ' + str(network.score(treat_set, treat_labels)))

        print('treat_test acc: ' + str(network.score(treat_test_set, treat_test_labels)))

    info['base_iter'] = (info['base_iter'])/experiment_count
    info['base_acc_reg'] = (info['base_acc_reg'])/experiment_count
    info['base_acc_treat'] = (info['base_acc_treat'])/experiment_count

    info['warm_iter'] = (info['warm_iter'])/experiment_count
    info['warm_acc_reg'] = (info['warm_acc_reg'])/experiment_count
    info['warm_acc_treat'] = (info['warm_acc_treat'])/experiment_count

    info['treat_iter'] = (info['treat_iter'])/experiment_count
    info['treat_acc_reg'] = (info['treat_acc_reg'])/experiment_count
    info['treat_acc_treat'] = (info['treat_acc_treat'])/experiment_count


    write_results(info,(dataname+'_'+str(train_size)))
    

def write_results(info,dataname):
    with open('results/'+dataname+'.csv','w') as csvfile:
        writer = csv.writer(csvfile,delimiter=',')
        for i in info.items():

            writer.writerow(i)


if __name__ == "__main__":
    main()
