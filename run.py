#139738
#49289
import numpy as np
from numpy import random
import Sorec
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
import copy
import math
from tqdm import tqdm
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def get_trust_data(filename="epinion_dateset/trust_data.txt",theshape=(49290,49290)):
    f = open(filename)
    lines = f.readlines()
    row = []
    col = []
    data = []
    for line in tqdm(lines):
        alist = line.strip('\n').split()
        row.append(int(alist[0])-1)
        col.append(int(alist[1])-1)
        data.append(float(alist[2]))
    mtx = coo_matrix((data, (row, col)), shape=theshape)
    indeg = mtx.sum(axis=0)
    outdeg = mtx.sum(axis=1)
    factor = copy.deepcopy(mtx)
    for k in range(factor.data.shape[0]):
        i = factor.row[k]
        j = factor.col[k]
        factor.data[k] = math.sqrt(indeg[0, j]/(indeg[0,j]+outdeg[i, 0]))
    return csr_matrix(factor)

def get_ratings_data(filename="epinion_dateset/ratings_data.txt",theshape=(49290,139739)):
    f = open(filename)
    train_data = []
    train_row = []
    train_col = []
    vali_data = []
    vali_row = []
    vali_col = []
    lines = f.readlines()

    random.shuffle(lines)
    ind = -1
    pos = int(len(lines)*0.99)
    for line in tqdm(lines):
        ind += 1
        alist = line.strip('\n').split()
        if ind>=pos:
            vali_row.append(int(alist[0]))
            vali_col.append(int(alist[1]))
            vali_data.append(int(alist[2]))
            continue
        train_row.append(int(alist[0]))
        train_col.append(int(alist[1]))
        train_data.append(int(alist[2]))

    train_mtx = csr_matrix((train_data, (train_row,train_col)), shape=theshape, dtype='float64')
    vali_mtx = csr_matrix((vali_data, (vali_row, vali_col)), shape=theshape, dtype='float64')
    return train_mtx, vali_mtx

if __name__ == '__main__':
    print("loading data...")
    trust_data = get_trust_data()
    # train set and validae set
    ratings_data_train, ratings_data_validate = get_ratings_data()
    print("loading done...")
    print("begin to train...")
    socmodel = Sorec.MF(ratings_data_train, ratings_data_validate, trust_data, lr=0.01, momentum=0.5, latent_size=10,iters=300)
    U,V,Z,train_loss_list, validate_loss_list = socmodel.train()
    print("train done...")




