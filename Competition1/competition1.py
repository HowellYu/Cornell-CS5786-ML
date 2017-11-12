
# coding: utf-8

# In[1]:

from itertools import permutations
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sklearn import manifold
from matplotlib import offsetbox
import multiprocessing
from functools import partial
from contextlib import contextmanager
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import kneighbors_graph
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import cluster
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  
from sklearn.neighbors import KNeighborsClassifier


# In[8]:

# def Euc_dist(pt1, pt2):
#     return np.sum(np.square(np.subtract(pt1, pt2)))
# # transform the graph into a matrix 
# connected_pts = pd.DataFrame.from_csv('../data/Graph.csv', header=None, index_col=None)
# graph_col_0 = connected_pts[0].tolist() # pt1
# graph_col_1 = connected_pts[1].tolist() # pt2

# A2 = np.ndarray(shape=(6000,6000), dtype=float)
# for i in range(len(graph_col_0)):
#     pt1_index = graph_col_0[i] - 1
#     pt2_index = graph_col_1[i] - 1
#     dist = Euc_dist(X[pt1_index, :], X[pt2_index, :])
#     A2[pt1_index, pt2_index] = np.exp(-dist/6000)
#     A2[pt2_index, pt1_index] = np.exp(-dist/6000)


# In[2]:

def spectral_embedding(A, method, K):
    '''
    Spectral embedding with Adjacency Matrix A
    '''
    
    print "Computing Spectral embedding"
    start = int(round(time.time()))

    X_spec = manifold.SpectralEmbedding(n_components= K, affinity= method, 
                                         random_state=None, eigen_solver=None,n_jobs = 4
                                        ).fit_transform(A)
    end = int(round(time.time()))
    print "--Spectral Embedding finished in ", (end-start), "s--------------" 
    print "Done."
    return X_spec


# In[3]:

def cca(X,Y,K):
    '''
    Perform CCA on two views X, Y and reduce dimension to K
    
    return pro
    '''
    cca = CCA(n_components = K,scale=False,max_iter = 1000)
    X_c, Y_c = cca.fit_transform(X, Y)
    return X_c,Y_c, cca


# In[4]:

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def evaluate(permu, cluster, label):
    pred = [permu[c] for c in cluster]
    return np.sum(pred == label)/60.

def labeling(assignment, cluster):
    pred = [assignment[c] for c in cluster]
    return np.array(pred,dtype=np.intp)
def get_center(X,seed,method):
    n,p = X.shape
    a = pd.DataFrame(X)
    b = pd.DataFrame({'label':seed[:,1]})
    c = pd.merge(a,b,'right',left_index=True, right_index=True)
    if method == 'mean':
        center = np.array(c.groupby(by = ['label'],axis = 0)[range(p)].mean())
    if method == 'median':
        center = np.array(c.groupby(by = ['label'],axis = 0)[range(p)].median())
    return center


def visualize_cluster(data,prediction,title):
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=500)
    tsne_data = tsne.fit_transform(data)
    plt.scatter(tsne_data[:,0], tsne_data[:,1], c= prediction, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.title(title)
    plt.show()
    
def ensemble(KNN_pred, SVM_pred, NN_pred):
    ensemble_result = []
    for knn, svm, nn in zip(KNN_pred, SVM_pred, NN_pred):
        if knn == svm:
            ensemble_result.append(knn)
        elif knn == nn:
            ensemble_result.append(knn)
        elif svm == nn:
            ensemble_result.append(svm)
        else:
            ensemble_result.append(knn)
    return ensemble_result
def CV_accuracy(pred, label):
    count = 0.0
    for i in range(len(pred)):
        if pred[i] == label[i]:
            count += 1.0
    return count * 1.0 / len(pred)


# In[ ]:

if __name__ == "__main__":  
    
    print "Start Preprocessing..."
    X = np.genfromtxt('../data/Extracted_features.csv',delimiter=',')
    X_6000 = X[:6000,]
    seed = np.genfromtxt('../data/seed.csv',delimiter=',',dtype=np.intp)
    X_4000 = X[6000:,]
    seed_idx = np.array(seed[:,0] - 1, dtype=np.intp)
     
    connected_pts = pd.DataFrame.from_csv('../data/Graph.csv', header=None, index_col=None)
    graph_col_0 = connected_pts[0].tolist()
    graph_col_1 = connected_pts[1].tolist()

    A1 = np.ndarray(shape=(6000,6000), dtype=float)
    for i in range(len(graph_col_0)):
        A1[graph_col_1[i] - 1, graph_col_0[i] - 1] = 1
        A1[graph_col_0[i] - 1, graph_col_1[i] - 1] = 1
    
    print "Preprocess Done...\n"
    
    # Spectral Embedding
    X_sc = spectral_embedding(A1,'precomputed', 100)   
    start = int(round(time.time()))
    
    print 'CCA Start...'
    X_c,Y_c, cca_fit = cca(X_6000, X_sc, 16) 
    print 'CCA Done\n'
    
    # get center for K-means initialization
    mean1 = get_center(Y_c,seed,'mean')    
    kmeans1 = KMeans(n_clusters=10, random_state=0, init = mean1, max_iter = 2000).fit(Y_c)
    
    mean2 = get_center(X_c,seed,'mean')    
    kmeans2 = KMeans(n_clusters=10, random_state=0, init = mean1, max_iter = 2000).fit(X_c)

    
    l = list(permutations(range(0, 10)))
    
    print "Searching Best Assignment ..."    
    # CCA views from Graph
    with poolcontext(processes=3) as pool:
        results1 = pool.map(partial(evaluate, cluster = kmeans1.labels_[seed_idx], label = seed[:,1]), l)
    m1 = max(results1)
    print m1, l[results1.index(m1)]
    assignment1 = l[results1.index(m1)]
    pred1 = labeling(assignment1,kmeans1.labels_)
    end = int(round(time.time()))
    
    # CCA views from features
    with poolcontext(processes=3) as pool:
        results2 = pool.map(partial(evaluate, cluster = kmeans2.labels_[seed_idx], label = seed[:,1]), l)
    m2 = max(results2)
    assignment2 = l[results2.index(m2)]
    pred2 = labeling(assignment2,kmeans2.labels_)     
    print "Searching finished in ", (end-start), "s"
    print "Done...\n"
    
    print "Start visualizing clusters"
    visualize_cluster(Y_c,pred1,"Clustering and Labeling Results (From Graph)")
    visualize_cluster(X_c,pred2 , "Clustering and Labeling Results (From Features)")

    print "Done...\n"
    
    print "Supervised Learning Based on 6000 labeled points..."
    X_train, X_val, y_train, y_val = train_test_split(
        X_6000, pred1, test_size=0.2, random_state=0)
    
    print "SVM Start..."
    cList = [10**float(x) for x in range(-3,2)]
    for c in cList:
        clf = svm.SVC(kernel='linear', C = c).fit(X_train, y_train)
        print c , clf.score(X_val, y_val)
    
    clf = svm.SVC(kernel = 'linear', C = 0.01).fit(X_6000,pred1)
    SVM_pred4000 = clf.predict(X_4000)
    print "SVM Done...\n"
    
    print "KNN Start"
    for k in [3,5,8, 10, 15, 20]:
        knn = KNeighborsClassifier(n_neighbors = k, weights = 'distance')
        knn.fit(X_train, y_train)
        print  k, knn.score(X_val, y_val)
    knn = KNeighborsClassifier(n_neighbors = 20, weights = 'distance').fit(X_6000,pred1)
    KNN_pred4000 = knn.predict(X_4000)
    print "KNN Done...\n"
    
    print "Neural Network Start ..."
    scaler = StandardScaler()  
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)  
    X_val = scaler.transform(X_val) 
    aList = [10**float(x) for x in range(-7,-3)]
    for a in aList:
        clf = MLPClassifier(solver='adam', activation = 'relu',max_iter = 1000,
                            alpha=a,hidden_layer_sizes=(1084,500,500,2000,30),random_state=1)
        clf.fit(X_train, y_train)
        print c, clf.score(X_val, y_val) 
    
    scaler = StandardScaler()  
    scaler.fit(X_6000)
    X_6000nn = scaler.transform(X_6000)  
    X_4000nn = scaler.transform(X_4000) 
    clf = MLPClassifier(solver='adam', activation = 'relu',
                            alpha=0.0001,hidden_layer_sizes=(1084,500,500,2000,30),random_state=1)
    clf.fit(X_6000nn,pred1)
    NN_pred4000 = clf.predict(X_4000nn)
    print "Neural Network Done ...\n"
    
    print "Start Ensemble ..."
    
    # svm 
    clf_svm = svm.SVC(kernel='linear', C = 0.01).fit(X_train, y_train)
    test_svm = clf_svm.predict(X_val)
    
    # knn
    knn = KNeighborsClassifier(n_neighbors = 20, weights = 'distance').fit(X_train, y_train)
    test_knn = knn.predict(X_val)
    
    # nn
    scaler = StandardScaler()  
    scaler.fit(X_train)
    X_train = scaler.transform(X_train) 
    X_val = scaler.transform(X_val) 
    clf_nn = MLPClassifier(solver='adam', activation = 'relu',
                        alpha=10e-6,hidden_layer_sizes=(1084,500,500,2000,30),random_state=1)
    clf_nn.fit(X_train, y_train)
    test_nn = clf_nn.predict(X_val)
    
    test_ensemble_result = ensemble(test_knn, test_svm, test_nn)
    print(CV_accuracy(test_ensemble_result, y_val), "on validation set.")
    ensemble_pred_4000 = ensemble(KNN_pred4000,SVM_pred4000,NN_pred4000)
    print "Ensemble Done ...\n"
    
    output = pd.DataFrame({'Id' :range(6001,10001),'Label':ensemble_pred_4000})
    output.to_csv('../result/submission_ensemble_1.csv.csv', index = False)
    print "Output csv Done."

