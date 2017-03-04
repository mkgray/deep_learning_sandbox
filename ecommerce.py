# -*- coding: utf-8 -*-
"""
Created on Sat Mar 04 13:44:56 2017

@author: Matthew
"""

import numpy as np
import pandas as pd

from forwardprop import forward, classification_rate

def normalize(a):
    return (a-a.mean())/a.std()

def get_data():
    df = pd.read_csv('ecommerce_data.csv')
    data = df.as_matrix()
    
    X = data[:, :-1]
    Y = data[:, -1]
    
    # Normalize numerical columns
    X[:,1] = normalize(X[:,1])
    X[:,2] = normalize(X[:,2])
    
    N, D = X.shape
    X2 = np.zeros((N, D+3))
    X2[:,0:(D-1)] = X[:,0:(D-1)]
    
    for n in xrange(N):
        t = int(X[n,D-1])
        X2[n,t+D-1] = 1
        
    Z = np.zeros((N, 4))
    Z[np.arange(N), X[:,D-1].astype(np.int32)] = 1
    # X2[:,-4:] = Z
    assert(np.abs(X2[:,-4:] - Z).sum() < 10e-10)
    
    return X2, Y

def get_binary_data():
    X, Y = get_data()
    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]
    return X2, Y2

if __name__ == '__main__':
    
    X, Y = get_data()
    
    M=5 # number hidden units
    D = X.shape[1]
    K = len(set(Y))
    
    W1 = np.random.randn(D,M)
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K)
    b2 = np.zeros(K)
    
    P_Y_given_X = forward(X, W1, b1, W2, b2)
    predictions = np.argmax(P_Y_given_X, axis=1)
    
    print("Score:", classification_rate(Y, predictions))