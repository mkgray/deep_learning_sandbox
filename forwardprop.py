# -*- coding: utf-8 -*-
"""
Created on Sat Mar 04 11:14:52 2017

@author: Matthew
"""

import numpy as np
import matplotlib.pyplot as plt

# Softmax capable of handling N records as rows
def softmax(a):
    return np.divide(np.exp(a),np.sum(np.exp(a), axis=1, keepdims=True))
    
def forward(X, W1, b1, W2, b2):
    Z = 1 / (1 + np.exp(-X.dot(W1) - b1))
    A = Z.dot(W2) + b2
    Y = softmax(A)
    return Y

def classification_rate(Y, P):
    return np.mean(Y==P)

if __name__ == "__main__":
    
    Nclass = 500
    
    X1 = np.random.randn(Nclass, 2) + np.array([0, -2])
    X2 = np.random.randn(Nclass, 2) + np.array([2, 2])
    X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])
    X = np.vstack([X1, X2, X3])
    
    Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
    
    plt.scatter(X[:,0], X[:,1], c=Y, s=5, alpha=0.5)
    plt.show()
    
    D=2 # Number of dimensions
    M=3 # Hidden layer size
    K=3 # Number of classes
    
    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)
    
    P_Y_given_X = forward(X, W1, b1, W2, b2)
    P = np.argmax(P_Y_given_X, axis=1)
    
    assert(len(P) == len(Y))
    
    print("Classification rate for randomly chosen weights:", classification_rate(Y, P))