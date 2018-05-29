# -*- coding: utf-8 -*-
""" Implementation of dynamic time warping. This module implments dynamic time 
warping (DTW) for calculating the similarity distance between two multivariable
time series. 
"""

import numpy as np
import pandas as pd

def dynamic_time_wraping(a, b, metric):
    """Returns the minimum cost between two time series, distance matrix, and 
    the cost matrix.
    Args:
        a (list or numpy array): list/array with length n_a, which represents 
        data points of the time series. 
        b (list or numpy array): list/array with length n_a, which represents 
        data points of the time series.        
        metrix (callable): local distance function to calculate the distance 
        between data points in two time series.
        
    Returns:
        float: the minimum cost between two time series.
        array: distance matrix
        array: cost matrix
    """
    # Check the inputs
    for x in (a,b):
        if not (isinstance(x, list) or isinstance(x, np.ndarray)):
            raise TypeError("input must be a list or numpy array")
    
    # Create cost matrix with shape [n_a, n_b]
    a, b = np.array(a), np.array(b)
    n_a, n_b = len(a), len(b)
    cost_matrix = np.zeros((n_a, n_b)) 
    
    # Initialize the first row and column
    cost_matrix[0,0] = metric(a[0],b[0])
    for i in range(1, n_a):
        cost_matrix[i, 0] = cost_matrix[i-1, 0] + metric(a[i],b[0])
    for j in range(1, n_b):
        cost_matrix[0, j] = cost_matrix[0, j-1] + metric(a[0], b[j])
    
    # Calculate the subsection of cost matrix
    for i in range(1, n_a):
        for j in range(1, n_b):
            cost_matrix[i, j] = min(cost_matrix[i-1, j-1], cost_matrix[i, j-1], 
                      cost_matrix[i-1, j]) + metric(a[i],b[j])
            
    # Return DTW distance
    return np.sqrt(cost_matrix[n_a-1, n_b-1]), cost_matrix[1:,1:]

def wraping_path(cost_matrix):
    """Return the optimal warping path. Code is based on the algorithm from 
    Chapter 4 in "Information Retrieval for Music and Motion".    
    """
    # The last point of the optimal wraping path
    path = [[x-1 for x in cost_matrix.shape]]
    n, m = path[-1]
    
    while not (n == 0 and m == 0):
        if n == 0:
            m -= 1
        elif m == 0:
            n -= 1
        else:
            choices = [cost_matrix[n-1, m-1], cost_matrix[n-1, m], cost_matrix[n, m-1]]
            index = np.argmin(choices)
            if index == 0:
                n, m = n-1, m-1
            elif index == 1:
                n -= 1
            else:
                m -= 1
        path.append([n,m])
    path.reverse()
    
    return np.array(path)

def LB_Keogh(s1,s2,r):
    LB_sum=0
    for ind,i in enumerate(s1):

        lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])

        if i>upper_bound:
            LB_sum+=(i-upper_bound)**2
        elif i<lower_bound:
            LB_sum+=(i-lower_bound)**2

    return np.sqrt(LB_sum)
   
def LB_MV(a, b, r):
    """Returns the lower bound Keogh for multivariate time series.
    Args:
        a (list or numpy array): list/array with length n_a, which represents 
        data points of the time series. 
        b (list or numpy array): list/array with length n_a, which represents 
        data points of the time series.        
        metrix (callable): local distance function to calculate the distance 
        between data points in two time series.
        
    Returns:
        float: the minimum cost between two time series.
        array: distance matrix
        array: cost matrix
    """
    
    LB_sum=0
    for ind,a_i in enumerate(a):
        # 
        lower_bound=np.min(b[(ind-r if ind-r>=0 else 0):(ind+r)], axis=0)
        upper_bound=np.max(b[(ind-r if ind-r>=0 else 0):(ind+r)], axis=0)

        LB_sum += np.dot(a_i > upper_bound,(a_i-upper_bound)**2)
        LB_sum += np.dot(a_i < lower_bound,(a_i-lower_bound)**2)

    return np.sqrt(LB_sum)   
 
def test():
    import matplotlib.pyplot as plt
    from scipy.spatial import distance
    from timeit import default_timer as timer
    
    metric = lambda x,y: np.sqrt(np.sum((x-y)**2))
    
    x=np.linspace(0,4*np.pi,48)
    ts1=np.random.rand(48,8)
    ts2=np.random.rand(48,8)
    
    plt.figure()
    plt.plot(x, ts1)
    plt.plot(x,ts2)
    
    r = int(round(len(ts1)/10.0))
    print(r)
    
    start = timer()
    print("DTW:", dynamic_time_wraping(ts1, ts2, metric)[0])
    end = timer()
    print('time:', end-start)
    
    
    start = timer()
    print("LB_MV:", LB_MV(ts1,ts2,r))
    end = timer()
    print(end-start)
    
    
if __name__ == "__main__":
    test()
