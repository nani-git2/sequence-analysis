#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Generating a 1D Gaussian mixture model (MM). Then using the observed variables, evaluating the 
mixture components (parameters) using expectation maximization (EM) algorithm.

"""
import numpy as np
import random as rn
import statistics as st
import progressbar as pb
import matplotlib.pyplot as plt


#creating a 2 component Gaussian mixture model
def mixture_model(N,p1,p2,mu1,sigma1,mu2,sigma2):
    observed_seq = np.empty(N)
    latent_seq = np.empty(N)
    
    for i in range(N):
        if rn.choices([1,2],weights=[p1,p2])[0] == 1:
            x=rn.gauss(mu1,sigma1)
            latent_seq[i] = 1
            observed_seq[i] = x

        else:
            x=rn.gauss(mu2,sigma2)
            latent_seq[i] = 2
            observed_seq[i] = x
        
    return observed_seq, latent_seq



# Expectation maximization algorithm to learn the parameters used to generate the observed dataset
def em_algorithm(X):
    '''
    Takes in the list of observed sequences and makes a prediction of the 
    parameters used, and of the latent variables
    '''
    # we choose the mu1 and mu2 initially by randomly picking any points between
    # the max and min of X
    
    mu_a, mu_b = rn.uniform(min(X), max(X)), rn.uniform(min(X), max(X))
    sigma_a, sigma_b = 1, 2          # simply assigned random std deviations
    
    #contains the probabilties of latent variables
    A=np.zeros(len(X))      
    B=np.zeros(len(X))
        
    #assigned equal priors to both distributions
    prior_a = 0.5           
    prior_b = 1-prior_a
    
    for n in pb.progressbar(range(1000)):        
        for i in range(len(X)):
            likelihood1 = (1/(sigma_a*(2*np.pi)**0.5))*(np.exp(-(X[i]-mu_a)**2/(2*sigma_a**2)))
            likelihood2 = (1/(sigma_b*(2*np.pi)**0.5))*(np.exp(-(X[i]-mu_b)**2/(2*sigma_b**2)))
            
            norm = likelihood1*prior_a + likelihood2*prior_b      #normalization factor for Bayes' rule
            A[i] = prior_a*likelihood1/norm
            B[i] = 1-A[i]
            
        #calculating new mean and standard deviation for A    
        mu_a = np.sum(A*X)/np.sum(A)                    
        sigma_a = np.sum((A*(X-mu_a)**2)/np.sum(A))**0.5
        #for b
        mu_b = np.sum(B*X)/np.sum(B)                    
        sigma_b = np.sum((B*(X-mu_b)**2)/np.sum(B))**0.5
        
        #recalculating the priors
        prior_a = np.sum(A)/len(A)               
        prior_b = 1-prior_a
    
    latent_var=[]
    for i in range(len(X)):
        if A[i] > B[i]:
            latent_var.append(1)
        elif A[i] <= B[i]:
            latent_var.append(2)
            
    return mu_a, sigma_a, mu_b, sigma_b, latent_var, prior_a, prior_b


#testing
if __name__ == "__main__":
    
    #parameters for generating MM
    N=200
    p1 = 0.3        #probabilites for choosing a distribution.
    p2 = 0.7 
    #parameters for first
    mu1 =  2.0            
    sigma1 = 3.0
    mu2 = 5.4
    sigma2 = 7.0
    
    X, Z= mixture_model(N,p1,p2,mu1,sigma1,mu2,sigma2)
    
    #checking the output
    mu_a, sigma_a, mu_b, sigma_b, latent_var, pa, pb = em_algorithm(X)
    #normalized Hamming distance
    dist=(sum(i!=j for i,j in zip(Z,latent_var)))/len(Z)       

    print("Actual : \n mu1={}, sigma1={}, mu2={}, sigma2={}".format(mu1, sigma1,mu2,sigma2))
    print("Predicted: \n mu1={}, sigma1={}, mu2={}, sigma2={}".format(mu_a, sigma_a,mu_b,sigma_b))
    print("\nActual p1 = {}, Predicted p1 = {}".format(p1, pa))
    print("\n\n Normalized Hamming distance : {}".format(dist))    

