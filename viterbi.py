#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a sequence based on a hidden Markov model, and then use Viterbi
algorithm to predict the hidden sequence.

https://en.wikipedia.org/wiki/Viterbi_algorithm

"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def gen_hmm(S, T, a, e):
    """
    Inputs: S- set of possible observations.
            T- space of hidden states
            a- transition matrix between hidden states (also includes start/stop)
            e- matrix of emission probabilties

    Output: Returns the sequence of hidden states pi, and the sequence of
            observations x. pi has two elements more than x (for the start and
            stop states)
    """
    # empty lists for outputs
    pi = [0]  # hidden states starts with 0
    x = []

    def observation(state):
        """
        Returns an output based on the emission probabilities for given state
        """
        if state == 1:
            return random.choices(S, e[0, :])
        elif state == 2:
            return random.choices(S, e[1, :])

    # generates the hidden and observed sequences
    while True:
        s0 = pi[-1]
        s1 = random.choices(T, a[s0, :])
        pi.append(s1[0])
        if s1[0] == 0:
            break
        obs = observation(s1[0])
        x.append(obs[0])

    return pi, x



def viterbi(S, T, a, e, x):
    """
    Inputs: Set of possible observations S, set of hidden states T, transition
    matrix a, emission matrix e, and the observed sequence x.

    Output: The sequence of hidden states pi.
    """
    num_obs = len(x)
    num_states = len(T)

    def emission_prob(state, obs):
        """Probability of making an observatioin in a given state."""
        emission_df = pd.DataFrame(data=e, index=T[1:], columns=S)
        return emission_df[obs].loc[state]

    def transition_prob(i, j):
        """Prob of state i going to j."""
        trans_df = pd.DataFrame(data=a, index=T, columns=T)
        return trans_df[j].loc[i]
	
    # empty arrays for the dynamic programming table and for pointers.
    dyn_array = np.zeros((num_states - 1, num_obs))
    pointer_array = np.zeros((num_states - 1, num_obs))

    # filling values in the arrays

    for j in range(0, num_obs):
        for i in range(0, num_states - 1):
            state = T[i] + 1
            observation = x[j]
            eij = emission_prob(state, observation)

            if j == 0:  # first column
                dyn_array[i, j] = transition_prob(0, T[i] + 1) * emission_prob(
                    T[i] + 1, x[j])
            else:
                temp_list = []
                for k in range(0, num_states - 1):
                    aik = transition_prob(state, T[k] + 1)
                    prev_val = dyn_array[k, j - 1]
                    temp_list.append(aik * prev_val)

                dyn_array[i, j] = eij * max(temp_list)
                pointer_array[i, j] = temp_list.index(max(temp_list))

    # traceback and generating the predicted sequence
    predicted = [T[np.argmax(dyn_array[:, -1]) + 1]]
    j = num_obs - 1
    point = int(pointer_array[np.argmax(dyn_array[:, -1]), -1])
    while True:
        predicted.append(T[int(point + 1)])
        j -= 1
        point = int(pointer_array[point, j])
        if j == 0:
            return predicted[::-1]



########################## Output #########################

#Parameters
S = ["H", "T"]                          # observables space
T = [0, 1, 2]                           # state space
e = np.array([(0.5, 0.5), (0.8, 0.2)])  # emission matrix
a = np.array([(0, 0.5, 0.5), (0.01, 0.94, 0.05), (0.01, 0.05, 0.94)])    #transition matrix


# to generate output and view the results set to True. If the next bit of code is 
# to be run, set this one to False (or 0).
if True:
    
    pi, x = gen_hmm(S, T, a, e)
    predicted = viterbi(S, T, a, e, x)
    pi=pi[1:-1]

    pi=[str(i) for i in pi]
    predicted=[str(i) for i in predicted]
    print("Sequence length : {}".format(len(pi)) )
    print("Observed sequence : "+"".join(x))
    print("Hidden :" +"".join(pi))
    print("Predicted : "+"".join(predicted))

    dist=sum([i!=j for i,j in zip(pi,predicted)])
    print("\nThe normalized Hamming distance is {}".format(dist/len(pi)))
    

# to check the effect of switching rate. Set to True for executing this bit, 
# and False (or 0) if not required. Also, leave out the bit of code above, if
# this is to be run.

# This is is prone to error, if the generated sequence, by chance, if of length 1.
if False:    
    prob_switch=[0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
    result=np.zeros(len(prob_switch))

    for i in range(len(prob_switch)):
        a[1,2]=a[2,1]=prob_switch[i]
        a[1,1]=a[2,2]=1-(prob_switch[i]+0.01)    
        total=0
        for j in range(30):
            pi,x=gen_hmm(S,T,a,e)
            predicted=viterbi(S,T,a,e,x)
            pi=pi[1:-1]
            dist=(sum(i!=j for i,j in zip(pi,predicted)))/len(pi)
            total+=dist
        result[i]=total/30

    plt.plot(np.array(prob_switch), result)
    plt.xlabel('Switching probability')
    plt.ylabel('Normalized Hamming distance')
    plt.grid(True)
    plt.savefig('switch.png')
    plt.show()
	       
