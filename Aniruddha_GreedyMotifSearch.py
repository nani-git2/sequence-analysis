#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: aniruddha

Code for the subtle motif problem (given a set of DNA strings, finds a
set of commmon motifs (with few differences) from each string).
Using greedy search algorithm.

The input data used here is taken from Ch.2 of the online version of Compeau book. 
https://www.bioinformaticsalgorithms.org/bioinformatics-chapter-2

(instead of denoting the nucleotides by (A,G,C,T), I have just used numbers (1,2,3,4)
and used their ASCII equivalents)
"""

import numpy as np
from collections import Counter
import progressbar as pb


#parameters
k=15        #length of "hidden" motif in each Dna string

char_int=np.vectorize(ord) 	#function. Takes in array and converts 
int_char=np.vectorize(chr)	# characters to their ASCII code (and back) 

file1=open(r'subtle_motif.txt')
dna=np.array([])
for line in file1:
    line=line.replace('*','')
    line=line.replace('\n','')
    a=np.array(list(line))
    dna=np.append(dna,a)
    
dna=dna.reshape((10,600))    
dna=char_int(dna)       #dna strings are stored in the form of ints, in rows of this array


def Profile(motifs):
	'''
	Takes in the array (containing the collection of motifs) and 
	returns the profile matrix for it. A pseudocount of 1 is used.
	'''
	t=motifs.shape[0]
	n=motifs.shape[1]
	profile=np.zeros((4,n))
	for i in range(n):
		countA=0
		countG=0
		countT=0
		countC=0
		for j in range(t):
			base=chr(motifs[j,i])
			if base=='A':
				countA+=1
			elif base=='G':
				countG+=1
			elif base=='T':
				countT+=1
			elif base=='C':
				countC+=1

		profile[0,i]=countA
		profile[1,i]=countG
		profile[2,i]=countT
		profile[3,i]=countC
	return (profile+1)/(t+4)


def ProbabilityKmer(profile,kmer):
	'''
	Given a profile matrix and a k-mer, returns the
	probabililty of that k-mer
	'''
	prob=1
	k=len(kmer)
	for j in range(k):
		base=chr(kmer[j])
		if base=='A':
			prob*=profile[0,j]
		elif base=='G':	
			prob*=profile[1,j]
		elif base=='T':
			prob*=profile[2,j]
		elif base=='C':
			prob*=profile[3,j]
	return prob


def ProfileMostProbable(profile,Dna):
	'''
	Given a profile matrix and a Dna string (as ASCII int), 
	returns	the most probable k-mer from the string
	'''
	k=profile.shape[1]
	prob0=0
	most_probable=np.zeros(k)	
	for i in range(0,len(Dna)-k+1):
		kmer=Dna[i:i+k]
		prob=ProbabilityKmer(profile,kmer)
		if prob>=prob0:
			most_probable=kmer
			prob0=prob
	return most_probable
		

def Score(motifs):
	'''
	Takes in a collection of motifs (as an array of ASCII values)
	and returns the score
	'''
	n=motifs.shape[1]
	t=motifs.shape[0]
	count=0

	for i in range(n):
		column=motifs[:,i]
		c=Counter(column) 
		freq_base=max(c, key=c.get) 	#most frequent base in a position(ASCII)
		for j in range(t):
			if motifs[j,i]!=freq_base:
				count+=1
	
	return count 


def GreedyMotifSearch(Dna,k):
	'''
	Inputs: 
	Matrix Dna, where each row is supposed to be a Dna string
	(they need not all be of same length)
	k: length of motif to find

	Returns the best motif (most common) in each Dna string
	'''
	t=Dna.shape[0] 		#number of strings of Dna
	n=Dna.shape[1]		#length of any given Dna string	
	
	BestMotifs=Dna[:,0:k]	#initialize best motifs formed 
            				#by first k-mer in each string
	for i in pb.progressbar(range(0, n-k+1)):
		motif=Dna[0,i:i+k]
		motif=motif.reshape((1,k))
		for j in range(1,t):
			profile=Profile(motif)
			motif1=ProfileMostProbable(profile, Dna[j,:])
			motif1=motif1.reshape((1,k))
			motif=np.append(motif, motif1,axis=0)

		if Score(motif)<Score(BestMotifs):
			BestMotifs=motif

	return BestMotifs


best_motifs=int_char(GreedyMotifSearch(dna,k))
print(best_motifs)
