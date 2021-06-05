#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Needleman-Wunsch algorithm for aligning two sequences
(Using scoring matrix BLOSUM62)

https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm
https://en.wikipedia.org/wiki/BLOSUM
"""

import numpy as np
from Bio import SeqIO


def blosum_reader(fname):
	'''
	Input: blosum text file
	Output: Returns a 23X23 numpy array containing the blosum matrix
	Remark: I have retained the X,Z and B entries from the blosum62.txt file
		since the protein sequence taken from NCBI website contains those.
	'''
	lst=[]
	myarray=np.empty([0,24])
	#to convert array of strings to an array of ints
	vconv=np.vectorize(lambda x : int(x))	
	with open(fname) as f:
		for line in f:
			if line.split()[0]!='#':
				lst.append(line.split())
	lst.remove(lst[0])		
	for i in lst:
		i.remove(i[0])
		arr=vconv(i)
		arr=np.reshape(arr,(1,24))
		myarray=np.append(myarray,arr,axis=0)
	#Remove unwanted rows and columns from array 
	myarray=myarray[:-1,:-1]		
	return myarray


def readfasta(fname):
	'''
	Input: filename in FASTA format
	Output: A list containing the sequence ids and sequences.
		[(seq_id1, sequence1), (seq_id2, sequence2),...]
		Uses biopython module.
	'''
	sequences=[]
	for record in SeqIO.parse(fname,'fasta'):
		seq_id=record.id
		seq=str(record.seq)
		sequences.append((seq_id, seq))
	return sequences


def score(blosum,a,b):
	'''
	Takes in two characters (for two amino acids) and returns
	the score from the given scoring matrix (blosum62 in this case)
	'''
	#(ideally we should get this sequence from the blosum62.txt 
	# file directly, but I have put it by hand here)
	seq=('A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  B  Z  X')
	lst=seq.split()
	i=lst.index(a)
	j=lst.index(b)
	return blosum[i,j]


def nw_align(sequences,matrix,gap):
	'''
	Inputs:
	sequences: list of two sequences to be aligned
	matrix: scoring matrix to be used
	gap: gap penalty imposed (scalar value)

	Output:
	score of the alignment and the aligned sequences
	'''
	#create an alignment matrix & fill initial values using gap penalty
	val1=len(sequences[0])
	val2=len(sequences[1])
	dp_matrix=np.zeros((val1+1,val2+1))
	for i in range(0,val1+1):
		dp_matrix[i,0]=i*gap
	for j in range(0,val2+1):
		dp_matrix[0,j]=j*gap
	
	#array to store pointers
	pointers=np.zeros([val1+1,val2+1])
	for i in range(0,val1+1):
		pointers[i,0]=1
	for j in range(0,val2+1):
		pointers[0,j]=2
	pointers[0,0]=0	
	
	#to store aligned sequences
	seq1=''
	seq2=''

	#filling in the values
	for i in range(1,val1+1):
		for j in range(1,val2+1):
			up=dp_matrix[i-1,j]+gap
			left=dp_matrix[i,j-1]+gap
			diag=dp_matrix[i-1,j-1]+score(matrix,list(sequences[0])[i-1],list(sequences[1])[j-1])
			dp_matrix[i,j]=max(up,left,diag)
			if max(up,left,diag)==up:
				pointers[i,j]=1
			elif max(up,left,diag)==left:
				pointers[i,j]=2
			elif max(up,left,diag)==diag:
				pointers[i,j]=3
	
	#tracing back to the origin to get aligned sequences		
	i,j=val1,val2
	while True:
		if pointers[i,j]==1:
			seq1=seq1+sequences[0][i-1]
			seq2=seq2+'-'
			i-=1
		elif pointers[i,j]==2:
			seq1=seq1+'-'
			seq2=seq2+sequences[1][j-1]
			j-=1
		elif pointers[i,j]==3:
			seq1=seq1+sequences[0][i-1]
			seq2=seq2+sequences[1][j-1]
			i-=1
			j-=1
		if i==0 and j==0:
			break
	seq1=seq1[::-1]
	seq2=seq2[::-1]
	return dp_matrix[-1,-1],seq1,seq2,pointers,dp_matrix



#Test run. The file ubiquitin.fa contains the sequences (needs to be there in the same directory 
#as the code). The aligned sequences and score etc. can be stored in a file if required.

a=1 		#set to 0 if the following run is to be tested. 
if a==0:
	lst=readfasta('ubiquitin.fa')
	sequences=[lst[0][1],lst[1][1]]
	matrix=blosum_reader('blosum62.txt')
	gap=-4
	
	score,seq1,seq2=nw_align(sequences, matrix, gap)
	print('The sequences being aligned are:\n {} and {}\n\n'.format(lst[0][0], lst[1][0]))
	print('Aligned sequences are:\n \t{}\n\t{}\n\n The score is: {}'.format(seq1,seq2,score))


s1='lhkmqkrimp'.upper()		#input sequences
s2='lyqrrlpp'.upper()
sequences=[s1,s2]
matrix=blosum_reader('blosum62.txt')	#scoring matrix
gap=0
score,seq1,seq2,pointer,dp_mat=nw_align(sequences,matrix,gap)
print('Score : {} \n\n Alignment: \n{}\n{}'.format(score,seq2,seq1))
