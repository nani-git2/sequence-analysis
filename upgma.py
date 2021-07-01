#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Implementation of Unweighted Pair Group Method with Arithmetic mean (UPGMA) 
wikipedia.org/wiki/UPGMA

Given a distance matrix, generates a phylogenetic tree and returns a string in the Newick format.
'''

import numpy as np
import pandas as pd
import copy


class Node:    
    
    def __init__(self, label: str, children:list, downheight:float, upheight:float, leaves:list):
        self.label = label
        self.children = children            #list containing children
        self.downheight = downheight        # height from below (leaves are zero)
        self.upheight = 0.0                 # height from above (root is zero)
        self.leaves = leaves                # list containing all leaves under a node
        
        if len(self.children) == 0:
            self.downheight = 0.0
     
        

def distance(a,b):
    '''
    Returns the distance between leaf a and b, based on the distance matrix D
    '''
    trans_df = pd.DataFrame(data=dist_matrix, index=species, columns=species)
    return trans_df[a].loc[b]



def clus_distance(C1,C2):
    '''
    Returns the distance between the two clusters C1 and C2 (these are lists of leaves)
    '''
    total=0
    pairs=[[i,j] for i in C1 for j in C2]       #list of pairs between leaves in C1 and C2
    for n in pairs:
        total += distance(n[0],n[1])
    avg_sum = total/(len(C1) * len(C2))
    return avg_sum



def upgma(species, dis_matrix):
    '''
    Takes in a list of species names and a matrix containing their phylogenic distances.
    Returns the phylogenic tree
    '''
    #nodes list is initialized with leaves
    nodes = [Node(species[i], [], 0.0, 0.0, [species[i]]) for i in range(len(species))]            
    dist_array  = copy.deepcopy(dist_matrix)
    #main loop
    while True:
        np.fill_diagonal(dist_array, np.inf)
        #identify nodes with least distance
        children = divmod(np.argmin(dist_array), dist_array.shape[0])   
        downheight = (dist_array[children[0], children[1]])/2
        
        #instantiate new parent node
        parent_node = Node('', [nodes[children[0]], nodes[children[1]]], \
                           downheight = downheight, upheight = 0.0, \
                           leaves = nodes[children[0]].leaves + nodes[children[1]].leaves)        
        
        # assigning appropriate heights to children nodes
        nodes[children[0]].upheight = downheight - nodes[children[0]].downheight
        nodes[children[1]].upheight = downheight - nodes[children[1]].downheight
    
        # removes previous two children nodes from the list of nodes. The order of indices here 
        # must be maintained, else pop returns indexing error
        nodes.pop(children[1])      
        nodes.pop(children[0])      
        nodes += [parent_node]        # add parent node to list of nodes

        if len(nodes) < 2:          # termination criteria
            return nodes[0]
        
        #this deletes the entries corresponding to the closest nodes
        dist_array = np.delete(dist_array, children, axis=0)         
        dist_array = np.delete(dist_array, children, axis=1)
        
        if len(nodes)-1 != dist_array.shape[0]:     # to make sure that the indices are in sync
            raise Exception('nodes and distance array not of compatible sizes')

        #empty row to hold newly computed distances
        new_distances = np.zeros((1,(len(nodes)-1)), dtype=float)       
        
        for k in range(new_distances.shape[1]):     # fills values for new distances
            new_distances[0,k] = clus_distance(nodes[k].leaves, nodes[-1].leaves)
            
        # this contains  new distances, to be appended to distance matrix    
        new_distances = np.append(new_distances, 0.0).reshape((1, len(nodes))) 
        #distance matrix includes new parent
        dist_array = np.append(dist_array, new_distances[0,:-1].reshape((1, len(nodes)-1)), axis=0)     
        dist_array = np.append(dist_array, new_distances.transpose(), axis=1)
                
        
                
def tree_to_newick(tree):
    '''
    Returns a string with the Newick format for the generated tree
    '''
    if tree.children==[]:
        return tree.label+":"+str(tree.upheight)
    else:
        return "("+",".join([tree_to_newick(x) for x in tree.children])+"):"+str(tree.upheight)




if __name__ == '__main__':
#input example from wikipedia.org/wiki/UPGMA

    species = ['a', 'b', 'c', 'd', 'e']
    dist_matrix = np.array([
        [0, 17, 21, 31, 23],
        [17, 0, 30, 34, 21],
        [21, 30, 0, 28, 39],
        [31, 34, 28, 0, 43],
        [23, 21, 39, 43, 0]],
        dtype=float)
    
    tree = upgma(species, dist_matrix)
    print(tree_to_newick(tree))
