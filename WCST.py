"""
Generates cards for the different WCST tasks.

Pauline Bock
09-04-2019
"""
import numpy as np 
import random
import sys

def perception(nb_dim, nb_templates, nb_features):
    """
    Create and return reference cards.
    """
    percep_shape = (nb_templates, nb_dim, nb_features)
    percep = np.zeros(percep_shape, dtype=int)
    
    #Random generating of binary features
    for i in range(0, nb_templates):
        for j in range(0, nb_dim):
            percep[i][j].put([i], 1)

    return percep 

def response_item(nb_dim, nb_features, milner_list):#Milner
    """
    Create and return one card for Milner's version that was not already created.
    """
    item = create_card(nb_dim, nb_features)   
    unique = check_unity(item, milner_list)

    while(unique == 1):
        item = create_card(nb_dim, nb_features)
        unique = check_unity(item, milner_list)

    milner_list.append(item)
    return item

def response_item_Nelson(nb_dim, nb_features, m_percep):
    """ 
    Create and return a card for Nelson's version.
    """
    item = create_card_Nelson(nb_dim, nb_features)  

    return item

def response_item_Reasoning(nb_dim, nb_features, m_percep, reasoning_list):
    """ 
    Create and return a card for Reasoning version that was not already created. (36 ambiguous cards).
    """
    item = create_card_Reasoning(nb_dim, nb_features)  
    #check if different from reference cards
    eq = check_equality(m_percep, item, nb_dim)
    unique = check_unity(item, reasoning_list)

    while(eq == 1 or unique==1):
        item = create_card_Reasoning(nb_dim, nb_features)
        eq = check_equality(m_percep, item, nb_dim)
        unique = check_unity(item, reasoning_list)

    reasoning_list.append(item)
    return item

def check_unity(item, cardlist):
    if inList(item, cardlist) == True:
        return 1
    else:
        return 0
        
def create_card(nb_dim, nb_features):
    """
    Create a card with ambiguity or not (Milner).
    """
    item_shape = (nb_dim, nb_features)
    item = np.zeros(item_shape, dtype=int)
    #Random generating of binary features 

    rd_list = []
    for i in range(0, nb_dim):
        rd_feat = random.randrange(0+nb_features*i,nb_features+i*nb_features)
        while(rd_feat in rd_list):
            rd_feat = random.randrange(0+nb_features*i,nb_features+i*nb_features)
        rd_list.append(rd_feat)
        item.put([rd_feat], 1)
    
    return item

def create_card_Nelson(nb_dim, nb_features):
    """
    Create a card with no ambiguity.
    """
    item_shape = (nb_dim, nb_features)
    item = np.zeros(item_shape, dtype=int)
    #Random generating of binary features 

    rd_list = []
    for i in range(0, nb_dim):
        rd_feat = random.randrange(0+nb_features*i,nb_features+i*nb_features)
        while(rd_feat%nb_features in rd_list):
            rd_feat = random.randrange(0+nb_features*i,nb_features+i*nb_features)
        rd_list.append(rd_feat%nb_features)
        item.put([rd_feat], 1)

    
    return item

def create_card_Reasoning(nb_dim, nb_features):
    """
    Create a card with ambiguity.
    """
    item_shape = (nb_dim, nb_features)
    item = np.zeros(item_shape, dtype=int)

    r = [0,1,2]
    #random 2 different dimensions
    idim1 = random.randint(0,2)
    dim1 = r[idim1]
    r.remove(dim1)
    idim2 = random.randint(0,len(r)-1)
    dim2 = r[idim2]
    r.remove(dim2)
    dim3 = r[0]
    
    feat = random.randint(0,3)
    feat2 = random.randint(0,3)
    while(feat == feat2):
        feat2 = random.randint(0,3)

    np.put(item[dim1],[feat], 1)
    np.put(item[dim2],[feat], 1)
    np.put(item[dim3],[feat2], 1)

    return item


def check_equality(m_percep, item, nb_dim):
    """
    Check if the card is different from the reference ones.
    """
    for temp in range(0, m_percep.shape[0]):
        dim_eq = 0
        for dim in range(0, nb_dim):
            if np.array_equal(m_percep[temp][dim], item[dim]):
                dim_eq +=1

        if dim_eq == nb_dim:
            return 1
    
    return 0

def inList(array, arraylist):
    for element in arraylist:
        if np.array_equal(element, array):
            return True
    return False

