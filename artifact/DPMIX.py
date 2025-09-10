# -*- coding: utf-8 -*-
"""
DPMIX.py contains a comprehensive set of functions for analyzing DP-Mix.
"""
import pickle


from math import exp
from scipy import constants
from Routing import Routing

from scipy.stats import expon
import simpy
import random
import numpy  as np
import pickle
import math
import json
from FCP_Functions import Greedy_, Greedy, Random, Greedy_For_Fairness


from itertools import product
import numpy as np
    

def pick_m_from_n(n, m):
    if m > n:
        raise ValueError("m must be less than or equal to n")
    return np.random.choice(n, size=m, replace=False)
   
def P_compute(P,G1,G2):
    
    W = len(P)
    List = []
    for i in range(W):
        for j in range(W):
            for k in range(W):
                List.append(P[i]*(G1[i,j])*(G2[j,k]))
    return List

def normalize_rows_by_sum(matrix):
    row_sums = matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    return matrix / row_sums

#A = np.array([[1, 2, 3],
#             [4, 0, 2],
#            [0, 0, 0]])

#normalized = normalize_rows_by_sum(A)
#print(normalized)


def compute_conditionals(probs_):
    probs = probs_.copy()
    W = round(len(probs) ** (1/3))
    A = np.zeros((W, W, W), dtype=float)
    
    for i in range(W):
        for j in range(W):
            start = i * (W ** 2) + j * W
            end = start + W
            A[i, j, :] = probs[start:end]
    
    P_y_given_x = np.zeros((W, W))  # rows: X=i, cols: Y=j
    P_z_given_y = np.zeros((W, W))  # rows: Y=i, cols: Z=j
    P_List = []
    
    for i in range(W):
        for j in range(W):
            P_y_given_x[i, j] = np.sum(A[i, j, :])      # P(X=i, Y=j)
            P_z_given_y[i, j] = np.sum(A[:, i, j])      # P(Y=i, Z=j)
            
    for i in range(W):
        P_List.append(np.sum(probs_[i*(W**2):(i+1)*(W**2)]))
            
    
    return P_List,normalize_rows_by_sum(P_y_given_x), normalize_rows_by_sum(P_z_given_y)


def Gradient_descent_(Gamma,Beta,a):#Gamma and beta restrication matrix with learning parameter a 
    import numpy as np
    (n1,n2) = np.shape(Gamma)
    empty = np.zeros((n1,n2))
    ones_Gamma = np.ones((n1,n2)).dot(Gamma)
    one_beta = np.ones((n1,1)).dot(np.transpose(Beta))
    
    Gamma1  = np.copy(Gamma)
    ALPHA = np.log(Gamma1+0.0000001)
    
    for i in range(n1):
        for j in range(n2):
            
            x = np.copy(empty)
            x[i,j] = ALPHA[i,j]
            
            y = np.transpose(x)
            z = np.trace(y.dot(ones_Gamma)-y.dot(one_beta))
            ALPHA[i,j] = ALPHA[i,j] +a*z

    return normalize_rows_by_sum(np.exp(ALPHA))

def Gradient_descent__(Gamma,Beta,a):#Gamma and beta restrication matrix with learning parameter a 
    import numpy as np
    (n1,n2) = np.shape(Gamma)
    empty = np.zeros((n1,n2))
    ones_Gamma = np.ones((n1,n2)).dot(Gamma)
    one_beta = np.ones((n1,1)).dot(np.transpose(Beta))
    
    Gamma1  = np.copy(Gamma)
    ALPHA = np.log(Gamma1+0.0000001)
    
    for i in range(n1):

            
        x = np.copy(empty)
        x[i,:] = ALPHA[i,:]
        
        y = np.transpose(x)
        z = np.trace(y.dot(ones_Gamma)-y.dot(one_beta))
        ALPHA[i,:] = ALPHA[i,:] +a*z

    return normalize_rows_by_sum(np.exp(ALPHA))






def Gradient_descent(A,Beta,a):
    import numpy as np

    """
    Normalize a square matrix:
    1. Normalize each column by its sum.
    2. Normalize each row of the resulting matrix by its row sum.
    
    Parameters:
    A (np.ndarray): An n x n square matrix.
    
    Returns:
    np.ndarray: The normalized matrix.
    """
    A = np.array(A, dtype=float)  # Ensure it's a float NumPy array

    # Step 1: Normalize each column by its sum
    col_sums = A.sum(axis=0)
    # Avoid division by zero
    col_sums[col_sums == 0] = 1
    A = A / col_sums

    # Step 2: Normalize each row by its sum
    row_sums = A.sum(axis=1)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    A = (A.T / row_sums).T

    return A







def Gradient_descent_IT(Gamma,Beta,a,It):
    count = 0
    while count < It:
       Gamma =  Gradient_descent(Gamma,Beta,a)
       count +=1


    return Gamma

def generate_combinations(list1, list2, list3):
    return list(product(list1, list2, list3))
def count_unique_per_column(data):
    arr = np.transpose(np.array(data))
    return [len(np.unique(arr[:, i])) for i in range(arr.shape[1])]

def JAR_Regions(List,W):
    list1 = List[:W]
    list2 = List[W:2*W]
    list3 = List[2*W:]
    a = (generate_combinations(list1,list2,list3))
    b = count_unique_per_column(a)
    return b
                
              
def convert_to_lat_lon(x, y, z):
    radius = 6371  # Earth's radius in kilometers

    # Convert Cartesian coordinates to spherical coordinates
    longitude = math.atan2(y, x)
    hypotenuse = math.sqrt(x**2 + y**2)
    latitude = math.atan2(z, hypotenuse)

    # Convert radians to degrees
    latitude = math.degrees(latitude)
    longitude = math.degrees(longitude)

    return latitude, longitude

def classify_region(lat, lon):
    if 15 <= lat <= 75 and -170 <= lon <= -50:
        return "North America"
    elif -60 <= lat <= 15 and -90 <= lon <= -30:
        return "South America"
    elif 35 <= lat <= 70 and -10 <= lon <= 40:
        return "Europe"
    elif 5 <= lat <= 80 and 40 <= lon <= 180:
        return "Asia"
    else:
        return "Other"

def classify_points(matrix):
    regions = []
    counts = {"North America": 0, "South America": 0, "Europe": 0, "Asia": 0, "Other": 0}

    for row in matrix:
        lat, lon = convert_to_lat_lon(*row)
        region = classify_region(lat, lon)
        regions.append(region)
        counts[region] += 1

    return regions


def find_row_permutation(A, B):
    #print(A,B)
    """
    Finds the row permutation mapping from A to B.

    Parameters:
        A (numpy.ndarray): Original matrix (N x M)
        B (numpy.ndarray): Permuted matrix (N x M)

    Returns:
        list: Mapping of rows from A to their positions in B
    """
    A = np.array(A)
    B = np.array(B)

    # Convert each row to a tuple so we can use list index
    A_list = [tuple(row) for row in A]
    B_list = [tuple(row) for row in B]

    # Find indices of A's rows in B
    mapping = [A_list.index(row) for row in B_list]

    return mapping
#Example
'''
A = np.array([[10, 20], 
              [30, 40], 
              [50, 60]])

B = np.array([[50, 60],  # Row from index 2 in A
              [10, 20],  # Row from index 0 in A
              [30, 40]]) # Row from index 1 in A

L = find_row_permutation(A, B)
print(L)  # Output: [2, 0, 1]

List_ = [0]*len(List1)
for i in range(len(List1)):
    
    List_[List2.index(List1[i])] = List1[i]
    
    
'''


def MAP_to_MAP(L1, L2):
    """
    Computes the permutation from A to C given the permutations from A to B (L1) and B to C (L2).
    
    Parameters:
        L1 (list): Permutation from A to B
        L2 (list): Permutation from B to C
    
    Returns:
        list: Permutation from A to C
    """
    return [L2[i] for i in L1]


'''
import numpy as np

# Define example matrices
A = np.array([[10, 20], 
              [30, 40], 
              [50, 60]])

B = np.array([[50, 60],  # Row from index 2 in A
              [10, 20],  # Row from index 0 in A
              [30, 40]]) # Row from index 1 in A

C = np.array([[30, 40],  # Row from index 1 in B (originally from index 2 in A)
              [50, 60],  # Row from index 0 in B (originally from index 0 in A)
              [10, 20]]) # Row from index 2 in B (originally from index 1 in A)

# Find permutations
L1 = find_row_permutation(A, B)  # Mapping from A to B
L2 = find_row_permutation(B, C)  # Mapping from B to C

# Compute permutation from A to C
L_final = MAP_to_MAP(L1, L2)

# Print results
print("L1 (A to B):", L1)  # Expected: [2, 0, 1]
print("L2 (B to C):", L2)  # Expected: [1, 0, 2]
print("Final Mapping (A to C):", L_final)  # Expected: [1, 2, 0]
'''    
#Example
'''
A = np.array([[10, 20], 
              [30, 40], 
              [50, 60]])

B = np.array([[50, 60],  # Row from index 2 in A
              [10, 20],  # Row from index 0 in A
              [30, 40]]) # Row from index 1 in A

L = find_row_permutation(C,A)
print(L)  # Output: [2, 0, 1]    
    
C = np.array([[50, 60],
              [30, 40],# Row from index 2 in A
              [10, 20]]) # Row from index 1 in A    
L2 = find_row_permutation(A,B)
print(L2)

print(MAP_to_MAP(L,L2))
'''
def remove_elements_by_index(values, indices):
    """
    Removes elements from 'values' based on the indices given in 'indices'.
    
    Parameters:
        values (list): A list of values.
        indices (list): A list of indices referring to elements to be removed.

    Returns:
        list: The modified 'values' list with specified indices removed.
    """
    # Convert indices to a set to avoid duplicate processing
    index_set = set(indices)
    
    # Create a new list excluding elements at specified indices
    filtered_values = [val for i, val in enumerate(values) if i not in index_set]
    
    return filtered_values


def add_elements_by_index(values, indices):
    """
    Inserts `0` at the specified indices in `values`, maintaining order.

    Parameters:
        values (list): The original list of values.
        indices (list): The list of indices where `0` should be inserted.

    Returns:
        list: A new list with `0` inserted at the specified indices.
    """
    result = []  # Store the final modified list
    value_index = 0  # Track the index in the original values list
    
    for i in range(len(values) + len(indices)):  # Iterate through new length
        if i in indices:  
            result.append(0)  # Insert `0` at specified index
        else:
            result.append(values[value_index])  # Insert original element
            value_index += 1  # Move to the next element in values

    return result

def Corruption_c(List,N):
    Corrupted_List ={}
    
    for i in range(N):

        Corrupted_List['PM'+str(i+1)] = False
        
    for j in List:
        Corrupted_List['PM'+str(j+1)] = True
        
    return Corrupted_List
            

def permutation_matrix(AA, BB):
    
    A = [(item*10000)/10000 for item in AA]
    B = [(item*10000)/10000 for item in BB]    
    """
    Computes the permutation matrix that maps list A to list B.

    Args:
        A (list): The original list.
        B (list): The target list (a permutation of A).

    Returns:
        numpy.ndarray: The permutation matrix P such that P @ A_sorted = B.
    """
    if sorted(A) != sorted(B):
       # print(A,B)
        raise ValueError("Lists must be permutations of each other.")
    
    n = len(A)
    P = np.zeros((n, n))

    # Create index mapping from A to B
    index_map = {value: i for i, value in enumerate(B)}

    for i, value in enumerate(A):
        P[index_map[value], i] = 1  # Place a 1 at the corresponding position

    return P

'''
# Example usage:
A = [3, 1, 2, 4]
B = [1, 3, 4, 2]

P = permutation_matrix(A, B)
print("Permutation Matrix:\n", P)
'''










def Latency_extraction(data0,Positions,L):
    List = []
    n1,n2 = np.shape(data0)
    for i in range(L-1):
        List_ = []
        for j in range(int(n1/L)):
            List__ = []
            for k in range(int(n1/L)):
                List__.append(data0[Positions[i][j],Positions[i+1][k]])
            List_.append(List__)
        List.append(List_)
    return List
                
def Norm_List(List,term):
    S = np.sum(List)
    return [List[i]*(term/S)for i in range(len(List))]
def To_list(List):
    if type(List) == list:
        return List
    import numpy as np
    List_ = List.tolist()
    if len(List_)==1:
        output = List_[0]
    else:
        output = List_
    
    return output
def dist_List(List):
    Sum = np.sum(List)
    
    return [List[i]/Sum for i in range(len(List))]

#print(dist_List([14,17,9]))
    
'''
#Ex:
A = np.matrix([[2,4,7,5,6,7],[5,6,7,1,1,1],[6,7,8,5,3,2],[2,4,7,5,6,7],[5,6,7,1,1,1],[6,7,8,5,3,2]])
P = [[2,5],[0,4],[1,3]]
List = Latency_extraction(A, P, 3)

print(List)
print(Norm_List([1,2,3,4],4))
'''



def find_median_from_cdf(cdf):
    """
    Finds the median of a discrete distribution given its CDF.

    Args:
        cdf (list): A list representing the cumulative probabilities of a discrete distribution.

    Returns:
        int: The index of the median value in the distribution.
    """
    for i, value in enumerate(cdf):
        if value >= 0.5:
            return i  # The first index where CDF reaches or exceeds 0.5 is the median index.
    
    raise ValueError("Invalid CDF: It should reach at least 0.5 somewhere.")



def I_key_finder(x,y,z,matrix,data):
    
    List = [x,y,z]
    Index1 = np.sum(np.abs(matrix - List),axis = 1)
    index = Index1.tolist()
    Index2 = min(index)
    Index = index.index(Index2)

    
    return data[Index]
            
def Medd(List):
    N = len(List)

    List_ = []
    import statistics
    for i in range(N):

        List_.append( statistics.median(List[i]))
        
    return List_


#print(Medd([[1,2,7,9,10],[1,9,10]]))

def Loc_finder(I_key,data):
    for i in range(len(data)):
        if data[i]['i_key'] == I_key:
            return i
    
def Ent(List):
    L =[]
    for item in List:
       
        if item!=0:
            L.append(item)
    l = sum(L)
    for i in range(len(L)):
        L[i]=L[i]/l
    ent = 0
    for item in L:
        ent = ent - item*(np.log(item)/np.log(2))
    return ent

def Med(List):
    N = len(List)

    List_ = []
    import statistics
    for i in range(N):

        List_.append( statistics.median(List[i]))
        
    return List_

def SC_Latency(Matrix,Positions,L):
    N = len(Matrix)
    A = np.zeros((N,N))
    W = int(N/L)
    for i in range(N):
        n1 = Positions[i//W][i%W]
        for j in range(N):
            n2 = Positions[j//W][j%W]
            A[i,j] = Matrix[n1,n2]
  
    return A
    



from itertools import combinations
from math import radians, sin, cos, sqrt, atan2






class CirMixNet(object):
    
    def __init__(self,Targets,Iteration,Capacity,run,delay1,delay2,W1,W2,L,base,Initial = False):
        self.Iterations = Iteration
        #print( self.Iterations )
        self.ML_a = 0.2
        self.ML_It = 1
        self.CAP = Capacity
        self.delay1 = delay1
        self.delay2 = delay2
        self.Targets = Targets
        self.W1 = W1
        self.W2 = W2
        self.L = L
        self.N1 = self.W1*self.L
        self.N2 = self.W2*self.L
        self.b = base
        self.WW = {'NYM':self.W1,'RIPE':self.W2}
        self.run = run
        self.Data_type = ['NYM','RIPE']
        self.Design = ['DNA']
        self.Method = ['L_CD','L_C','Band']
        #self.Method = ['L_C']        
        self.Tau = [0.085,0.2,0.4,0.6,0.8,1]
        self.T = [2,12,25,38,50,80]  
        self.EPS = [ 0,3,5,7,8]
        self.nn = 20
        self.CF = 0.3
        self.Initial = Initial
        self.RST_tau = 0.6
        self.RST_T = 12
        self.CDF = [i/10 for i in range(51)]
        #self.Data_Set_General = self.data_generator(Iteration)
        self.Data_Set_General = {'NYM':{},'RIPE':{}}
        
        if self.Initial== False:
            with open('dataset.pkl','rb') as pkl_file:
                data0 = pickle.load(pkl_file)
            #print( self.Iterations )
            for item in self.Data_type:
                
                for It in range(self.Iterations):
                    #print( self.Iterations )
                    self.Data_Set_General[item]['It'+str(It+1)] = data0[item]['It'+str(It+1)]
                    #print(data0[item]['It'+str(It+1)]['DNA']['Loc'])
                
            
            
        data_W1 = {}
        for i1 in range(self.W1*(self.L)):
            data_W1['PM'+str(i1+1)] = False
        data_W2 = {}
        for i1 in range(self.W2*(self.L)):
            data_W2['PM'+str(i1+1)] = False   
        self.Corrupted_Mix = {self.W1:data_W1,self.W2:data_W2}
        self.dict_R = {'RLP':'Linear' , 'RST':'alpha_closest' , 'REB': 'EXP_New'}
            
        

    def JAR_data(self,Iterations):
        data1 = {'NYM':{},'RIPE':{}}
        
        with open('dataset.pkl','rb') as pkl_file:
            data0 = pickle.load(pkl_file)
            #print( self.Iterations )
            for item in self.Data_type:
                
                for It in range(self.Iterations):
                    #print( self.Iterations )
                    Num_Regions = JAR_Regions(classify_points(data0[item]['It'+str(It+1)]['DNA']['Loc']),self.WW[item])
                    data1[item]['It'+str(It+1)] = Num_Regions
        return data1
        
        
    def data_RIPE(self,Iteration):
        corrupted_Mix = {}
        for i in range(9):
            corrupted_Mix['PM%d' %i] = False        
        from data_set import Dataset
        from DNA import DNA
        from Clustering import Clustering
        from Arrangement import Mix_Arrangements
        data_class = Dataset(self.W, self.L)
        data0 = {}
        
        for It in range(Iteration):
            data0_ = {}
            LATLON, Cart, Matrix = data_class.RIPE()
            Cart_1 = np.copy(Cart)
            Matrix_1 = np.copy(Matrix)
            Cart_2 = np.copy(Cart)
            Matrix_2 = np.copy(Matrix)            
            Cart_3 = np.copy(Cart)
            Matrix_3 = np.copy(Matrix)
            Cart_4 = np.copy(Cart)
            Matrix_4 = np.copy(Matrix)   
#######################Random###################################################            
            data0_['Random'] = [np.transpose(Cart_1), Matrix_1]
            
#######################LARMix###################################################
            #Clusterign the mixnodes##########################################
            Clustring_ = 'kmedoids'  # Change Clustring to Clustering in the next line
            K_cluster = 5
            Class = Clustering(np.transpose(Cart_1), Clustring_, K_cluster, self.L, corrupted_Mix)
            Clusters = Class.Centers
            Mixes = Class.Mixes
            Labels = Class.Labels
            Fisher = Class.Fisher
            Map_1 = Class.Map  
            #Diversification##################################################
            Class_D = Mix_Arrangements(Mixes, Labels, Clusters, corrupted_Mix, self.L)
            Topology = Class_D.Topology
            Mapping = Class_D.Mapping_
            #Mapping##########################################################
            Loc_1,Mat_1 = Class.Map_to_new(np.transpose(Cart_2), Matrix_2, Map_1)
            Loc_2,Mat_2 = Class.Map_to_new(Loc_1, Mat_1, Mapping)
            data0_['Div'] = [Loc_2,Mat_2]

            
##############################DNA###############################################            

            # Create the DNA object
            dna = DNA(np.transpose(Cart_3), Matrix_3,self.L)
            
            # Test the DNA_Arrangement function
            selection = dna.DNA_Arrangement()

            mix_net, positions = dna.Map(selection)          
            
            data0_['DNA'] = [positions,mix_net]
            
            
###########################Synthetic data########################################
            
            synthetic_location, synthetic_latency = data_class.generate_synthetic_data(Cart_4, Matrix_4)
            data0_['Synthetic'] = [np.transpose(synthetic_location), synthetic_latency]
            data0['Iteration'+str(It+1)] = data0_
        return data0
    
    


    def data_generator(self,Iteration):
        corrupted_Mix = {}
        for i in range(9):
            corrupted_Mix['PM%d' %i] = False        
        from data_set import Dataset
        from DNA import DNA
        
        data0 = {}
        W = {'NYM':self.W1,'RIPE':self.W2}
        
        for Data in ['NYM','RIPE']:
            data_class = Dataset(W[Data], self.L)
            data1 = {}
            for It in range(Iteration):

                
                LATLON, Cart, Matrix_,Omega_ = eval('data_class.' + Data+ '()')
                Matrix = np.copy(Matrix_)
                Omega = Omega_.copy()
                #########DNA#########################################################
                Class_DNA = DNA(Omega,Matrix,self.L,self.b)    
                Positions,_ = Class_DNA.DNA_Arrangement_W()
                
                Latency_List = Latency_extraction(Matrix, Positions, self.L)
                O_ = [Norm_List(item,W[Data]) for item in _]
                
                data4 = {'Latency_List': Latency_List,'Omega':O_, 'Positions':Positions,'Loc':np.transpose(Cart),'Matrix':Matrix,'x':Omega_,'xx':_}
                '''
                ########Random#######################################################
                Positions_R = [[W[Data]*j+i for i in range(W[Data])] for j in range(self.L)]
                Latency_List_R = Latency_extraction(Matrix, Positions_R, self.L)
                Omega_ = [[Omega[j*W[Data]+i] for i in range(W[Data])] for j in range(self.L)]
                O_R = [Norm_List(item,W[Data]) for item in Omega_]
                data3 = {'Latency_List': Latency_List_R,'Omega':O_R, 'Positions':Positions_R,'Loc':np.transpose(Cart)}
                '''
                data1['It'+str(It+1)] = {'DNA':data4}
                
            data0[Data] = data1
                
        return data0
    
    
    def data_FCP(self,Iteration):
        data0 = {}
        W = {'NYM':self.W1,'RIPE':self.W2}
        
        for Data in ['NYM','RIPE']:
            data1 = {}
            for It in range(Iteration):

                Matrix = self.Data_Set_General[Data]['It'+str(It+1)]['DNA']['Matrix']
                Latency_List = self.Data_Set_General[Data]['It'+str(It+1)]['DNA']['Latency_List']
                Positions = self.Data_Set_General[Data]['It'+str(It+1)]['DNA']['Positions']
                Loc = self.Data_Set_General[Data]['It'+str(It+1)]['DNA']['Loc']
                O_ = self.Data_Set_General[Data]['It'+str(It+1)]['DNA']['Omega']
                Omega = self.Data_Set_General[Data]['It'+str(It+1)]['DNA']['x']
                _ = self.Data_Set_General[Data]['It'+str(It+1)]['DNA']['xx']

                O_1 = []
                for item in _:
                    O_1 += item
                #Latency_List_R = Latency_extraction(Matrix, Positions_R, self.L)
                #Omega_x = [[Omega[j*W[Data]+i] for i in range(W[Data])] for j in range(self.L)]
                #O_R = [Norm_List(item,W[Data]) for item in Omega_x]
                #Omega_ = [[Omega[j*W[Data]+i] for i in range(W[Data])] for j in range(self.L)]
                
                data4 = {'Latency_List': Latency_List,'Omega':O_, 'Positions':Positions,'Loc':Loc,'beta':[Omega,O_1],'L_M':Matrix}
                '''
                ########Random#######################################################
                Positions_R = [[W[Data]*j+i for i in range(W[Data])] for j in range(self.L)]
                Latency_List_R = Latency_extraction(Matrix, Positions_R, self.L)
                Omega_ = [[Omega[j*W[Data]+i] for i in range(W[Data])] for j in range(self.L)]
                O_R = [Norm_List(item,W[Data]) for item in Omega_]
                data3 = {'Latency_List': Latency_List_R,'Omega':O_R, 'Positions':Positions_R,'Loc':np.transpose(Cart)}
                '''
                data1['It'+str(It+1)] = data4
                
            data0[Data] = data1
                
        return data0        

    def data_FCP_(self,Iteration):
        corrupted_Mix = {}
        for i in range(9):
            corrupted_Mix['PM%d' %i] = False        
        from data_set import Dataset
        from DNA import DNA
        
        data0 = {}
        W = {'NYM':self.W1,'RIPE':self.W2}
        
        for Data in ['NYM','RIPE']:
            data_class = Dataset(W[Data], self.L)
            data1 = {}
            for It in range(Iteration):

                
                LATLON, Cart, Matrix,Omega = eval('data_class.' + Data+ '()')
                #########DNA#########################################################
                Class_DNA = DNA(Omega,Matrix,self.L,self.b)    
                Positions,_ = Class_DNA.DNA_Arrangement_W()
                #Positions_R = [[W[Data]*j+i for i in range(W[Data])] for j in range(self.L)]
                Latency_List = Latency_extraction(Matrix, Positions, self.L)
                O_ = [Norm_List(item,W[Data]) for item in _]
                O_1 = []
                for item in _:
                    O_1 += item
                #Latency_List_R = Latency_extraction(Matrix, Positions_R, self.L)
                #Omega_x = [[Omega[j*W[Data]+i] for i in range(W[Data])] for j in range(self.L)]
                #O_R = [Norm_List(item,W[Data]) for item in Omega_x]
                #Omega_ = [[Omega[j*W[Data]+i] for i in range(W[Data])] for j in range(self.L)]
                
                data4 = {'Latency_List': Latency_List,'Omega':O_, 'Positions':Positions,'Loc':np.transpose(Cart),'beta':[Omega,O_1],'L_M':Matrix}
                '''
                ########Random#######################################################
                Positions_R = [[W[Data]*j+i for i in range(W[Data])] for j in range(self.L)]
                Latency_List_R = Latency_extraction(Matrix, Positions_R, self.L)
                Omega_ = [[Omega[j*W[Data]+i] for i in range(W[Data])] for j in range(self.L)]
                O_R = [Norm_List(item,W[Data]) for item in Omega_]
                data3 = {'Latency_List': Latency_List_R,'Omega':O_R, 'Positions':Positions_R,'Loc':np.transpose(Cart)}
                '''
                data1['It'+str(It+1)] = data4
                
            data0[Data] = data1
                
        return data0    
    
    
    
    def PDFs(self,data):
        from Routing import Routing
        data0 = {}
        #print(len(data['NYM']))
        for It in range(len(data['NYM'])):
            data_0 = {}
            
            for dataset_type in self.Data_type:
                #data_1 = {}
                Class_R = Routing(self.WW[dataset_type]*self.L,self.L)
                        
                for design in self.Design:
                    data_2 = {}
                    Rnd_Num = int(self.WW[dataset_type]*np.random.rand(1)[0])
                    if Rnd_Num == self.WW[dataset_type]:
                        Rnd_Num = Rnd_Num -1
                    L_Mix1 = data[dataset_type]['It'+str(It+1)][design]['Latency_List']
                    #print(L_Mix1)
                    L_Mix  = [L_Mix1[0][Rnd_Num],np.matrix(L_Mix1[0]),np.matrix(L_Mix1[1]),L_Mix1[1][Rnd_Num]]
                    O_Mix  = data[dataset_type]['It'+str(It+1)][design]['Omega'] 
                    for method in self.Method:
                        if method == 'L_C':
                            Ent_List,La_List = Class_R.AL_EXP(L_Mix,False)
                        elif method =='L_CD':
                            Ent_List,La_List = Class_R.AL_EXP(L_Mix,True)
                        elif method == 'Band':
                            Ent_List,La_List = Class_R.Band_EXP(O_Mix)
                        
                        data_2[method] = [Ent_List,La_List]
                        
                data_0[dataset_type] = data_2
                
            data0['It'+str(It+1)] = data_0
                        
                        
        return data0
                            


    
    def PDFs1(self,data):
        
        from Routing import Routing
        data0 = {}
        #print(len(data['NYM']))
        for It in range(len(data['NYM'])):
            data_0 = {}
            
            for dataset_type in self.Data_type:
                #data_1 = {}
                Class_R = Routing(self.WW[dataset_type]*self.L,self.L)
                        

                data_2 = {}
                Rnd_Num = int(self.WW[dataset_type]*np.random.rand(1)[0])
                if Rnd_Num == self.WW[dataset_type]:
                    Rnd_Num = Rnd_Num -1
                L_Mix1 = data[dataset_type]['It'+str(It+1)]['Latency_List']
                #print(L_Mix1)
                L_Mix  = [L_Mix1[0][Rnd_Num],np.matrix(L_Mix1[0]),np.matrix(L_Mix1[1]),L_Mix1[1][Rnd_Num]]
                O_Mix  = data[dataset_type]['It'+str(It+1)]['Omega'] 
                for method in self.Method:
                    Dis = []
                    for eps in self.EPS:
                        if method == 'L_C':
                            dis_1,P_ = Class_R.EXP_Latency_C1(L_Mix,eps)
                            e1,e2,e3 = Class_R.R1_R2(dis_1)
                            dis_ = [e1,e2]
                        elif method =='L_CD':
                            dis_1,P_ = Class_R.EXP_Latency_CD1(L_Mix,eps)
                            e1,e2,e3 = Class_R.R1_R2(dis_1)
                            dis_ = [e1,e2]                                
                        elif method == 'Band':
                            dis_1,P_ = Class_R.EXP_Band1(O_Mix,eps)
                            e1,e2,e3 = Class_R.R1_R2(dis_1)
                            dis_ = [e1,e2]                                
                        Dis.append([dis_,P_])
                    
                    data_2[method] = Dis
                        
                data_0[dataset_type] = data_2
                
            data0['It'+str(It+1)] = data_0
                        
                        
        return data0

    def PDFs_FCP(self,data):
        from Routings import Routing
        #self.Tau = [0.6]
        #self.T = [2]
        data0 = {}
        for It in range(len(data['NYM'])):
            data_0 = {}
            
            for dataset_type in self.Data_type:
                
                data_1 = {}
                Class_R = Routing(self.WW[dataset_type]*self.L,self.L)
                        

                data_2 = {}
                L_Mix = data[dataset_type]['It'+str(It+1)]['Latency_List'] 
                O_Mix = data[dataset_type]['It'+str(It+1)]['Omega'] 
                for method in self.Method:
                    data_3 = {}
                    
                    if not method == 'RST':
            
                        for tau in self.Tau:

                            
                            List_R = [Class_R.Matrix_routing(method,np.matrix(L_Mix[j]),O_Mix[j+1],tau) for j in range(self.L-1)]
                            List_B = [Class_R.BALD(List_R[j],O_Mix[j+1],O_Mix[j]) for j in range(self.L-1)]
                            data_3['tau'+str(int(10*tau))] = [List_R,List_B]
                                

                    else:
            
                        for tau in self.Tau:
                            
                            List_R = [Class_R.Matrix_routing(method,np.matrix(L_Mix[j]),O_Mix[j+1],(tau,self.RST_T)) for j in range(self.L-1)]
                            List_B = [Class_R.BALD(List_R[j],O_Mix[j+1],O_Mix[j]) for j in range(self.L-1)]
                            data_3['tau'+str(int(10*tau))] = [List_R,List_B]
                            
                        for _ in self.T: 
                            
                            List_R = [Class_R.Matrix_routing(method,np.matrix(L_Mix[j]),O_Mix[j+1],(self.RST_tau,_)) for j in range(self.L-1)]
                            List_B = [Class_R.BALD(List_R[j],O_Mix[j+1],O_Mix[j]) for j in range(self.L-1)]
                            data_3['T'+str(int(_))] = [List_R,List_B]  
                            
                    data_2[method] = data_3                      

                data_0[dataset_type] = data_2
            data0['It'+str(It+1)] = data_0
        return data0                               

 
    

    def PDF_Latency(self,data,mod = False):
        self.ML_a = 0.2
        
        from Routing import Routing
        data0 = {}
        for It in range(len(data['NYM'])):
            data_0 = {}
            
            for dataset_type in self.Data_type:
                #print('1')
                Rnd_Num = int(self.WW[dataset_type]*np.random.rand(1)[0])
                if Rnd_Num == self.WW[dataset_type]:
                    Rnd_Num = Rnd_Num -1
                
                L_Mix1 = data[dataset_type]['It'+str(It+1)]['DNA']['Latency_List']

                L_Mix  = [L_Mix1[0][Rnd_Num],np.matrix(L_Mix1[0]),np.matrix(L_Mix1[1]),L_Mix1[1][Rnd_Num]]
                Class_R = Routing(self.WW[dataset_type]*self.L,self.L)
                Class_R.EPS = [5*i for i in range(8)]
                beta_ = np.matrix([[1]]*self.WW[dataset_type])
                Ent_List,La_List,Path_P,Lat1    =  Class_R.AL_EXP(L_Mix,False)
                Ent_List2,La_List2,Path_P2,Lat2 =  Class_R.AL_EXP(L_Mix,True)
                #print('2')
                
                if not mod:
                    
                    Ent_List_Ba = []
                    La_List_Ba  = []
                    Ent_List_Ba2 = []
                    La_List_Ba2  = []                    
                    #self.Entropy_Transformation(T,P)
                else:
                    G1G2P = []
                    G1G2P_= []
                    G1G2P2 = []
                    G1G2P_2= []   
                    
                #For CL
                for item_ in Path_P:

                    P_,G1,G2 = compute_conditionals(item_)
                    
                    if mod:
                        G1G2P.append({'P':P_,'G1':G1,'G2':G2,'PP':item_})
                    #print('NOW')   
                    Gamma1 = Gradient_descent_IT(G1,beta_,self.ML_a,self.ML_It)
                    #print('ok1')
                    #print('now')
                    Gamma2 = Gradient_descent_IT(G2,beta_,self.ML_a,self.ML_It)  
                    #print('ok2')
                    if not mod:
                        Ent_List_Ba.append(Class_R.Entropy_Transformation(Gamma1.dot(Gamma2),P_))
                    P_new = P_compute(P_,Gamma1,Gamma2)
                    if mod:
                        G1G2P_.append({'P':P_,'G1':Gamma1,'G2':Gamma2,'PP':P_new})
                    if not mod:
                        La_List_Ba.append(Class_R.Ave_Latency(P_new,Lat1))
                        
                #print('DONNNNNNN')       
                #For CDL       
                for item_ in Path_P2:

                    P_,G1,G2 = compute_conditionals(item_)
                    
                    if mod:
                        G1G2P2.append({'P':P_,'G1':G1,'G2':G2,'PP':item_})
                    #print('NOW444')    
                    Gamma1 = Gradient_descent_IT(G1,beta_,self.ML_a,self.ML_It)
                    Gamma2 = Gradient_descent_IT(G2,beta_,self.ML_a,self.ML_It)  
                    #print('okkkkkkk')
                    if not mod:
                        Ent_List_Ba2.append(Class_R.Entropy_Transformation(Gamma1.dot(Gamma2),P_))
                    P_new = P_compute(P_,Gamma1,Gamma2)
                    if mod:
                        G1G2P_2.append({'P':P_,'G1':Gamma1,'G2':Gamma2,'PP':P_new})
                    if not mod:
                        La_List_Ba2.append(Class_R.Ave_Latency(P_new,Lat2))                       
                        
                #print('Donnnnn22222222222')      
                 
                if not mod:
                    
                    data_0[dataset_type] = {'LC':[[Ent_List,La_List],[Ent_List_Ba,La_List_Ba]],'LCD':[[Ent_List2,La_List2],[Ent_List_Ba2,La_List_Ba2]]}
                else:
                    data_0[dataset_type] = {'LC':[G1G2P,G1G2P_],'LCD':[G1G2P2,G1G2P_2]}
                
            data0['It'+str(It+1)] = data_0
                        
                        
        return data0
   
    
    
    
    
    
    
    
    
    
    
    def PDF_Latencyyy(self,data,mod = False):
        self.ML_a = 0.2
        
        from Routing import Routing
        data0 = {}
        for It in range(len(data['NYM'])):
            data_0 = {}
            
            for dataset_type in self.Data_type:
                #print('1')
                Rnd_Num = int(self.WW[dataset_type]*np.random.rand(1)[0])
                if Rnd_Num == self.WW[dataset_type]:
                    Rnd_Num = Rnd_Num -1
                
                L_Mix1 = data[dataset_type]['It'+str(It+1)]['DNA']['Latency_List']

                L_Mix  = [L_Mix1[0][Rnd_Num],np.matrix(L_Mix1[0]),np.matrix(L_Mix1[1]),L_Mix1[1][Rnd_Num]]
                Class_R = Routing(self.WW[dataset_type]*self.L,self.L)
                Class_R.EPS = [1.5*i for i in range(4)]
                beta_ = np.matrix([[1]]*self.WW[dataset_type])
                Ent_List,La_List,Path_P,Lat1    =  Class_R.AL_EXP(L_Mix,False)
                Ent_List2,La_List2,Path_P2,Lat2 =  Class_R.AL_EXP(L_Mix,True)
                #print('2')
                
                if not mod:
                    
                    Ent_List_Ba = []
                    La_List_Ba  = []
                    Ent_List_Ba2 = []
                    La_List_Ba2  = []                    
                    #self.Entropy_Transformation(T,P)
                else:
                    G1G2P = []
                    G1G2P_= []
                    G1G2P2 = []
                    G1G2P_2= []   
                    
                #For CL
                for item_ in Path_P:

                    P_,G1,G2 = compute_conditionals(item_)
                    
                    if mod:
                        G1G2P.append({'P':P_,'G1':G1,'G2':G2,'PP':item_})
                    #print('NOW')   
                    Gamma1 = Gradient_descent_IT(G1,beta_,self.ML_a,self.ML_It)
                    #print('ok1')
                    #print('now')
                    Gamma2 = Gradient_descent_IT(G2,beta_,self.ML_a,self.ML_It)  
                    #print('ok2')
                    if not mod:
                        Ent_List_Ba.append(Class_R.Entropy_Transformation(Gamma1.dot(Gamma2),P_))
                    P_new = P_compute(P_,Gamma1,Gamma2)
                    if mod:
                        G1G2P_.append({'P':P_,'G1':Gamma1,'G2':Gamma2,'PP':P_new})
                    if not mod:
                        La_List_Ba.append(Class_R.Ave_Latency(P_new,Lat1))
                        
                #print('DONNNNNNN')       
                #For CDL       
                for item_ in Path_P2:

                    P_,G1,G2 = compute_conditionals(item_)
                    
                    if mod:
                        G1G2P2.append({'P':P_,'G1':G1,'G2':G2,'PP':item_})
                    #print('NOW444')    
                    Gamma1 = Gradient_descent_IT(G1,beta_,self.ML_a,self.ML_It)
                    Gamma2 = Gradient_descent_IT(G2,beta_,self.ML_a,self.ML_It)  
                    #print('okkkkkkk')
                    if not mod:
                        Ent_List_Ba2.append(Class_R.Entropy_Transformation(Gamma1.dot(Gamma2),P_))
                    P_new = P_compute(P_,Gamma1,Gamma2)
                    if mod:
                        G1G2P_2.append({'P':P_,'G1':Gamma1,'G2':Gamma2,'PP':P_new})
                    if not mod:
                        La_List_Ba2.append(Class_R.Ave_Latency(P_new,Lat2))                       
                        
                #print('Donnnnn22222222222')      
                 
                if not mod:
                    
                    data_0[dataset_type] = {'LC':[[Ent_List,La_List],[Ent_List_Ba,La_List_Ba]],'LCD':[[Ent_List2,La_List2],[Ent_List_Ba2,La_List_Ba2]]}
                else:
                    data_0[dataset_type] = {'LC':[G1G2P,G1G2P_],'LCD':[G1G2P2,G1G2P_2]}
                
            data0['It'+str(It+1)] = data_0
                        
                        
        return data0
      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def PDF_Reliability(self,data,mod = False):
        
        from Routing import Routing
        data0 = {}
        for It in range(len(data['NYM'])):
            data_0 = {}
            
            for dataset_type in self.Data_type:
                #data_1 = {}
                Class_R = Routing(self.WW[dataset_type]*self.L,self.L)
                Class_R.EPS = [5*i for i in range(8)]
                data_2 = {}

                O_Mix  = data[dataset_type]['It'+str(It+1)]['DNA']['Omega'] 

                beta_ = np.matrix([[1]]*self.WW[dataset_type])
                Ent_List,La_List,Path_P = Class_R.Band_EXP(O_Mix)

                if not mod:
                    
                    Ent_List_Ba = []
                    La_List_Ba  = []
                    #self.Entropy_Transformation(T,P)
                else:
                    G1G2P = []
                    G1G2P_= []
                for item_ in Path_P:

                    P_,G1,G2 = compute_conditionals(item_)
                    
                    if mod:
                        G1G2P.append({'P':P_,'G1':G1,'G2':G2,'PP':item_})
                        
                    Gamma1 = Gradient_descent_IT(G1,beta_,self.ML_a,self.ML_It)
                    Gamma2 = Gradient_descent_IT(G2,beta_,self.ML_a,self.ML_It)  
                    if not mod:
                        Ent_List_Ba.append(Class_R.Entropy_Transformation(Gamma1.dot(Gamma2),P_))
                    P_new = P_compute(P_,Gamma1,Gamma2)
                    if mod:
                        G1G2P_.append({'P':P_,'G1':Gamma1,'G2':Gamma2,'PP':P_new})
                    if not mod:
                        La_List_Ba.append(Class_R.Path_2_KLD(P_new,O_Mix,self.L))
                 
                if not mod:
                    
                    data_0[dataset_type] = [[Ent_List,La_List],[Ent_List_Ba,La_List_Ba]]
                else:
                    data_0[dataset_type] = [G1G2P,G1G2P_]
                
            data0['It'+str(It+1)] = data_0
                        
                        
        return data0














    def PDF_Reliabilityyy(self,data,mod = False):
        
        from Routing import Routing
        data0 = {}
        for It in range(len(data['NYM'])):
            data_0 = {}
            
            for dataset_type in self.Data_type:
                #data_1 = {}
                Class_R = Routing(self.WW[dataset_type]*self.L,self.L)
                Class_R.EPS = [1.5*i for i in range(4)]

                data_2 = {}

                O_Mix  = data[dataset_type]['It'+str(It+1)]['DNA']['Omega'] 

                beta_ = np.matrix([[1]]*self.WW[dataset_type])
                Ent_List,La_List,Path_P = Class_R.Band_EXP(O_Mix)

                if not mod:
                    
                    Ent_List_Ba = []
                    La_List_Ba  = []
                    #self.Entropy_Transformation(T,P)
                else:
                    G1G2P = []
                    G1G2P_= []
                for item_ in Path_P:

                    P_,G1,G2 = compute_conditionals(item_)
                    
                    if mod:
                        G1G2P.append({'P':P_,'G1':G1,'G2':G2,'PP':item_})
                        
                    Gamma1 = Gradient_descent_IT(G1,beta_,self.ML_a,self.ML_It)
                    Gamma2 = Gradient_descent_IT(G2,beta_,self.ML_a,self.ML_It)  
                    if not mod:
                        Ent_List_Ba.append(Class_R.Entropy_Transformation(Gamma1.dot(Gamma2),P_))
                    P_new = P_compute(P_,Gamma1,Gamma2)
                    if mod:
                        G1G2P_.append({'P':P_,'G1':Gamma1,'G2':Gamma2,'PP':P_new})
                    if not mod:
                        La_List_Ba.append(Class_R.Path_2_KLD(P_new,O_Mix,self.L))
                 
                if not mod:
                    
                    data_0[dataset_type] = [[Ent_List,La_List],[Ent_List_Ba,La_List_Ba]]
                else:
                    data_0[dataset_type] = [G1G2P,G1G2P_]
                
            data0['It'+str(It+1)] = data_0
                        
                        
        return data0



















    def PDF_JAR(self,data,mod = False):
        from Routing import Routing
        data0 = {}
        for It in range(len(data['NYM'])):
            data_0 = {}
            
            for dataset_type in self.Data_type:
                #data_1 = {}
                Class_R = Routing(self.WW[dataset_type]*self.L,self.L)
                Class_R.EPS = [5*i for i in range(8)]        
                data_2 = {}

                JAR_List = data[dataset_type]['It'+str(It+1)]

#Gradient_descent_IT
                beta_ = np.matrix([[1]]*self.WW[dataset_type])
                Ent_List,Path_P = Class_R.JAR_EXP(JAR_List)
                if not mod:
                    
                    Ent_List_Ba = []
                    JAR_Regions_Ba = []
                    #self.Entropy_Transformation(T,P)
                    JAR_Regions_Ave = []
                else:
                    G1G2P = []
                    G1G2P_= []
                for item_ in Path_P:
                    if not mod:
                        JAR_Regions_Ave.append(sum(a * b for a, b in zip(item_,JAR_List)))
                    P_,G1,G2 = compute_conditionals(item_)
                    if mod:
                        G1G2P.append({'P':P_,'G1':G1,'G2':G2,'PP':item_})
                    Gamma1 = Gradient_descent_IT(G1,beta_,self.ML_a,self.ML_It)
                    Gamma2 = Gradient_descent_IT(G2,beta_,self.ML_a,self.ML_It)  
                    if not mod:
                        Ent_List_Ba.append(Class_R.Entropy_Transformation(Gamma1.dot(Gamma2),P_))
                    P_new = P_compute(P_,Gamma1,Gamma2)
                    if mod:
                        G1G2P_.append({'P':P_,'G1':Gamma1,'G2':Gamma2,'PP':P_new})
                    if not mod:
                        JAR_Regions_Ba.append(sum(a * b for a, b in zip(P_new,JAR_List)))
                 
                if not mod:
                    
                    data_0[dataset_type] = [[Ent_List,JAR_Regions_Ave],[Ent_List_Ba,JAR_Regions_Ba]]
                else:
                    data_0[dataset_type] = [G1G2P,G1G2P_]
                
            data0['It'+str(It+1)] = data_0
                        
                        
        return data0
    
    
    
    
    
    
    
    
    
    def PDF_JARRR(self,data,mod = False):
        from Routing import Routing
        data0 = {}
        for It in range(len(data['NYM'])):
            data_0 = {}
            
            for dataset_type in self.Data_type:
                #data_1 = {}
                Class_R = Routing(self.WW[dataset_type]*self.L,self.L)
                Class_R.EPS = [1.5*i for i in range(4)]       
                data_2 = {}

                JAR_List = data[dataset_type]['It'+str(It+1)]

#Gradient_descent_IT
                beta_ = np.matrix([[1]]*self.WW[dataset_type])
                Ent_List,Path_P = Class_R.JAR_EXP(JAR_List)
                if not mod:
                    
                    Ent_List_Ba = []
                    JAR_Regions_Ba = []
                    #self.Entropy_Transformation(T,P)
                    JAR_Regions_Ave = []
                else:
                    G1G2P = []
                    G1G2P_= []
                for item_ in Path_P:
                    if not mod:
                        JAR_Regions_Ave.append(sum(a * b for a, b in zip(item_,JAR_List)))
                    P_,G1,G2 = compute_conditionals(item_)
                    if mod:
                        G1G2P.append({'P':P_,'G1':G1,'G2':G2,'PP':item_})
                    Gamma1 = Gradient_descent_IT(G1,beta_,self.ML_a,self.ML_It)
                    Gamma2 = Gradient_descent_IT(G2,beta_,self.ML_a,self.ML_It)  
                    if not mod:
                        Ent_List_Ba.append(Class_R.Entropy_Transformation(Gamma1.dot(Gamma2),P_))
                    P_new = P_compute(P_,Gamma1,Gamma2)
                    if mod:
                        G1G2P_.append({'P':P_,'G1':Gamma1,'G2':Gamma2,'PP':P_new})
                    if not mod:
                        JAR_Regions_Ba.append(sum(a * b for a, b in zip(P_new,JAR_List)))
                 
                if not mod:
                    
                    data_0[dataset_type] = [[Ent_List,JAR_Regions_Ave],[Ent_List_Ba,JAR_Regions_Ba]]
                else:
                    data_0[dataset_type] = [G1G2P,G1G2P_]
                
            data0['It'+str(It+1)] = data_0
                        
                        
        return data0    
    
    
    
    

    def Basic_Analysis(self,File_Path=''):
        import numpy as np
        import pickle
        from Routing import Routing
        import json
        data0 = self.PDFs(self.Data_Set_General)
        
        Iterations = len(data0)
        data3 = {}
        for typ in self.Data_type:
            
            Class_R = Routing((self.WW[typ]*self.L),self.L)
            data1 = {}

            for mtd in self.Method:
                
                List_1 = []
                List_2 = []
                
                for It in range(Iterations):
                    
                    List_1.append(data0['It'+str(It+1)][typ][mtd][0])
                    List_2.append(data0['It'+str(It+1)][typ][mtd][1]) 
                x1 = Medd(To_list(np.transpose(np.matrix(List_1))))
                y1 = Medd(To_list(np.transpose(np.matrix(List_2))))
                data1[mtd] = [x1,y1]
            data3[typ] = data1

            
            
        with open(File_Path + '/Basic_EXP.pkl','wb') as file:

            pickle.dump(data3, file)          
            
        return data3
                         

    def Basic_JAR(self,Iterations,File_Path=''):
        import numpy as np
        import pickle
        from Routing import Routing
        import json
        data_0 = self.JAR_data(Iterations) 
        data0 = self.PDF_JAR(data_0)

        data3 = {}
        data4 = {}
        for typ in self.Data_type:
            List_1 = []
            List_2 = []   
            List_3 = []
            List_4 = []
            for It in range(Iterations):
                
                List_1.append(data0['It'+str(It+1)][typ][0][0])
                List_2.append(data0['It'+str(It+1)][typ][0][1]) 
                List_3.append(data0['It'+str(It+1)][typ][1][0])
                List_4.append(data0['It'+str(It+1)][typ][1][1])                 
            x1 = (To_list(np.transpose(np.matrix(List_1))))
            y1 = (To_list(np.transpose(np.matrix(List_2))))
            x2 = (To_list(np.transpose(np.matrix(List_3))))
            y2 = (To_list(np.transpose(np.matrix(List_4))))           
            
            data3[typ] = [[Medd(x1),Medd(y1)],[Medd(x2),Medd(y2)]]

            
            
        with open(File_Path + '/Basic_EXP.pkl','wb') as file:

            pickle.dump(data3, file)          
            
        return data3
    
    

    def Basic_Reliability(self,Iterations):
        import numpy as np
        import pickle
        from Routing import Routing
        import json
        data_0 =  self.Data_Set_General
        data0 = self.PDF_Reliability(data_0)

        data3 = {}
        data4 = {}
        for typ in self.Data_type:
            List_1 = []
            List_2 = []   
            List_3 = []
            List_4 = []
            for It in range(Iterations):
                
                List_1.append(data0['It'+str(It+1)][typ][0][0])
                List_2.append(data0['It'+str(It+1)][typ][0][1]) 
                List_3.append(data0['It'+str(It+1)][typ][1][0])
                List_4.append(data0['It'+str(It+1)][typ][1][1])                 
            x1 = (To_list(np.transpose(np.matrix(List_1))))
            y1 = (To_list(np.transpose(np.matrix(List_2))))
            x2 = (To_list(np.transpose(np.matrix(List_3))))
            y2 = (To_list(np.transpose(np.matrix(List_4))))           
            
            data3[typ] = [[Medd(x1),Medd(y1)],[Medd(x2),Medd(y2)]]

            
            
       
            
        return data3
    
    
    
    def Basic_Latency(self,Iterations):
        import numpy as np
        import pickle
        from Routing import Routing
        import json
        data_0 =  self.Data_Set_General
        data0 = self.PDF_Latency(data_0)

        data3 = {}
        data4 = {}
        for typ in self.Data_type:
            List_1 = []
            List_2 = []   
            List_3 = []
            List_4 = []
            
            List_10 = []
            List_20 = []   
            List_30 = []
            List_40 = []            
            for It in range(Iterations):
                
                List_1.append(data0['It'+str(It+1)][typ]['LC'][0][0])
                List_2.append(data0['It'+str(It+1)][typ]['LC'][0][1]) 
                List_3.append(data0['It'+str(It+1)][typ]['LC'][1][0])
                List_4.append(data0['It'+str(It+1)][typ]['LC'][1][1])  
                
                List_10.append(data0['It'+str(It+1)][typ]['LCD'][0][0])
                List_20.append(data0['It'+str(It+1)][typ]['LCD'][0][1]) 
                List_30.append(data0['It'+str(It+1)][typ]['LCD'][1][0])
                List_40.append(data0['It'+str(It+1)][typ]['LCD'][1][1])                 
                
            x1 = (To_list(np.transpose(np.matrix(List_1))))
            y1 = (To_list(np.transpose(np.matrix(List_2))))
            x2 = (To_list(np.transpose(np.matrix(List_3))))
            y2 = (To_list(np.transpose(np.matrix(List_4))))           

            xx1 = (To_list(np.transpose(np.matrix(List_10))))
            yy1 = (To_list(np.transpose(np.matrix(List_20))))
            xx2 = (To_list(np.transpose(np.matrix(List_30))))
            yy2 = (To_list(np.transpose(np.matrix(List_40))))    
            
            data3[typ] = {'LC':[[Medd(x1),Medd(y1)],[Medd(x2),Medd(y2)]], 'LCD':[[Medd(xx1),Medd(yy1)],[Medd(xx2),Medd(yy2)]]}

                    
            
        return data3
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def Sim(self,List_L,List_R,P,nn,W,Corrupted_Mix):
        
        Mix_dict = {'Routing':List_R,'Latency':List_L,'First':P}
        
        
        
        from Sim import Simulation
        
        
        Sim_ = Simulation(self.Targets,self.run,self.delay1,self.delay2,W*self.L,self.L )
        
        Latency_Sim,Entropy_Sim = Sim_.Simulator(Corrupted_Mix,Mix_dict,nn)
        
        
        return Latency_Sim, Entropy_Sim                                                        
                    
  

    def Test_Basic(self,Method,Tau):
        import numpy as np
        import pickle
        from Routings import Routing
        from Sim import Simulation
        import json
        data0 = self.PDFs(self.Data_Set_General)
        
        Iterations = len(data0)
        data3 = {}
        for typ in ['NYM']:
            
            Class_R = Routing((self.WW[typ]*self.L),self.L)
            data2 = {}
            for des in ['DNA']:
                
                data1 = {}
                for mtd in Method:
                    
                    
                    if not mtd == 'RST':
                       
                        L_0 = []
                        H_0 = []
                        W_0 = []
                        LB_0 = []
                        HB_0 = []
                        WB_0 = []                         
                        for tau in Tau:
                            
                            L_1 = []
                            H_1 = []
                            W_1 = []
                            LB_1 = []
                            HB_1 = []
                            WB_1 = []                             
                            for It in range(Iterations):
                                
                                
                                O_Mix = self.Data_Set_General[typ]['It'+str(It+1)][des]['Omega'] 
                                P = dist_List(O_Mix[0])
                                datum = data0['It'+str(It+1)][typ][des][mtd]['tau'+str(int(10*tau))]
                                
                                L1=[np.matrix(datum[0][i][0]) for i in range(self.L-1)]
                                R1 =[datum[0][i][1] for i in range(self.L-1)]
                                RB1 =[datum[1][i][1] for i in range(self.L-1)]
                                
                                L_1.append(Class_R.Latency_Measure(L1, R1, P))
                                H_1.append(Class_R.Entropy_AVE(Class_R.Entropy_Transformation(R1),P))
                                W_1.append(Class_R.Bandwidth(R1, O_Mix, P))
                                
                                LB_1.append(Class_R.Latency_Measure(L1, RB1, P))
                                HB_1.append(Class_R.Entropy_AVE(Class_R.Entropy_Transformation(RB1),P))
                                WB_1.append(Class_R.Bandwidth(RB1, O_Mix, P))
                            L_0.append(Medd([L_1]))
                            H_0.append(Medd([H_1]))
                            W_0.append(Medd(To_list(np.transpose(np.matrix(W_1)))))
                            LB_0.append(Medd([LB_1]))
                            HB_0.append(Medd([HB_1]))
                            WB_0.append(Medd(To_list(np.transpose(np.matrix(WB_1)))))    
                        data1[mtd] = {'L':L_0,'LB':LB_0,'H':H_0,'HB':HB_0,'Band':W_0,'Band_B':WB_0}
            
        return data1




    def E2E_Analysis(self,e2e_limit,method,dataset_type,Iterations,T = False,File_Path=''):
        import numpy as np
        import pickle
        from Routings import Routing
        from Sim import Simulation
        import json
        self.alpha0 = [i/10 for i in range(11)]
        self.alpha0[0] = 0.082

        if T:
            self.alpha0 = [2,7,12,18,25,32,38,44,50,60,70] 
        design = 'DNA'
        data = self.Data_Set_General
        data0 = {}
        Class_R = Routing(self.WW[dataset_type]*self.L,self.L)
        L_0 = []
        H_0 = []
        W_0 = []
        LB_0 = []
        HB_0 = []
        WB_0 = []   
        SL_0 = []
        SH_0 = []
        SLB_0 = []
        SHB_0 = []        
    
        
        for It in range(Iterations):
            L_Mix = data[dataset_type]['It'+str(It+1)][design]['Latency_List'] 
            O_Mix = data[dataset_type]['It'+str(It+1)][design]['Omega']  
            P = dist_List(O_Mix[0])  
            L_1 = []
            H_1 = []
            W_1 = []
            LB_1 = []
            HB_1 = []
            WB_1 = []   
            SL_1 = []
            SH_1 = []
            SLB_1 = []
            SHB_1 = []
            
            for tau in self.alpha0:      
                if not T:
                    
                    
                    if not method == 'RST':
                        
                        List_R = [[L_Mix[j],Class_R.Matrix_routing(method,np.matrix(L_Mix[j]),O_Mix[j+1],tau)] for j in range(self.L-1)]
                    else:
                        List_R = [[L_Mix[j],Class_R.Matrix_routing(method,np.matrix(L_Mix[j]),O_Mix[j+1],(tau,self.RST_T))] for j in range(self.L-1)]
                else:
                    List_R = [[L_Mix[j],Class_R.Matrix_routing(method,np.matrix(L_Mix[j]),O_Mix[j+1],(self.RST_tau,tau))] for j in range(self.L-1)]
                    
                List_B = [[L_Mix[j],Class_R.BALD(List_R[j][1],O_Mix[j+1],O_Mix[j])] for j in range(self.L-1)]
                
                L11 = [np.matrix(L_Mix[j]) for j in range(self.L-1) ]
                R11 = [List_R[j][1] for j in range(self.L-1)]
                RB11 = [List_B[j][1] for j in range(self.L-1)]                
                
                
                Rouitng_Latency0 = Class_R.Latency_Measure(L11, R11, P)
                L_1.append(Rouitng_Latency0)
                H_1.append(Class_R.Entropy_AVE(Class_R.Entropy_Transformation(R11),P))
                W_1.append(find_median_from_cdf(Class_R.Bandwidth(R11, O_Mix, P))*10)
                Rouitng_Latency1 = Class_R.Latency_Measure(L11, RB11, P)
                LB_1.append(Rouitng_Latency1)
                HB_1.append(Class_R.Entropy_AVE(Class_R.Entropy_Transformation(RB11),P))
                WB_1.append(find_median_from_cdf(Class_R.Bandwidth(RB11, O_Mix, P))*10)

                    

                
                L1=[L_Mix[i] for i in range(self.L-1)]
                R1 =[To_list(List_R[i][1]) for i in range(self.L-1)]
                RB1 =[To_list(List_B[i][1]) for i in range(self.L-1)]
                
                self.delay1 = (e2e_limit - Rouitng_Latency0)/self.L
                #print(self.delay1)
                Latency_Sim0,Entropy_Sim0 = self.Sim(L1,R1,P,self.nn,self.WW[dataset_type],self.Corrupted_Mix[self.WW[dataset_type]])
                self.dealy1 = (e2e_limit - Rouitng_Latency1)/self.L
                LatencyB_Sim0,EntropyB_Sim0 = self.Sim(L1,RB1,P,self.nn,self.WW[dataset_type],self.Corrupted_Mix[self.WW[dataset_type]])
                
                SL_1.append(np.mean(Latency_Sim0))
                SH_1.append(np.mean(Entropy_Sim0))

                SLB_1.append(np.mean(LatencyB_Sim0))
                SHB_1.append(np.mean(EntropyB_Sim0)) 
                
                
            L_0.append(L_1)
            H_0.append(H_1)
            W_0.append(W_1)
            LB_0.append(LB_1)
            HB_0.append(HB_1)
            WB_0.append(WB_1)  
            SL_0.append(SL_1)
            SH_0.append(SH_1)
            SLB_0.append(SLB_1)
            SHB_0.append(SHB_1)   
        
        L_2   = Med(To_list(np.transpose(np.matrix(L_0))))
        H_2   = Med(To_list(np.transpose(np.matrix(H_0))))
        W_2   = Med(To_list(np.transpose(np.matrix(W_0))))
        LB_2  = Med(To_list(np.transpose(np.matrix(LB_0))))
        HB_2  = Med(To_list(np.transpose(np.matrix(HB_0))))
        WB_2  = Med(To_list(np.transpose(np.matrix(WB_0))))   
        SL_2  = Med(To_list(np.transpose(np.matrix(SL_0))))
        SH_2  = Med(To_list(np.transpose(np.matrix(SH_0))))
        SLB_2 = Med(To_list(np.transpose(np.matrix(SLB_0))))
        SHB_2 = Med(To_list(np.transpose(np.matrix(SHB_0))))  
 

    
        data0['Imbalance'] = {'A_L':L_2,'A_H':H_2,'W':W_2,'S_L':SL_2,'S_H':SH_2}
        data0['Balance'] = {'A_L':LB_2,'A_H':HB_2,'W':WB_2,'S_L':SLB_2,'S_H':SHB_2}
        
        with open(File_Path +'/FCP_EXP.pkl','wb') as file:
            pickle.dump(data0, file)             
        return data0
                              
                    

    def FCP(self,R_List,P,List_C,W,TYPE = False):
        #print(List_C)
        R1 = np.matrix(R_List[0])
        R2 = np.matrix(R_List[1]) 

        
        if not TYPE:
            List = []
            
            for i in range(self.L):
                
                List_ = []
                for item in List_C:
                    
                    if W*i <= item < W*(i+1):
                        List_.append(item-W*i)
                List.append(List_)
        else:
            List = List_C
            


        Path_C  = 0
        for i in (List[0]):
            for j in (List[1]):
                for k in (List[2]):
                    
                    Path_C += P[i]*R1[i,j]*R2[j,k]

        if Path_C>1:
            pass

        return Path_C

    def C_Mix(self,L_M,K,Max_Omega,beta,Transformed_beta):
        Transformed_beta = beta
        #print(beta,Transformed_beta)
        A = permutation_matrix(beta,Transformed_beta)
        #print(beta)
        #print(Transformed_beta)
        N = len(L_M)

        
        List_c1_ = Greedy(L_M,Max_Omega,beta)
        temp = [0]*len(beta)
        for i in List_c1_:
            temp[i] = 1      
        List_c11 = To_list(np.matrix(temp).dot(A))
        List_c1 = []
        for i  in range(len(List_c11)):
            if int(List_c11[i]) == 1:
                List_c1.append(i)
        
        Sim_c1 = Corruption_c(List_c1,N)

        List_c2_ = Random(Max_Omega,beta)
        temp = [0]*len(beta)
        for i in List_c2_:
            temp[i] = 1      
        List_c22 = To_list(np.matrix(temp).dot(A))
        List_c2 = []
        for i  in range(len(List_c22)):
            if int(List_c22[i]) == 1:
                List_c2.append(i)
        Sim_c2 = Corruption_c(List_c2,N)
        
        data0 = {'LP_R':[L_M,Max_Omega,Transformed_beta],'LP_S': None,'G_R':List_c1,'G_S':Sim_c1,'R_R':List_c2,'R_S':Sim_c2}
        return data0
    
    
    

        
    
    def FCP_Analysis(self,Iterations,K,File_Path=''):
        
        data0 = self.data_FCP(Iterations)
        data1 = self.PDFs1(data0)
        
        data_cc = {}
        for typ in self.Data_type:
            for It in range(Iterations):
                datum_ = data0[typ]['It'+str(It+1)]
                data_c = self.C_Mix(datum_['L_M'],K,self.WW[typ]*self.CF*self.L,datum_['beta'][0],datum_['beta'][1]) 
                data_cc[typ+'It'+str(It+1)] = data_c
        
        data_0 = {}
        for typ in self.Data_type:
            data_1 = {}
            #Class_R = Routing(self.WW[typ]*self.L,self.L)

            for mtd in self.Method:
                F_LP_0 = [] 
                F_G_0 = []                           
                F_R_0 = []                         
                for eps_i in range(len(self.EPS)):
                    F_LP_1 = []  
                    F_G_1 = []              
                    F_R_1 = []

                    for It in range(Iterations):
                        data_c = data_cc[typ+'It'+str(It+1)]                           
                        O_Mix_ = np.matrix(data0[typ]['It'+str(It+1)]['Omega'] )
                        O_Mix = To_list(O_Mix_)
                        P     = data1['It'+str(It+1)][typ][mtd][eps_i][1]
                        datum = data1['It'+str(It+1)][typ][mtd][eps_i][0]
    
                        R1    = datum
                        
                        R1_   = [np.matrix(datum[i]) for i in range(self.L-1)]

                        
                        #Greedy_For_Fairness
    
                        List_C_Mix_LP_Im = Greedy_For_Fairness(self.CF*self.L*self.WW[typ],O_Mix,R1_,self.L)
                        O_Mix = To_list(O_Mix_)


                        F_LP_1.append(self.FCP(R1,P,List_C_Mix_LP_Im,self.WW[typ],True))
                            
                        F_G_1.append(self.FCP(R1,P,data_c['G_R'],self.WW[typ]))

                        F_R_1.append(self.FCP(R1,P,data_c['R_R'],self.WW[typ]))
                    F_LP_0.append(Medd([F_LP_1])[0])
                    F_G_0.append(Medd([F_G_1])[0])                        
                    F_R_0.append(Medd([F_R_1])[0]) 
                data_1[mtd] = {'F_LP':F_LP_0,'F_G':F_G_0,'F_R':F_R_0}
            data_0[typ] = data_1



            
            
        with open(File_Path +'/FCP_EXP.pkl','wb') as file:

            pickle.dump(data_0, file)          
            
        return data_0
                         


    def FCP_Analysis_B(self,Iterations,K,File_Path=''):
        self.EPS = [3]
        eps_i = 0
        self.cc_today = [0.2,0.25,0.3,0.4,0.5]
        data0 = self.data_FCP(Iterations)
        data1 = self.PDFs1(data0)
        
        data_cc = {}
        for typ in self.Data_type:
            for cff in self.cc_today:
                self.CF = cff
                for It in range(Iterations):
                    datum_ = data0[typ]['It'+str(It+1)]
                    data_c = self.C_Mix(datum_['L_M'],K,self.WW[typ]*self.CF*self.L,datum_['beta'][0],datum_['beta'][1]) 
                    data_cc[typ+'CF'+str(cff)+'It'+str(It+1)] = data_c
        
        data_0 = {}
        for typ in self.Data_type:
            data_1 = {}
            #Class_R = Routing(self.WW[typ]*self.L,self.L)

            for mtd in self.Method:
                F_LP_0 = [] 
                F_G_0 = []                           
                F_R_0 = []                         
                for cff in self.cc_today:
                    F_LP_1 = []  
                    F_G_1 = []              
                    F_R_1 = []
                    self.CF = cff
                    #print(cff)
                    for It in range(Iterations):
                        data_c = data_cc[typ+'CF'+str(cff)+'It'+str(It+1)]                           
                        O_Mix_ = np.matrix(data0[typ]['It'+str(It+1)]['Omega'] )
                        O_Mix = To_list(O_Mix_)
                        P     = data1['It'+str(It+1)][typ][mtd][eps_i][1]
                        datum = data1['It'+str(It+1)][typ][mtd][eps_i][0]
    
                        R1    = datum
                        
                        R1_   = [np.matrix(datum[i]) for i in range(self.L-1)]

                        
                        #Greedy_For_Fairness
    
                        List_C_Mix_LP_Im = Greedy_For_Fairness(self.CF*self.L*self.WW[typ],O_Mix,R1_,self.L)
                        O_Mix = To_list(O_Mix_)


                        F_LP_1.append(self.FCP(R1,P,List_C_Mix_LP_Im,self.WW[typ],True))
                            
                        F_G_1.append(self.FCP(R1,P,data_c['G_R'],self.WW[typ]))

                        F_R_1.append(self.FCP(R1,P,data_c['R_R'],self.WW[typ]))
                    F_LP_0.append(Medd([F_LP_1])[0])
                    F_G_0.append(Medd([F_G_1])[0])                        
                    F_R_0.append(Medd([F_R_1])[0]) 
                data_1[mtd] = {'F_LP':F_LP_0,'F_G':F_G_0,'F_R':F_R_0}
            data_0[typ] = data_1



            
            
        with open(File_Path +'/FCP_EXP_B.pkl','wb') as file:

            pickle.dump(data_0, file)          
            
        return data_0
                       


    def FCP_Budget(self,Iterations,tau,File_Path =''):
        
        data0 = self.data_FCP(Iterations)
        data1 = self.PDFs_FCP(data0)
        CF = [0.1,0.15,0.2,0.25,0.3,0.35]
        
        data_cc = {}
        for typ in self.Data_type:
            for It in range(Iterations):
                datum_ = data0[typ]['It'+str(It+1)]
                data_c = self.C_Mix(datum_['L_M'],5,self.WW[typ]*self.CF*self.L,datum_['beta'][0],datum_['beta'][1]) 
                data_cc[typ+'It'+str(It+1)] = data_c
        
        data_0 = {}
        for typ in self.Data_type:
            data_1 = {}
            if typ == 'NYM':
                data_Sim = {}
            for mtd in self.Method:
                data_Sim_ = {}
                if not mtd == 'RST':
                    F_LP_0 = []
                    F_LPB_0 = []   
                    F_G_0 = []
                    F_GB_0 = []                           
                    F_R_0 = []
                    F_RB_0 = []                           
                    for cf in CF:
                        self.CF = cf
                        F_LP_1 = []
                        F_LPB_1 = []   
                        F_G_1 = []
                        F_GB_1 = []                           
                        F_R_1 = []
                        F_RB_1 = [] 
                        data_Sim__ = {}
                        for It in range(Iterations):
                            data_c = data_cc[typ+'It'+str(It+1)]                           
                            O_Mix_ = np.matrix(data0[typ]['It'+str(It+1)]['Omega'] )
                            O_Mix = To_list(O_Mix_)
                            P = dist_List(O_Mix[0])
                            datum = data1['It'+str(It+1)][typ][mtd]['tau'+str(int(10*tau))]
    
                            R1 =[datum[0][i] for i in range(self.L-1)]
                            RB1 =[datum[1][i] for i in range(self.L-1)]
                            
                            R1_ =[np.matrix(datum[0][i]) for i in range(self.L-1)]
                            RB1_ =[np.matrix(datum[1][i]) for i in range(self.L-1)]
                            
                            #Greedy_For_Fairness

                            List_C_Mix_LP_Im = Greedy_For_Fairness(self.CF*self.L*self.WW[typ],O_Mix,R1_,self.L)
                            O_Mix = To_list(O_Mix_)
                            #print(len(List_C_Mix_LP_Im),len(List_C_Mix_LP_Im[0]))
                            List_C_Mix_LP_Ba = Greedy_For_Fairness(self.CF*self.L*self.WW[typ],O_Mix,RB1_,self.L) 
                            #print(self.WW[typ] )
                            List_C_Mix_LP =[List_C_Mix_LP_Im[JJ][II] +JJ*self.WW[typ]  for JJ in range(len(List_C_Mix_LP_Im)) for II in range(len(List_C_Mix_LP_Im[JJ]))]
                            
                            data_Sim__['It'+str(It+1)] = Corruption_c(List_C_Mix_LP,self.L*self.WW[typ])
                            
                            #print('tau'+str(tau))
                            #print(self.FCP(R1,P,List_C_Mix_LP,self.WW[typ]))
                            F_LP_1.append(self.FCP(R1,P,List_C_Mix_LP_Im,self.WW[typ],True))
                            
                            F_LPB_1.append(self.FCP(RB1,P,List_C_Mix_LP_Ba,self.WW[typ],True))
    
                            F_G_1.append(self.FCP(R1,P,data_c['G_R'],self.WW[typ]))
                            F_GB_1.append(self.FCP(RB1,P,data_c['G_R'],self.WW[typ]))
                            F_R_1.append(self.FCP(R1,P,data_c['R_R'],self.WW[typ]))
                            F_RB_1.append(self.FCP(RB1,P,data_c['R_R'],self.WW[typ]))

        
                        F_LP_0.append(Medd([F_LP_1])[0])
                        F_LPB_0.append(Medd([F_LPB_1])[0])  
                        F_G_0.append(Medd([F_G_1])[0])
                        F_GB_0.append(Medd([F_GB_1])[0])                         
                        F_R_0.append(Medd([F_R_1])[0])
                        F_RB_0.append(Medd([F_RB_1])[0])  
                        data_Sim_['tau'+str(int(10*tau))] = data_Sim__
                    data_1[mtd] = {'F_LP':F_LP_0,'F_LPB':F_LPB_0,'F_G':F_G_0,'F_GB':F_GB_0,'F_R':F_R_0,'F_RB':F_RB_0}
                else:
                            
                    F_LP_0 = []
                    F_LPB_0 = []   
                    F_G_0 = []
                    F_GB_0 = []                           
                    F_R_0 = []
                    F_RB_0 = []                           
                    TF_LP_0 = []
                    TF_LPB_0 = []   
                    TF_G_0 = []
                    TF_GB_0 = []                           
                    TF_R_0 = []
                    TF_RB_0 = []                           
                          
                    for cf in CF:
                        self.CF = cf
                        F_LP_1 = []
                        F_LPB_1 = []   
                        F_G_1 = []
                        F_GB_1 = []                           
                        F_R_1 = []
                        F_RB_1 = [] 
                        data_Sim__ = {}
                           
                        for It in range(Iterations):
                            data_c = data_cc[typ+'It'+str(It+1)]                           
                            O_Mix_ = np.matrix(data0[typ]['It'+str(It+1)]['Omega']) 
                            O_Mix = To_list(O_Mix_)
                            P = dist_List(O_Mix[0])
                            datum = data1['It'+str(It+1)][typ][mtd]['tau'+str(int(10*tau))]
    
                            R1 =[datum[0][i] for i in range(self.L-1)]
                            RB1 =[datum[1][i] for i in range(self.L-1)]
                            R1_ =[np.matrix(datum[0][i]) for i in range(self.L-1)]
                            RB1_ =[np.matrix(datum[1][i]) for i in range(self.L-1)]                            
                            #Greedy_For_Fairness

                            List_C_Mix_LP_Im = Greedy_For_Fairness(self.CF*self.L*self.WW[typ],O_Mix,R1_,self.L)
                            O_Mix = To_list(O_Mix_)
                            List_C_Mix_LP_Ba = Greedy_For_Fairness(self.CF*self.L*self.WW[typ],O_Mix,RB1_,self.L) 
                            
                            List_C_Mix_LP =[List_C_Mix_LP_Im[JJ][II] +JJ*self.WW[typ]  for JJ in range(len(List_C_Mix_LP_Im)) for II in range(len(List_C_Mix_LP_Im[JJ]))]
                            
                            data_Sim__['It'+str(It+1)] = Corruption_c(List_C_Mix_LP,self.L*self.WW[typ])
                            
                            #print('tau'+str(tau))
                            #print(self.FCP(R1,P,List_C_Mix_LP,self.WW[typ]))
                            F_LP_1.append(self.FCP(R1,P,List_C_Mix_LP_Im,self.WW[typ],True))
                            
                            F_LPB_1.append(self.FCP(RB1,P,List_C_Mix_LP_Ba,self.WW[typ],True))                            

    
    
                            F_G_1.append(self.FCP(R1,P,data_c['G_R'],self.WW[typ]))
                            F_GB_1.append(self.FCP(RB1,P,data_c['G_R'],self.WW[typ]))
                            F_R_1.append(self.FCP(R1,P,data_c['R_R'],self.WW[typ]))
                            F_RB_1.append(self.FCP(RB1,P,data_c['R_R'],self.WW[typ]))


                        F_LP_0.append(Medd([F_LP_1])[0])
                        F_LPB_0.append(Medd([F_LPB_1])[0])  
                        F_G_0.append(Medd([F_G_1])[0])
                        F_GB_0.append(Medd([F_GB_1])[0])                         
                        F_R_0.append(Medd([F_R_1])[0])
                        F_RB_0.append(Medd([F_RB_1])[0])                                 
                        data_Sim_['tau'+str(int(10*tau))] = data_Sim__
      

                            
                            
                            
                            
                            
                    d1 = {'F_LP':F_LP_0,'F_LPB':F_LPB_0,'F_G':F_G_0,'F_GB':F_GB_0,'F_R':F_R_0,'F_RB':F_RB_0}
                    data_1[mtd] = d1
                data_Sim[mtd] = data_Sim_
            data_0[typ] = data_1
                
        


            

            
            
        with open(File_Path+'/Budget_EXP.pkl','wb') as file:

            pickle.dump(data_0, file)          
            
        return data_0,data_1
    
    def sort_and_get_mapping(self,initial_list):
        # Sort the initial list in ascending order and get the sorted indices
        sorted_indices = sorted(range(len(initial_list)), key=lambda x: initial_list[x])
        sorted_list = [initial_list[i] for i in sorted_indices]
    
        # Create a mapping from sorted index to original index
        mapping = {sorted_index: original_index for original_index, sorted_index in enumerate(sorted_indices)}
    
        return sorted_list, mapping
    
    def restore_original_list(self,sorted_list, mapping):
        # Create the original list by mapping each element back to its original position
        original_list = [sorted_list[mapping[i]] for i in range(len(sorted_list))]
        
        return original_list
    def LARMIX(self,LIST_,Tau):#We materealize our function for making the trade off
        #In this function just for one sorted distribution
        t = Tau
        A, mapping = self.sort_and_get_mapping(LIST_)
        T = 1-t
    
        import math
        B=[]
        D=[]
    
    
        r = 1
        for i in range(len(A)):
            j = i
            J = (j*(1/(t**(r))))**(1-t)
    
            E = math.exp(-1)
            R = E**J
    
            B.append(R)
            A[i] = A[i]**(-T)
    
            g = A[i]*B[i]
    
            D.append(g)
        n=sum(D)
        for l in range(len(D)):
            D[l]=D[l]/n
        restored_list = self.restore_original_list(D, mapping)
    
        return restored_list
    
    
    def LAMP_MC(self):
        data = self.Data_Set_General        
        data_0 = self.data_FCP(self.Iterations)
        tau = 0.6
        r = 0.015
        data1 = {}
        #print(self.Iterations)
        for typ in self.Data_type:
            #print(typ)
            data2 = {}
            Class_R = Routing((self.WW[typ]*self.L),self.L)
            L_0 = []
            H_0 = []
            W_0 = []
            HM_0 = []
            FCP_0 = []
            for It in range(self.Iterations):
                #print(It)
                L_Mix = data[typ]['It'+str(It+1)]['DNA']['Latency_List'] 
                O_Mix = data[typ]['It'+str(It+1)]['DNA']['Omega'] 
                
                R1 = []
                P = [1/self.WW[typ]]*self.WW[typ]

                for Layer_num in range(self.L-1):
                    R_Mix1 = []
                    for W_num in range(self.WW[typ]):
                        indices = self.filter_matrix_entries(L_Mix[Layer_num][W_num],r)
                        
                        L_temp = remove_elements_by_index(L_Mix[Layer_num][W_num],indices)
                        
                        r_temp = self.LARMIX(L_temp,tau)
                        
                        r_policy = add_elements_by_index(r_temp,indices)
                        #print(r_policy)
                        R_Mix1.append(r_policy)
                        
                    R1.append(np.matrix(R_Mix1))
                L1= L_Mix.copy()
                R11 = [To_list(R1[i]) for i in range(self.L-1)]
                #print(len(R11),len(R11[0]),len(R11[0][0]))
                #print(len(L1),len(L1[0]),len(L1[0][0]))
                Latency_Sim0,Entropy_Sim0 = self.Sim(L1,R11,P,self.nn,self.WW[typ],self.Corrupted_Mix[self.WW[typ]])
                HM_0 = HM_0 + Entropy_Sim0
                L11 = [np.matrix(L1[i]) for i in range(self.L-1)]
                L_0.append(Class_R.Latency_Measure(L11, R1, P))
                H_0.append(Class_R.Entropy_AVE(Class_R.Entropy_Transformation(R1),P))
                #WW_0 = Class_R.Bandwidth(R1, O_Mix, P)
                #W_1 = [abs(WW_0[i]-0.5) for i in range(len(WW_0))]
                #W_0.append(self.CDF[W_1.index(min(W_1))])  
                W_0.append(Class_R.Bandwidth_(R1, O_Mix, P))
                #print(It)
                O_Mix_1 = np.matrix(data_0[typ]['It'+str(It+1)]['Omega'] )
                O_Mix1 = To_list(O_Mix_1)
                List_C = Greedy_For_Fairness(self.CF*self.L*self.WW[typ],O_Mix1,R1,self.L)
                FCP_0.append(self.FCP(R1,P,List_C,self.WW[typ],True))
              
            data2['L'] = np.mean(L_0)
            data2['H'] = np.mean(H_0)
            data2['W'] = np.mean(W_0)
            data2['HM'] = np.mean(HM_0)
            data2['FCP'] = np.mean(FCP_0)
            data1[typ] = data2
                   
                    
        return data1







    def LAMP_SC(self):

        data = self.Data_Set_General        
        data_0 = self.data_FCP(self.Iterations)
        tau = 0.6
        r = 0.015
        data1 = {}
        #print(self.Iterations)
        for typ in self.Data_type:
            elements = [i for i in range(self.WW[typ])]
            #print(typ)
            data2 = {}
            Class_R = Routing((self.WW[typ]*self.L),self.L)
            L_0 = []
            H_0 = []
            W_0 = []
            HM_0 = []
            FCP_0 = []
            for It in range(self.Iterations):
                #print(It)
                L_Mix = data[typ]['It'+str(It+1)]['DNA']['Latency_List'] 
                O_Mix = data[typ]['It'+str(It+1)]['DNA']['Omega'] 
                Latency_SC_Matrix = data[typ]['It'+str(It+1)]['DNA']['Matrix'] 
                SC_P = data[typ]['It'+str(It+1)]['DNA']['Positions'] 
                L_SC_Matrix = SC_Latency(Latency_SC_Matrix,SC_P,self.L)
                Helper = self.filter_SC(L_SC_Matrix,r,self.L)
                
                Routing1 = []
                Routing2 = {}
                for i_ in range(self.WW[typ]):
                    Routing2[str(i_)] = []

                P = [1/self.WW[typ]]*self.WW[typ]
                
                for I in range(len(Helper)):
                    L_temp = remove_elements_by_index(L_Mix[0][I],Helper[I][0])
                    ele_temp = remove_elements_by_index(elements.copy(),Helper[I][0])
                    r_temp = self.LARMIX(L_temp,tau)
                    
                    r_policy = add_elements_by_index(r_temp,Helper[I][0]) 
                    Routing1.append(r_policy)
                    x_list = []
                    for i in ele_temp:
                            
                        L_temp1 = remove_elements_by_index(L_Mix[1][i],Helper[I][1])
                        r_temp1 = self.LARMIX(L_temp1,tau)                            
                        r_policy1 = add_elements_by_index(r_temp1,Helper[I][1]) 
                        Routing2[str(i)].append(r_policy1)
                        #print(r_policy1)
                R_Final = []
                for i_ in range(self.WW[typ]):
                    y = To_list(np.mean(np.matrix(Routing2[str(i_)]),axis=0))
                    
                    if len(y)==0:
                        y = [0]*self.WW[typ]
                        y[0] = 1
                    
                    R_Final.append(y)
                    #if len(y)!= self.WW[typ]:
                        #print(Routing2[str(i_)])
                   
                    
                        
                
                R1 = [np.matrix(Routing1),np.matrix(R_Final)]
                
                L1= L_Mix.copy()
                R11 = [To_list(R1[i]) for i in range(self.L-1)]
                #print(len(R11),len(R11[0]),len(R11[0][0]))
                #print(len(L1),len(L1[0]),len(L1[0][0]))
                Latency_Sim0,Entropy_Sim0 = self.Sim(L1,R11,P,self.nn,self.WW[typ],self.Corrupted_Mix[self.WW[typ]])
                HM_0 = HM_0 + Entropy_Sim0
                L11 = [np.matrix(L1[i]) for i in range(self.L-1)]
                L_0.append(Class_R.Latency_Measure(L11, R1, P))
                H_0.append(Class_R.Entropy_AVE(Class_R.Entropy_Transformation(R1),P))
                #WW_0 = Class_R.Bandwidth(R1, O_Mix, P)
                #W_1 = [abs(WW_0[i]-0.5) for i in range(len(WW_0))]
                #W_0.append(self.CDF[W_1.index(min(W_1))])  
                W_0.append(Class_R.Bandwidth_(R1, O_Mix, P))
                #print(It)
                O_Mix_1 = np.matrix(data_0[typ]['It'+str(It+1)]['Omega'] )
                O_Mix1 = To_list(O_Mix_1)
                List_C = Greedy_For_Fairness(self.CF*self.L*self.WW[typ],O_Mix1,R1,self.L)
                FCP_0.append(self.FCP(R1,P,List_C,self.WW[typ],True))
              
            data2['L'] = np.mean(L_0)
            data2['H'] = np.mean(H_0)
            data2['W'] = np.mean(W_0)
            data2['HM'] = np.mean(HM_0)
            data2['FCP'] = np.mean(FCP_0)
            data1[typ] = data2
                   
                    
        return data1


















    def filter_SC(self,matrix, threshold,L):
        W = int(len(matrix)/L)
        flist = []
        for i in range(W):
            
            a = self.filter_matrix_entries(To_list(matrix[i,W:2*W]),threshold)
            b = self.filter_matrix_entries(To_list(matrix[i,2*W:3*W]),threshold)

            flist.append([a,b])

                
        return flist

      
   
    
    def filter_matrix_entries(self,matrix, threshold):
        flist = []
        for i in range(len(matrix)):
            
            if matrix[i] > threshold:
                
                flist.append(i)

        if len(flist)==len(matrix):
            List = matrix.copy()
            length = round(0.02*len(matrix))
            Indices = []
            for j in range(length):
                
                Index = List.index(min(List))
                Indices.append(Index)
                List[Index] = 100000000000
            out = remove_elements_by_index(flist,Indices)
            
            return out
                
                
        return flist



    def data_interface(self,data0):
        data = []
        
        for item in data0:
            
            lat, lon = self.convert_to_lat_lon(item[0],item[1], item[2])
            #print(lat,lon)
            data.append({'latitude':lat, 'longitude':lon})

        a = [(20,80),(-10,33) ]
        b = [(25,50),(-100,-50)]

        El1 = a[0][0]
        Eu1 = a[0][1]
        
        El2 = a[1][0]
        Eu2 = a[1][1]   
        
        
        Nl1 = b[0][0]
        Nu1 = b[0][1]
        
        Nl2 = b[1][0]
        Nu2 = b[1][1]   
        
        
        Europe = []
        North_America = []
        
        for i in range(len(data)):

            if El1 <data[i]['latitude'] < Eu1 and El2 <data[i]['longitude']< Eu2:
                Europe.append(i)
                
            elif Nl1 <data[i]['latitude'] < Nu1 and Nl2 <data[i]['longitude']< Nu2:
                North_America.append(i) 
        
        return [Europe, North_America]
    
    
    def Regional_Mix(self,Lists,Matrix,Omega):
        data0 = {}
        

        for i in range(len(Lists)):

            List = Lists[i]
            Latency_List = []
            Omega_List = []
            n = int(len(List)/3)*3
            for k1 in range(3):
                list_temp = []
                for k2 in range(int(n/3)):
                    list_temp.append(Omega[List[k1*int(n/3)+k2]])
                Omega_List.append(list_temp)
                    
            for k in range(2):
                Temp_list = []
                for j1 in range(int(n/3)):
                    Temp_list0 = []
                    for j2 in range(int(n/3)):
                        
                        temp = Matrix[List[j1+k*int(n/3)],List[(k+1)*int(n/3)+j2]]
                        Temp_list0.append(temp)
                    Temp_list.append(Temp_list0)
                Latency_List.append(Temp_list)
            
            data0['Region'+str(i+1)] = {'Latency_List':Latency_List,'Omega':Omega_List,'W':int(n/3)}
        return data0

    def convert_to_lat_lon(self,x, y, z):
        import math
        radius = 6371  # Earth's radius in kilometers
    
        # Convert Cartesian coordinates to spherical coordinates
        longitude = math.atan2(y, x)
        hypotenuse = math.sqrt(x**2 + y**2)
        latitude = math.atan2(z, hypotenuse)
    
        # Convert radians to degrees
        latitude = math.degrees(latitude)
        longitude = math.degrees(longitude)
    
        return latitude, longitude
    
    
    def LAMP_RM(self):
        data = self.Data_Set_General        
        tau = 0.6
        data1 = {}
        #print(self.Iterations)
        for typ in self.Data_type:
            #print(typ)
            data2 = {}
            data_2 = {}
            Class_R = Routing((self.WW[typ]*self.L),self.L)
            L_0_E = []
            H_0_E = []
            W_0_E = []
            HM_0_E = []
            FCP_0_E = []
            
            L_0_N = []
            H_0_N = []
            W_0_N = []
            HM_0_N = []
            FCP_0_N = []            
            
            
            
            for It in range(self.Iterations):
                #print(It)
                
                Loc = data[typ]['It'+str(It+1)]['DNA']['Loc']
                Lists_RM = self.data_interface(Loc)
                Matrix_Mix = data[typ]['It'+str(It+1)]['DNA']['Matrix']
                Omega_Mix = data[typ]['It'+str(It+1)]['DNA']['x']
              
                Regions = self.Regional_Mix(Lists_RM,Matrix_Mix,Omega_Mix)
                
                
                
                
                L_Mix = Regions['Region'+str(1)]['Latency_List']
                O_Mix66 = Regions['Region'+str(1)]['Omega'] 
                O_Mix = []
                for item in O_Mix66:
                    O_Mix.append(Norm_List(item,self.WW[typ]))                
                W_region = Regions['Region'+str(1)]['W']
                
                R1 = []
                P = [1/W_region]*W_region

                for Layer_num in range(self.L-1):
                    R_Mix1 = []
                    for W_num in range(W_region):

                        L_temp = L_Mix[Layer_num][W_num]
                        r_policy = self.LARMIX(L_temp,tau)

                        R_Mix1.append(r_policy)
                        
                    R1.append(np.matrix(R_Mix1))
                L1= L_Mix.copy()
                R11 = [To_list(R1[i]) for i in range(self.L-1)]
                C_Mix_Region = {}
                for Mix_index in range(self.L*W_region):
                    C_Mix_Region['PM'+str(Mix_index+1)] = False
                self.delay2 = 0.5*self.delay2*(self.WW[typ]/W_region)
                Latency_Sim0,Entropy_Sim0 = self.Sim(L1,R11,P,self.nn,W_region,C_Mix_Region)
                HM_0_E = HM_0_E + Entropy_Sim0
                
                L11 = [np.matrix(L1[i]) for i in range(self.L-1)]
                L_0_E.append(Class_R.Latency_Measure(L11, R1, P))
                H_0_E.append(Class_R.Entropy_AVE(Class_R.Entropy_Transformation(R1),P))
                W_0_E.append(Class_R.Bandwidth_(R1, O_Mix, P))
                #print(It)
                O_Mix_1 = np.matrix(Omega_Mix )
                O_Mix1 = To_list(O_Mix_1)
                List_C = Greedy_For_Fairness(self.CF*self.L*W_region,O_Mix,R1,self.L)
                FCP_0_E.append(self.FCP(R1,P,List_C,W_region,True))


                L_Mix = Regions['Region'+str(2)]['Latency_List']
                O_Mix66 = Regions['Region'+str(2)]['Omega'] 
                O_Mix = []
                for item in O_Mix66:
                    O_Mix.append(Norm_List(item,self.WW[typ]))                
                
                W_region = Regions['Region'+str(2)]['W']
                
                R1 = []
                P = [1/W_region]*W_region

                for Layer_num in range(self.L-1):
                    R_Mix1 = []
                    for W_num in range(W_region):

                        L_temp = L_Mix[Layer_num][W_num]
                        r_policy = self.LARMIX(L_temp,tau)

                        R_Mix1.append(r_policy)
                        
                    R1.append(np.matrix(R_Mix1))
                L1= L_Mix.copy()
                R11 = [To_list(R1[i]) for i in range(self.L-1)]
                C_Mix_Region = {}
                for Mix_index in range(self.L*W_region):
                    C_Mix_Region['PM'+str(Mix_index+1)] = False
                self.delay2 = 0.5*self.delay2*(self.WW[typ]/W_region)
                Latency_Sim0,Entropy_Sim0 = self.Sim(L1,R11,P,self.nn,W_region,C_Mix_Region)
                HM_0_N = HM_0_N + Entropy_Sim0
                
                L11 = [np.matrix(L1[i]) for i in range(self.L-1)]
                L_0_N.append(Class_R.Latency_Measure(L11, R1, P))
                H_0_N.append(Class_R.Entropy_AVE(Class_R.Entropy_Transformation(R1),P))
                W_0_N.append(Class_R.Bandwidth_(R1, O_Mix, P))
                #print(It)
                O_Mix_1 = np.matrix(Omega_Mix )
                O_Mix1 = To_list(O_Mix_1)
                List_C = Greedy_For_Fairness(self.CF*self.L*W_region,O_Mix,R1,self.L)
                FCP_0_N.append(self.FCP(R1,P,List_C,W_region,True))              
            data2['L'] = np.mean(L_0_E)
            data2['H'] = np.mean(H_0_E)
            data2['W'] = np.mean(W_0_E)
            data2['HM'] = np.mean(HM_0_E)
            data2['FCP'] = np.mean(FCP_0_E)
            data_2['L'] = np.mean(L_0_N)
            data_2['H'] = np.mean(H_0_N)
            data_2['W'] = np.mean(W_0_N)
            data_2['HM'] = np.mean(HM_0_N)
            data_2['FCP'] = np.mean(FCP_0_N)            
            
            data1[typ] = {'EU':data2,'NA':data_2}
                   
                    
        return data1


    def MAP_Latency_Omega(self,Map,Matrix,Omega,L):
        
        N = len(Matrix)
        W = int(N/L)
        
        Latency_List = []
        Omega_List = []
        for i in range(L-1):
            List0 = []
            
            for j in range(W):
                List1 = []
                
                for k in range(W):
                    #if Matrix[Map[i*W+j],Map[(i+1)*W+k]]>2:
                        
                        #print(Matrix[Map[i*W+j],Map[(i+1)*W+k]])
                    List1.append(Matrix[Map[i*W+j],Map[(i+1)*W+k]])
                List0.append(List1)
            Latency_List.append(List0)
                    
        for i in range(L):
            List0 = []
            
            for j in range(W):
                List0.append(Omega[i*W+j])
            Omega_List.append(List0)
                
                
        return Latency_List,Omega_List
                    
        
        
        

    def LARMIX_EXP(self):
        from CLUSTER import Clustering
        from MixNetArrangment import Mix_Arrangements
        from Greedy_LARMIX import Balanced_Layers
        
        data = self.Data_Set_General        
        data_0 = self.data_FCP(self.Iterations)
        tau = 0.6
        data1 = {}
        #print(self.Iterations)
        for typ in self.Data_type:
            #print(typ)
            data2 = {}
            Class_R = Routing((self.WW[typ]*self.L),self.L)
            L_0 = []
            H_0 = []
            W_0 = []
            HM_0 = []
            FCP_0 = []
            for It in range(self.Iterations):
                Loc = data[typ]['It'+str(It+1)]['DNA']['Loc']
                A_Loc = np.matrix([np.random.rand(3) for itr in range(self.WW[typ]*self.L)])
                Loc += A_Loc
                Matrix_Mix = data[typ]['It'+str(It+1)]['DNA']['Matrix']
                Omega_Mix = data[typ]['It'+str(It+1)]['DNA']['x'] 

                    
                ####LARMIx preparations########################################################
                Class_cluster = Clustering(np.copy(Loc),'kmedoids',5,self.L,0)
                New_Loc = Class_cluster.Mixes
                #print(Loc,'*',New_Loc)
                Labels = Class_cluster.Labels
                
                Map = Class_cluster.Map
                #print(Map)
                Class_Div = Mix_Arrangements( np.copy(New_Loc),0, Labels,Class_cluster.Centers,0,1,False)
                Final_Loc_ = To_list(Class_Div.Topology)
                Final_Loc = []
                for item in Final_Loc_:
                    Final_Loc += item
                MAP_ = find_row_permutation(New_Loc,np.matrix(Final_Loc))
                MAP_Final = MAP_to_MAP(Map,MAP_)
                #print(MAP_Final)
                Latency_List_LARMIX, Omega_List_LARMIX = self.MAP_Latency_Omega(MAP_Final,Matrix_Mix,Omega_Mix,self.L)

                
                L_Mix = Latency_List_LARMIX 
                O_Mix66 = Omega_List_LARMIX
                O_Mix = []
                for item in O_Mix66:
                    O_Mix.append(Norm_List(item,self.WW[typ]))
                                    
                R22 = []
                P = [1/self.WW[typ]]*self.WW[typ]

                for Layer_num in range(self.L-1):
                    R_Mix1 = []
                    for W_num in range(self.WW[typ]):
                        r_temp = self.LARMIX(L_Mix[Layer_num][W_num],tau)
                        R_Mix1.append(r_temp)
                    R22.append(np.matrix(R_Mix1))
                    
                R1 = []
                
                for item_ in R22:
                    Class_greedy = Balanced_Layers(5,'IDK',self.WW[typ])
                    Class_greedy.IMD = np.copy(item_)
                    Class_greedy.Iterations()                
                    R1.append(Class_greedy.IMD)
                    
                    
                    
                L1= L_Mix.copy()
                #if typ=='NYM':
                    #print(L1)
                R11 = [To_list(R1[i]) for i in range(self.L-1)]
                #print(len(R11),len(R11[0]),len(R11[0][0]))
                #print(len(L1),len(L1[0]),len(L1[0][0]))
                Latency_Sim0,Entropy_Sim0 = self.Sim(L1,R11,P,self.nn,self.WW[typ],self.Corrupted_Mix[self.WW[typ]])
                HM_0 = HM_0 + Entropy_Sim0
                L11 = [np.matrix(L1[i]) for i in range(self.L-1)]
                L_0.append(Class_R.Latency_Measure(L11, R1, P))
                H_0.append(Class_R.Entropy_AVE(Class_R.Entropy_Transformation(R1),P))
                #WW_0 = Class_R.Bandwidth(R1, O_Mix, P)
                #W_1 = [abs(WW_0[i]-0.5) for i in range(len(WW_0))]
                #W_0.append(self.CDF[W_1.index(min(W_1))])  
                W_0.append(Class_R.Bandwidth_(R1, O_Mix, P))
                #print(It)
                O_Mix_1 = np.matrix(O_Mix)
                O_Mix1 = To_list(O_Mix_1)
                List_C = Greedy_For_Fairness(self.CF*self.L*self.WW[typ],O_Mix1,R1,self.L)
                FCP_0.append(self.FCP(R1,P,List_C,self.WW[typ],True))
              
            data2['L'] = np.mean(L_0)
            data2['H'] = np.mean(H_0)
            data2['W'] = np.mean(W_0)
            data2['HM'] = np.mean(HM_0)
            data2['FCP'] = np.mean(FCP_0)
            data1[typ] = data2
                   
                    
        return data1



    def  Vanilla(self):
        from CLUSTER import Clustering
        from MixNetArrangment import Mix_Arrangements
        from Greedy_LARMIX import Balanced_Layers
        
        data = self.Data_Set_General        
        data_0 = self.data_FCP(self.Iterations)
        tau = 1
        data1 = {}
        #print(self.Iterations)
        for typ in self.Data_type:
            #print(typ)
            data2 = {}
            Class_R = Routing((self.WW[typ]*self.L),self.L)
            L_0 = []
            H_0 = []
            W_0 = []
            HM_0 = []
            FCP_0 = []
            for It in range(self.Iterations):
                Loc = data[typ]['It'+str(It+1)]['DNA']['Loc']
                A_Loc = np.matrix([np.random.rand(3) for itr in range(self.WW[typ]*self.L)])
                Loc += A_Loc
                Matrix_Mix = data[typ]['It'+str(It+1)]['DNA']['Matrix']
                Omega_Mix = data[typ]['It'+str(It+1)]['DNA']['x'] 

                    
                ####LARMIx preparations########################################################
                Class_cluster = Clustering(np.copy(Loc),'kmedoids',5,self.L,0)
                New_Loc = Class_cluster.Mixes
                #print(Loc,'*',New_Loc)
                Labels = Class_cluster.Labels
                
                Map = Class_cluster.Map
                #print(Map)
                Class_Div = Mix_Arrangements( np.copy(New_Loc),0, Labels,Class_cluster.Centers,0,1,False)
                Final_Loc_ = To_list(Class_Div.Topology)
                Final_Loc = []
                for item in Final_Loc_:
                    Final_Loc += item
                MAP_ = find_row_permutation(New_Loc,np.matrix(Final_Loc))
                MAP_Final = MAP_to_MAP(Map,MAP_)
                #print(MAP_Final)
                Latency_List_LARMIX, Omega_List_LARMIX = self.MAP_Latency_Omega(MAP_Final,Matrix_Mix,Omega_Mix,self.L)

                
                L_Mix = Latency_List_LARMIX 
                O_Mix66 = Omega_List_LARMIX
                O_Mix = []
                for item in O_Mix66:
                    O_Mix.append(Norm_List(item,self.WW[typ]))
                                    
                R22 = []
                P = [1/self.WW[typ]]*self.WW[typ]

                for Layer_num in range(self.L-1):
                    R_Mix1 = []
                    for W_num in range(self.WW[typ]):
                        r_temp = self.LARMIX(L_Mix[Layer_num][W_num],tau)
                        R_Mix1.append(r_temp)
                    R22.append(np.matrix(R_Mix1))
                    
                R1 = []
                
                for item_ in R22:
                    Class_greedy = Balanced_Layers(5,'IDK',self.WW[typ])
                    Class_greedy.IMD = np.copy(item_)
                    Class_greedy.Iterations()                
                    R1.append(Class_greedy.IMD)
                    
                    
                    
                L1= L_Mix.copy()
                #if typ=='NYM':
                    #print(L1)
                R11 = [To_list(R1[i]) for i in range(self.L-1)]
                #print(len(R11),len(R11[0]),len(R11[0][0]))
                #print(len(L1),len(L1[0]),len(L1[0][0]))
                Latency_Sim0,Entropy_Sim0 = self.Sim(L1,R11,P,self.nn,self.WW[typ],self.Corrupted_Mix[self.WW[typ]])
                HM_0 = HM_0 + Entropy_Sim0
                L11 = [np.matrix(L1[i]) for i in range(self.L-1)]
                L_0.append(Class_R.Latency_Measure(L11, R1, P))
                H_0.append(Class_R.Entropy_AVE(Class_R.Entropy_Transformation(R1),P))
                #WW_0 = Class_R.Bandwidth(R1, O_Mix, P)
                #W_1 = [abs(WW_0[i]-0.5) for i in range(len(WW_0))]
                #W_0.append(self.CDF[W_1.index(min(W_1))])  
                W_0.append(Class_R.Bandwidth_(R1, O_Mix, P))
                #print(It)
                O_Mix_1 = np.matrix(O_Mix)
                O_Mix1 = To_list(O_Mix_1)
                List_C = Greedy_For_Fairness(self.CF*self.L*self.WW[typ],O_Mix1,R1,self.L)
                FCP_0.append(self.FCP(R1,P,List_C,self.WW[typ],True))
              
            data2['L'] = np.mean(L_0)
            data2['H'] = np.mean(H_0)
            data2['W'] = np.mean(W_0)
            data2['HM'] = np.mean(HM_0)
            data2['FCP'] = np.mean(FCP_0)
            data1[typ] = data2
                   
                    
        return data1
    
    
    
    def PDFs_(self,data):
        from Routing import Routing
        data0 = {}
        #print(len(data['NYM']))
        for It in range(1):
            data_0 = {}
            
            for dataset_type in ['NYM']:
                #data_1 = {}
                Class_R = Routing(self.WW[dataset_type]*self.L,self.L)
                        
                for design in self.Design:
                    data_2 = {}
                    Rnd_Num = int(self.WW[dataset_type]*np.random.rand(1)[0])
                    if Rnd_Num == self.WW[dataset_type]:
                        Rnd_Num = Rnd_Num -1
                    L_Mix1 = data[dataset_type]['It'+str(It+1)][design]['Latency_List']
                    #print(L_Mix1)
                    L_Mix  = [L_Mix1[0][Rnd_Num],np.matrix(L_Mix1[0]),np.matrix(L_Mix1[1]),L_Mix1[1][Rnd_Num]]
                    O_Mix  = data[dataset_type]['It'+str(It+1)][design]['Omega'] 
                    for method in self.Method:
                        if method == 'L_C':
                            Ent_List,La_List = Class_R.AL_EXP(L_Mix,False)
                        elif method =='L_CD':
                            Ent_List,La_List = Class_R.AL_EXP(L_Mix,True)
                        elif method == 'Band':
                            Ent_List,La_List = Class_R.Band_EXP(O_Mix)
                        
                        data_2[method] = [Ent_List,La_List]
                        
                data_0[dataset_type] = data_2
                
            data0['It'+str(It+1)] = data_0
                        
                        
        return data0
                              
    
    
    
    
        
    def Adv_Fun(self, PPP, Budget):
        P = list(PPP)  # Make sure it's a list
        L = self.L     # Should be 3
        W = round(len(P) ** (1 / L))
    
        selected_dims = {0: set(), 1: set(), 2: set()}
        used_indices = set()
        Mix_FCP = 0
        count = 0
    
        # Sort flat indices by descending value
        sorted_indices = sorted(range(len(P)), key=lambda x: P[x], reverse=True)
    
        for flat_index in sorted_indices:
            i = flat_index // (W * W)
            j = (flat_index % (W * W)) // W
            k = flat_index % W
    
            # Count how many new dimensions this triplet touches
            new_dims = sum([
                i not in selected_dims[0],
                j not in selected_dims[1],
                k not in selected_dims[2]
            ])
    
            if count + new_dims > Budget:
                break
    
            # Mark new indices and update budget count
            if i not in selected_dims[0]:
                selected_dims[0].add(i)
                count += 1
            if j not in selected_dims[1]:
                selected_dims[1].add(j)
                count += 1
            if k not in selected_dims[2]:
                selected_dims[2].add(k)
                count += 1
    
            Mix_FCP += P[flat_index]
            used_indices.add(flat_index)
    
        # Compile final list of dimension indices with offset
        List_Index = [i for i in selected_dims[0]] + \
                     [j + W for j in selected_dims[1]] + \
                     [k + 2 * W for k in selected_dims[2]]
    
        return List_Index, Mix_FCP
    
        
                
        
    
    def Adv_Random(self,P,Budget):
        L = self.L
        #print(P)
        
        W = round((len(P))**(1/L))
       # print(L*W,Budget)
        List_Index = pick_m_from_n(L*W, Budget)
        Mix_FCP = 0
        
        for item in List_Index:
            Mix_FCP += P[item]
            
            
        return To_list(List_Index), Mix_FCP        
            
    def Simulation_ADV(self,corrupted_Mix,Mix_dict,nn,W):
        from Sim import Simulation
    
        Class_Sim = Simulation(self.Targets,self.run,self.delay1,self.delay2,self.L*W,self.L)


        _,ENT = Class_Sim.Simulator(corrupted_Mix,Mix_dict,nn)
        
        return ENT
    
    def Corruption_ADV(self,List,N):
        data = {}
        for i in range(N):
            data['PM'+str(1+i)] = False
        for item in List:
            data['PM'+str(item+1)] = True
        return data

    def FCP_JAR_(self,nn,Iterations):
        #print('1')
        import numpy as np
        import pickle
        from Routing import Routing
        import json
        data_0 = self.JAR_data(Iterations)
        data0 = self.PDF_JARRR(data_0,True)
        data_out = {}
        for typ in ['RIPE', 'NYM']:
 
            FCP_R_V0 = []
            FCP_G_V0 = []
            Ent_R_V0 = []
            Ent_G_V0 = []
            
            FCP_R_P0 = []
            FCP_G_P0 = []
            Ent_R_P0 = []
            Ent_G_P0 = [] 
            #for i in [2]:
            for i in range(len(data0['It1']['NYM'][0])):
                
                FCP_R_V = []
                FCP_G_V = []
                Ent_R_V = []
                Ent_G_V = []
                
                FCP_R_P = []
                FCP_G_P = []
                Ent_R_P = []
                Ent_G_P = []                
                
                for It in range(Iterations):
                    #FCP for Vanilla
                    List1 , FCP1 = self.Adv_Random(data0['It'+str(It+1)][typ][0][i]['PP'],int(0.25*self.L*self.WW[typ]))
                    List2 , FCP2 = self.Adv_Fun(data0['It'+str(It+1)][typ][0][i]['PP'],int(0.25*self.L*self.WW[typ]))
                    FCP_R_V.append(FCP1)
                    FCP_G_V.append(FCP2)
                    Mix_dict = {'First':data0['It'+str(It+1)][typ][0][i]['P']}
                    Mix_dict['Routing'] = [To_list(data0['It'+str(It+1)][typ][0][i]['G1']),To_list(data0['It'+str(It+1)][typ][0][i]['G2'])]
                  
                    ent1 = self.Simulation_ADV(self.Corruption_ADV(List1,self.L*self.WW[typ]),Mix_dict,nn,self.WW[typ])                   
                    Ent_R_V += ent1

                    ent2 = self.Simulation_ADV(self.Corruption_ADV(List2,self.L*self.WW[typ]),Mix_dict,nn,self.WW[typ])                   
                    Ent_G_V += ent2 

                    #FCP for PAD
                    List1 , FCP1 = self.Adv_Random(data0['It'+str(It+1)][typ][1][i]['PP'],int(0.25*self.L*self.WW[typ]))
                    List2 , FCP2 = self.Adv_Fun(data0['It'+str(It+1)][typ][1][i]['PP'],int(0.25*self.L*self.WW[typ]))
                    FCP_R_P.append(FCP1)
                    FCP_G_P.append(FCP2)
                    Mix_dict = {'First':data0['It'+str(It+1)][typ][1][i]['P']}
                    Mix_dict['Routing'] = [To_list(data0['It'+str(It+1)][typ][1][i]['G1']),To_list(data0['It'+str(It+1)][typ][1][i]['G2'])]
                  

                    
                FCP_R_V0.append(FCP_R_V)
                FCP_G_V0.append(FCP_G_V)
                Ent_R_V0.append(Ent_R_V)
                Ent_G_V0.append(Ent_G_V)
                
                FCP_R_P0.append(FCP_R_P)
                FCP_G_P0.append(FCP_G_P)

            x1 =  Medd(To_list((np.matrix(FCP_R_V0))))
            x2 =  Medd(To_list((np.matrix(FCP_G_V0))))
            x3 =  Medd(To_list((np.matrix(FCP_R_P0))))
            x4 =  Medd(To_list((np.matrix(FCP_G_P0))))     
            
            data_out[typ] = {'FCP_R_V': x1, 'FCP_R_P': x3, 'FCP_G_V': x2, 'FCP_G_P': x4}
            data_out[typ]['Ent_R_V'] = (To_list((np.matrix(Ent_R_V0))))
            data_out[typ]['Ent_G_V'] = (To_list((np.matrix(Ent_G_V0))))                


   
        return data_out                     
   
 
    
    def FCP_JAR(self,nn,Iterations):
        #print('1')
        import numpy as np
        import pickle
        from Routing import Routing
        import json
        data_0 = self.JAR_data(Iterations)
        data0 = self.PDF_JAR(data_0,True)
        data_out = {}
        for typ in ['RIPE', 'NYM']:
 
            FCP_R_V0 = []
            FCP_G_V0 = []
            Ent_R_V0 = []
            Ent_G_V0 = []
            
            FCP_R_P0 = []
            FCP_G_P0 = []
            Ent_R_P0 = []
            Ent_G_P0 = [] 
            #for i in [2]:
            for i in range(len(data0['It1']['NYM'][0])):
                
                FCP_R_V = []
                FCP_G_V = []
                Ent_R_V = []
                Ent_G_V = []
                
                FCP_R_P = []
                FCP_G_P = []
                Ent_R_P = []
                Ent_G_P = []                
                
                for It in range(Iterations):
                    #FCP for Vanilla
                    List1 , FCP1 = self.Adv_Random(data0['It'+str(It+1)][typ][0][i]['PP'],int(0.25*self.L*self.WW[typ]))
                    List2 , FCP2 = self.Adv_Fun(data0['It'+str(It+1)][typ][0][i]['PP'],int(0.25*self.L*self.WW[typ]))
                    FCP_R_V.append(FCP1)
                    FCP_G_V.append(FCP2)
                    Mix_dict = {'First':data0['It'+str(It+1)][typ][0][i]['P']}
                    Mix_dict['Routing'] = [To_list(data0['It'+str(It+1)][typ][0][i]['G1']),To_list(data0['It'+str(It+1)][typ][0][i]['G2'])]
                  
                    ent1 = self.Simulation_ADV(self.Corruption_ADV(List1,self.L*self.WW[typ]),Mix_dict,nn,self.WW[typ])                   
                    Ent_R_V += ent1

                    ent2 = self.Simulation_ADV(self.Corruption_ADV(List2,self.L*self.WW[typ]),Mix_dict,nn,self.WW[typ])                   
                    Ent_G_V += ent2 

                    #FCP for PAD
                    List1 , FCP1 = self.Adv_Random(data0['It'+str(It+1)][typ][1][i]['PP'],int(0.25*self.L*self.WW[typ]))
                    List2 , FCP2 = self.Adv_Fun(data0['It'+str(It+1)][typ][1][i]['PP'],int(0.25*self.L*self.WW[typ]))
                    FCP_R_P.append(FCP1)
                    FCP_G_P.append(FCP2)
                    Mix_dict = {'First':data0['It'+str(It+1)][typ][1][i]['P']}
                    Mix_dict['Routing'] = [To_list(data0['It'+str(It+1)][typ][1][i]['G1']),To_list(data0['It'+str(It+1)][typ][1][i]['G2'])]
                  
                    #ent1 = self.Simulation_ADV(self.Corruption_ADV(List1,self.L*self.WW[typ]),Mix_dict,nn,self.WW[typ])                   
                    #Ent_R_P += ent1

                    #ent2 = self.Simulation_ADV(self.Corruption_ADV(List2,self.L*self.WW[typ]),Mix_dict,nn,self.WW[typ])                   
                    #Ent_G_P += ent2 
                    #print('one')
                    
                FCP_R_V0.append(FCP_R_V)
                FCP_G_V0.append(FCP_G_V)
                Ent_R_V0.append(Ent_R_V)
                Ent_G_V0.append(Ent_G_V)
                
                FCP_R_P0.append(FCP_R_P)
                FCP_G_P0.append(FCP_G_P)
                #Ent_R_P0.append(Ent_R_P)
               # Ent_G_P0.append(Ent_G_P)  
            
            x1 =  Medd(To_list((np.matrix(FCP_R_V0))))
            x2 =  Medd(To_list((np.matrix(FCP_G_V0))))
            x3 =  Medd(To_list((np.matrix(FCP_R_P0))))
            x4 =  Medd(To_list((np.matrix(FCP_G_P0))))     
            
            data_out[typ] = {'FCP_R_V': x1, 'FCP_R_P': x3, 'FCP_G_V': x2, 'FCP_G_P': x4}
            data_out[typ]['Ent_R_V'] = (To_list((np.matrix(Ent_R_V0))))
            data_out[typ]['Ent_G_V'] = (To_list((np.matrix(Ent_G_V0))))                
            #data_out[typ]['Ent_R_P'] = (To_list((np.matrix(Ent_R_P0))))
            #data_out[typ]['Ent_G_P'] = (To_list((np.matrix(Ent_G_P0))))                  





    
    
    
        with open(File_Path +'/FCP_EXP.pkl','wb') as file:

            pickle.dump(data_out, file)     
            
            
        #print('2')  
        Budgets = [5/100*(i+1)+0.1 for i in range(6)]   
        data_out = {}
        for typ in ['RIPE', 'NYM']:
 
            FCP_R_V0 = []
            FCP_G_V0 = []

            FCP_R_P0 = []
            FCP_G_P0 = []
   
            for i in range(len(Budgets)):
                #print('iiii',i)
                
                FCP_R_V = []
                FCP_G_V = []
                Ent_R_V = []
                Ent_G_V = []
                
                FCP_R_P = []
                FCP_G_P = []
                Ent_R_P = []
                Ent_G_P = []                
                
                for It in range(Iterations):
                    #print('Ittt',It)
                    #FCP for Vanilla
                    List1 , FCP1 = self.Adv_Random(data0['It'+str(It+1)][typ][0][5]['PP'],int(Budgets[i]*self.L*self.WW[typ]))
                    #print('no')
                    List2 , FCP2 = self.Adv_Fun(data0['It'+str(It+1)][typ][0][5]['PP'],int(Budgets[i]*self.L*self.WW[typ]))
                    #print('yes')
                    FCP_R_V.append(FCP1)
                    FCP_G_V.append(FCP2)


                    #FCP for PAD
                    List1 , FCP1 = self.Adv_Random(data0['It'+str(It+1)][typ][1][5]['PP'],int(Budgets[i]*self.L*self.WW[typ]))
                    List2 , FCP2 = self.Adv_Fun(data0['It'+str(It+1)][typ][1][5]['PP'],int(Budgets[i]*self.L*self.WW[typ]))
                    FCP_R_P.append(FCP1)
                    FCP_G_P.append(FCP2)

        
                FCP_R_V0.append(FCP_R_V)
                FCP_G_V0.append(FCP_G_V)

                
                FCP_R_P0.append(FCP_R_P)
                FCP_G_P0.append(FCP_G_P)
 
            
            x1 =  Medd(To_list((np.matrix(FCP_R_V0))))
            x2 =  Medd(To_list((np.matrix(FCP_G_V0))))
            x3 =  Medd(To_list((np.matrix(FCP_R_P0))))
            x4 =  Medd(To_list((np.matrix(FCP_G_P0))))     
            
            data_out[typ] = {'FCP_R_V': x1, 'FCP_R_P': x3, 'FCP_G_V': x2, 'FCP_G_P': x4}

        with open(File_Path +'/Budget_EXP.pkl','wb') as file:

            pickle.dump(data_out, file)              
            

                           
    
    
    def FCP_JAR_2(self,nn,Iterations):
        #print('1')
        import numpy as np
        import pickle
        from Routing import Routing
        import json
        data_0 = self.JAR_data(Iterations)
        data0 = self.PDF_JARRR(data_0,True)

        #print('2')  
        Budgets = [10/100*(i+1) for i in range(4)]   
        data_out = {}
        for typ in ['RIPE', 'NYM']:
 
            FCP_R_V0 = []
            FCP_G_V0 = []

            FCP_R_P0 = []
            FCP_G_P0 = []
   
            for i in range(len(Budgets)):
                #print('iiii',i)
                
                FCP_R_V = []
                FCP_G_V = []
                Ent_R_V = []
                Ent_G_V = []
                
                FCP_R_P = []
                FCP_G_P = []
                Ent_R_P = []
                Ent_G_P = []                
                
                for It in range(Iterations):
                    #print('Ittt',It)
                    #FCP for Vanilla
                    List1 , FCP1 = self.Adv_Random(data0['It'+str(It+1)][typ][0][2]['PP'],int(Budgets[i]*self.L*self.WW[typ]))
                    #print('no')
                    List2 , FCP2 = self.Adv_Fun(data0['It'+str(It+1)][typ][0][2]['PP'],int(Budgets[i]*self.L*self.WW[typ]))
                    #print('yes')
                    FCP_R_V.append(FCP1)
                    FCP_G_V.append(FCP2)


                    #FCP for PAD
                    List1 , FCP1 = self.Adv_Random(data0['It'+str(It+1)][typ][1][2]['PP'],int(Budgets[i]*self.L*self.WW[typ]))
                    List2 , FCP2 = self.Adv_Fun(data0['It'+str(It+1)][typ][1][2]['PP'],int(Budgets[i]*self.L*self.WW[typ]))
                    FCP_R_P.append(FCP1)
                    FCP_G_P.append(FCP2)

        
                FCP_R_V0.append(FCP_R_V)
                FCP_G_V0.append(FCP_G_V)

                
                FCP_R_P0.append(FCP_R_P)
                FCP_G_P0.append(FCP_G_P)
 
            
            x1 =  Medd(To_list((np.matrix(FCP_R_V0))))
            x2 =  Medd(To_list((np.matrix(FCP_G_V0))))
            x3 =  Medd(To_list((np.matrix(FCP_R_P0))))
            x4 =  Medd(To_list((np.matrix(FCP_G_P0))))     
            
            data_out[typ] = {'FCP_R_V': x1, 'FCP_R_P': x3, 'FCP_G_V': x2, 'FCP_G_P': x4}
        
        return data_out


    
    
    
    
    
    
    def FCP_Reliability(self,nn,Iterations,File_Path=''):
        #print('1')
        import numpy as np
        import pickle
        from Routing import Routing
        import json
        data_0 = self.Data_Set_General
        data0 = self.PDF_Reliability(data_0,True)
        data_out = {}
        for typ in ['RIPE', 'NYM']:
 
            FCP_R_V0 = []
            FCP_G_V0 = []
            Ent_R_V0 = []
            Ent_G_V0 = []
            
            FCP_R_P0 = []
            FCP_G_P0 = []
            Ent_R_P0 = []
            Ent_G_P0 = [] 
            #for i in [2]:
            for i in range(len(data0['It1']['NYM'][0])):
                
                FCP_R_V = []
                FCP_G_V = []
                Ent_R_V = []
                Ent_G_V = []
                
                FCP_R_P = []
                FCP_G_P = []
                Ent_R_P = []
                Ent_G_P = []                
                
                for It in range(Iterations):
                    #FCP for Vanilla
                    List1 , FCP1 = self.Adv_Random(data0['It'+str(It+1)][typ][0][i]['PP'],int(0.25*self.L*self.WW[typ]))
                    List2 , FCP2 = self.Adv_Fun(data0['It'+str(It+1)][typ][0][i]['PP'],int(0.25*self.L*self.WW[typ]))
                    FCP_R_V.append(FCP1)
                    FCP_G_V.append(FCP2)
                    Mix_dict = {'First':data0['It'+str(It+1)][typ][0][i]['P']}
                    Mix_dict['Routing'] = [To_list(data0['It'+str(It+1)][typ][0][i]['G1']),To_list(data0['It'+str(It+1)][typ][0][i]['G2'])]
                  
                    ent1 = self.Simulation_ADV(self.Corruption_ADV(List1,self.L*self.WW[typ]),Mix_dict,nn,self.WW[typ])                   
                    Ent_R_V += ent1

                    ent2 = self.Simulation_ADV(self.Corruption_ADV(List2,self.L*self.WW[typ]),Mix_dict,nn,self.WW[typ])                   
                    Ent_G_V += ent2 

                    #FCP for PAD
                    List1 , FCP1 = self.Adv_Random(data0['It'+str(It+1)][typ][1][i]['PP'],int(0.25*self.L*self.WW[typ]))
                    List2 , FCP2 = self.Adv_Fun(data0['It'+str(It+1)][typ][1][i]['PP'],int(0.25*self.L*self.WW[typ]))
                    FCP_R_P.append(FCP1)
                    FCP_G_P.append(FCP2)
                    Mix_dict = {'First':data0['It'+str(It+1)][typ][1][i]['P']}
                    Mix_dict['Routing'] = [To_list(data0['It'+str(It+1)][typ][1][i]['G1']),To_list(data0['It'+str(It+1)][typ][1][i]['G2'])]
                  

                FCP_R_V0.append(FCP_R_V)
                FCP_G_V0.append(FCP_G_V)
                Ent_R_V0.append(Ent_R_V)
                Ent_G_V0.append(Ent_G_V)
                
                FCP_R_P0.append(FCP_R_P)
                FCP_G_P0.append(FCP_G_P)

            x1 =  Medd(To_list((np.matrix(FCP_R_V0))))
            x2 =  Medd(To_list((np.matrix(FCP_G_V0))))
            x3 =  Medd(To_list((np.matrix(FCP_R_P0))))
            x4 =  Medd(To_list((np.matrix(FCP_G_P0))))     
            
            data_out[typ] = {'FCP_R_V': x1, 'FCP_R_P': x3, 'FCP_G_V': x2, 'FCP_G_P': x4}
            data_out[typ]['Ent_R_V'] = (To_list((np.matrix(Ent_R_V0))))
            data_out[typ]['Ent_G_V'] = (To_list((np.matrix(Ent_G_V0))))                



        with open(File_Path +'/FCP_EXP.pkl','wb') as file:

            pickle.dump(data_out, file)     
            
            
        #print('2')  
        Budgets = [5/100*(i+1)+0.1 for i in range(6)]   
        data_out = {}
        for typ in ['RIPE', 'NYM']:
 
            FCP_R_V0 = []
            FCP_G_V0 = []

            FCP_R_P0 = []
            FCP_G_P0 = []
   
            for i in range(len(Budgets)):
                #print('iiii',i)
                
                FCP_R_V = []
                FCP_G_V = []
                Ent_R_V = []
                Ent_G_V = []
                
                FCP_R_P = []
                FCP_G_P = []
                Ent_R_P = []
                Ent_G_P = []                
                
                for It in range(Iterations):
                    #print('Ittt',It)
                    #FCP for Vanilla
                    List1 , FCP1 = self.Adv_Random(data0['It'+str(It+1)][typ][0][5]['PP'],int(Budgets[i]*self.L*self.WW[typ]))
                    #print('no')
                    List2 , FCP2 = self.Adv_Fun(data0['It'+str(It+1)][typ][0][5]['PP'],int(Budgets[i]*self.L*self.WW[typ]))
                    #print('yes')
                    FCP_R_V.append(FCP1)
                    FCP_G_V.append(FCP2)


                    #FCP for PAD
                    List1 , FCP1 = self.Adv_Random(data0['It'+str(It+1)][typ][1][5]['PP'],int(Budgets[i]*self.L*self.WW[typ]))
                    List2 , FCP2 = self.Adv_Fun(data0['It'+str(It+1)][typ][1][5]['PP'],int(Budgets[i]*self.L*self.WW[typ]))
                    FCP_R_P.append(FCP1)
                    FCP_G_P.append(FCP2)

        
                FCP_R_V0.append(FCP_R_V)
                FCP_G_V0.append(FCP_G_V)

                
                FCP_R_P0.append(FCP_R_P)
                FCP_G_P0.append(FCP_G_P)
 
            
            x1 =  Medd(To_list((np.matrix(FCP_R_V0))))
            x2 =  Medd(To_list((np.matrix(FCP_G_V0))))
            x3 =  Medd(To_list((np.matrix(FCP_R_P0))))
            x4 =  Medd(To_list((np.matrix(FCP_G_P0))))     
            
            data_out[typ] = {'FCP_R_V': x1, 'FCP_R_P': x3, 'FCP_G_V': x2, 'FCP_G_P': x4}

        with open(File_Path +'/Budget_EXP.pkl','wb') as file:

            pickle.dump(data_out, file)              
            
   
    
    def FCP_Reliability_2(self,nn,Iterations,File_Path=''):
        #print('1')
        import numpy as np
        import pickle
        from Routing import Routing
        import json
        data_0 = self.Data_Set_General
        data0 = self.PDF_Reliabilityyy(data_0,True)

            
            
        #print('2')  
        Budgets = [10/100*(i+1) for i in range(4)]   
        data_out = {}
        for typ in ['RIPE', 'NYM']:
 
            FCP_R_V0 = []
            FCP_G_V0 = []

            FCP_R_P0 = []
            FCP_G_P0 = []
   
            for i in range(len(Budgets)):
                #print('iiii',i)
                
                FCP_R_V = []
                FCP_G_V = []
                Ent_R_V = []
                Ent_G_V = []
                
                FCP_R_P = []
                FCP_G_P = []
                Ent_R_P = []
                Ent_G_P = []                
                
                for It in range(Iterations):
                    #print('Ittt',It)
                    #FCP for Vanilla
                    List1 , FCP1 = self.Adv_Random(data0['It'+str(It+1)][typ][0][2]['PP'],int(Budgets[i]*self.L*self.WW[typ]))
                    #print('no')
                    List2 , FCP2 = self.Adv_Fun(data0['It'+str(It+1)][typ][0][2]['PP'],int(Budgets[i]*self.L*self.WW[typ]))
                    #print('yes')
                    FCP_R_V.append(FCP1)
                    FCP_G_V.append(FCP2)


                    #FCP for PAD
                    List1 , FCP1 = self.Adv_Random(data0['It'+str(It+1)][typ][1][2]['PP'],int(Budgets[i]*self.L*self.WW[typ]))
                    List2 , FCP2 = self.Adv_Fun(data0['It'+str(It+1)][typ][1][2]['PP'],int(Budgets[i]*self.L*self.WW[typ]))
                    FCP_R_P.append(FCP1)
                    FCP_G_P.append(FCP2)

        
                FCP_R_V0.append(FCP_R_V)
                FCP_G_V0.append(FCP_G_V)

                
                FCP_R_P0.append(FCP_R_P)
                FCP_G_P0.append(FCP_G_P)
 
            
            x1 =  Medd(To_list((np.matrix(FCP_R_V0))))
            x2 =  Medd(To_list((np.matrix(FCP_G_V0))))
            x3 =  Medd(To_list((np.matrix(FCP_R_P0))))
            x4 =  Medd(To_list((np.matrix(FCP_G_P0))))     
            
            data_out[typ] = {'FCP_R_V': x1, 'FCP_R_P': x3, 'FCP_G_V': x2, 'FCP_G_P': x4}
            
        return data_out

    
    
    def FCP_Reliability_(self,nn,Iterations):
        #print('1')
        import numpy as np
        import pickle
        from Routing import Routing
        import json
        data_0 = self.Data_Set_General
        data0 = self.PDF_Reliabilityyy(data_0,True)
        data_out = {}
        for typ in ['RIPE', 'NYM']:
 
            FCP_R_V0 = []
            FCP_G_V0 = []
            Ent_R_V0 = []
            Ent_G_V0 = []
            
            FCP_R_P0 = []
            FCP_G_P0 = []
            Ent_R_P0 = []
            Ent_G_P0 = [] 
            #for i in [2]:
            for i in range(len(data0['It1']['NYM'][0])):
                
                FCP_R_V = []
                FCP_G_V = []
                Ent_R_V = []
                Ent_G_V = []
                
                FCP_R_P = []
                FCP_G_P = []
                Ent_R_P = []
                Ent_G_P = []                
                
                for It in range(Iterations):
                    #FCP for Vanilla
                    List1 , FCP1 = self.Adv_Random(data0['It'+str(It+1)][typ][0][i]['PP'],int(0.25*self.L*self.WW[typ]))
                    List2 , FCP2 = self.Adv_Fun(data0['It'+str(It+1)][typ][0][i]['PP'],int(0.25*self.L*self.WW[typ]))
                    FCP_R_V.append(FCP1)
                    FCP_G_V.append(FCP2)
                    Mix_dict = {'First':data0['It'+str(It+1)][typ][0][i]['P']}
                    Mix_dict['Routing'] = [To_list(data0['It'+str(It+1)][typ][0][i]['G1']),To_list(data0['It'+str(It+1)][typ][0][i]['G2'])]
                  
                    ent1 = self.Simulation_ADV(self.Corruption_ADV(List1,self.L*self.WW[typ]),Mix_dict,nn,self.WW[typ])                   
                    Ent_R_V += ent1

                    ent2 = self.Simulation_ADV(self.Corruption_ADV(List2,self.L*self.WW[typ]),Mix_dict,nn,self.WW[typ])                   
                    Ent_G_V += ent2 

                    #FCP for PAD
                    List1 , FCP1 = self.Adv_Random(data0['It'+str(It+1)][typ][1][i]['PP'],int(0.25*self.L*self.WW[typ]))
                    List2 , FCP2 = self.Adv_Fun(data0['It'+str(It+1)][typ][1][i]['PP'],int(0.25*self.L*self.WW[typ]))
                    FCP_R_P.append(FCP1)
                    FCP_G_P.append(FCP2)
                    Mix_dict = {'First':data0['It'+str(It+1)][typ][1][i]['P']}
                    Mix_dict['Routing'] = [To_list(data0['It'+str(It+1)][typ][1][i]['G1']),To_list(data0['It'+str(It+1)][typ][1][i]['G2'])]
                  
                    #ent1 = self.Simulation_ADV(self.Corruption_ADV(List1,self.L*self.WW[typ]),Mix_dict,nn,self.WW[typ])                   
                    #Ent_R_P += ent1

                    #ent2 = self.Simulation_ADV(self.Corruption_ADV(List2,self.L*self.WW[typ]),Mix_dict,nn,self.WW[typ])                   
                    #Ent_G_P += ent2 
                    #print('one')
                    
                FCP_R_V0.append(FCP_R_V)
                FCP_G_V0.append(FCP_G_V)
                Ent_R_V0.append(Ent_R_V)
                Ent_G_V0.append(Ent_G_V)
                
                FCP_R_P0.append(FCP_R_P)
                FCP_G_P0.append(FCP_G_P)
                #Ent_R_P0.append(Ent_R_P)
                #Ent_G_P0.append(Ent_G_P)  
            
            x1 =  Medd(To_list((np.matrix(FCP_R_V0))))
            x2 =  Medd(To_list((np.matrix(FCP_G_V0))))
            x3 =  Medd(To_list((np.matrix(FCP_R_P0))))
            x4 =  Medd(To_list((np.matrix(FCP_G_P0))))     
            
            data_out[typ] = {'FCP_R_V': x1, 'FCP_R_P': x3, 'FCP_G_V': x2, 'FCP_G_P': x4}
            data_out[typ]['Ent_R_V'] = (To_list((np.matrix(Ent_R_V0))))
            data_out[typ]['Ent_G_V'] = (To_list((np.matrix(Ent_G_V0))))                
            #data_out[typ]['Ent_R_P'] = (To_list((np.matrix(Ent_R_P0))))
            #data_out[typ]['Ent_G_P'] = (To_list((np.matrix(Ent_G_P0))))                  



        return data_out

            

    
    
    
    
    def FCP_Latency(self,nn,Iterations,method,File_Path=''):
        #print('1')
        import numpy as np
        import pickle
        from Routing import Routing
        import json
        data_0 = self.Data_Set_General
        data0 = self.PDF_Latency(data_0,True)
        data_out = {}
        for typ in ['RIPE', 'NYM']:
 
            FCP_R_V0 = []
            FCP_G_V0 = []
            Ent_R_V0 = []
            Ent_G_V0 = []
            
            FCP_R_P0 = []
            FCP_G_P0 = []
            Ent_R_P0 = []
            Ent_G_P0 = [] 
            #for i in [2]:
            for i in range(len(data0['It1']['NYM']['LC'][0])):
                
                FCP_R_V = []
                FCP_G_V = []
                Ent_R_V = []
                Ent_G_V = []
                
                FCP_R_P = []
                FCP_G_P = []
                Ent_R_P = []
                Ent_G_P = []                
                
                for It in range(Iterations):
                    #FCP for Vanilla
                    List1 , FCP1 = self.Adv_Random(data0['It'+str(It+1)][typ][method][0][i]['PP'],int(0.25*self.L*self.WW[typ]))
                    List2 , FCP2 = self.Adv_Fun(data0['It'+str(It+1)][typ][method][0][i]['PP'],int(0.25*self.L*self.WW[typ]))
                    FCP_R_V.append(FCP1)
                    FCP_G_V.append(FCP2)
                    Mix_dict = {'First':data0['It'+str(It+1)][typ][method][0][i]['P']}
                    Mix_dict['Routing'] = [To_list(data0['It'+str(It+1)][typ][method][0][i]['G1']),To_list(data0['It'+str(It+1)][typ][method][0][i]['G2'])]
                  
                    ent1 = self.Simulation_ADV(self.Corruption_ADV(List1,self.L*self.WW[typ]),Mix_dict,nn,self.WW[typ])                   
                    Ent_R_V += ent1

                    ent2 = self.Simulation_ADV(self.Corruption_ADV(List2,self.L*self.WW[typ]),Mix_dict,nn,self.WW[typ])                   
                    Ent_G_V += ent2 

                    #FCP for PAD
                    List1 , FCP1 = self.Adv_Random(data0['It'+str(It+1)][typ][method][1][i]['PP'],int(0.25*self.L*self.WW[typ]))
                    List2 , FCP2 = self.Adv_Fun(data0['It'+str(It+1)][typ][method][1][i]['PP'],int(0.25*self.L*self.WW[typ]))
                    FCP_R_P.append(FCP1)
                    FCP_G_P.append(FCP2)
                    Mix_dict = {'First':data0['It'+str(It+1)][typ][method][1][i]['P']}
                    Mix_dict['Routing'] = [To_list(data0['It'+str(It+1)][typ][method][1][i]['G1']),To_list(data0['It'+str(It+1)][typ][method][1][i]['G2'])]
                  
                    ent1 = self.Simulation_ADV(self.Corruption_ADV(List1,self.L*self.WW[typ]),Mix_dict,nn,self.WW[typ])                   
                    Ent_R_P += ent1

                    ent2 = self.Simulation_ADV(self.Corruption_ADV(List2,self.L*self.WW[typ]),Mix_dict,nn,self.WW[typ])                   
                    Ent_G_P += ent2 
                    #print('one')
                    
                FCP_R_V0.append(FCP_R_V)
                FCP_G_V0.append(FCP_G_V)
                Ent_R_V0.append(Ent_R_V)
                Ent_G_V0.append(Ent_G_V)
                
                FCP_R_P0.append(FCP_R_P)
                FCP_G_P0.append(FCP_G_P)
                Ent_R_P0.append(Ent_R_P)
                Ent_G_P0.append(Ent_G_P)  
            
            x1 =  Medd(To_list((np.matrix(FCP_R_V0))))
            x2 =  Medd(To_list((np.matrix(FCP_G_V0))))
            x3 =  Medd(To_list((np.matrix(FCP_R_P0))))
            x4 =  Medd(To_list((np.matrix(FCP_G_P0))))     
            
            data_out[typ] = {'FCP_R_V': x1, 'FCP_R_P': x3, 'FCP_G_V': x2, 'FCP_G_P': x4}
            data_out[typ]['Ent_R_V'] = (To_list((np.matrix(Ent_R_V0))))
            data_out[typ]['Ent_G_V'] = (To_list((np.matrix(Ent_G_V0))))                
            data_out[typ]['Ent_R_P'] = (To_list((np.matrix(Ent_R_P0))))
            data_out[typ]['Ent_G_P'] = (To_list((np.matrix(Ent_G_P0))))                  





            
            
        with open(File_Path +'/'+method+'FCP_EXP.pkl','wb') as file:

            pickle.dump(data_out, file)     
            
            
        #print('2')  
        Budgets = [5/100*(i+1)+0.1 for i in range(6)]   
        data_out = {}
        for typ in ['RIPE', 'NYM']:
 
            FCP_R_V0 = []
            FCP_G_V0 = []

            FCP_R_P0 = []
            FCP_G_P0 = []
   
            for i in range(len(Budgets)):
               # print('iiii',i)
                
                FCP_R_V = []
                FCP_G_V = []
                Ent_R_V = []
                Ent_G_V = []
                
                FCP_R_P = []
                FCP_G_P = []
                Ent_R_P = []
                Ent_G_P = []                
                
                for It in range(Iterations):
                    #print('Ittt',It)
                    #FCP for Vanilla
                    List1 , FCP1 = self.Adv_Random(data0['It'+str(It+1)][typ][method][0][5]['PP'],int(Budgets[i]*self.L*self.WW[typ]))
                    #print('no')
                    List2 , FCP2 = self.Adv_Fun(data0['It'+str(It+1)][typ][method][0][5]['PP'],int(Budgets[i]*self.L*self.WW[typ]))
                    #print('yes')
                    FCP_R_V.append(FCP1)
                    FCP_G_V.append(FCP2)


                    #FCP for PAD
                    List1 , FCP1 = self.Adv_Random(data0['It'+str(It+1)][typ][method][1][5]['PP'],int(Budgets[i]*self.L*self.WW[typ]))
                    List2 , FCP2 = self.Adv_Fun(data0['It'+str(It+1)][typ][method][1][5]['PP'],int(Budgets[i]*self.L*self.WW[typ]))
                    FCP_R_P.append(FCP1)
                    FCP_G_P.append(FCP2)

        
                FCP_R_V0.append(FCP_R_V)
                FCP_G_V0.append(FCP_G_V)

                
                FCP_R_P0.append(FCP_R_P)
                FCP_G_P0.append(FCP_G_P)
 
            
            x1 =  Medd(To_list((np.matrix(FCP_R_V0))))
            x2 =  Medd(To_list((np.matrix(FCP_G_V0))))
            x3 =  Medd(To_list((np.matrix(FCP_R_P0))))
            x4 =  Medd(To_list((np.matrix(FCP_G_P0))))     
            
            data_out[typ] = {'FCP_R_V': x1, 'FCP_R_P': x3, 'FCP_G_V': x2, 'FCP_G_P': x4}

        with open(File_Path +'/'+method+'Budget_EXP.pkl','wb') as file:

            pickle.dump(data_out, file)              
               
    
    
    
    def FCP_Latency_2(self,nn,Iterations,method):
        #print('1')
        import numpy as np
        import pickle
        from Routing import Routing
        import json
        data_0 = self.Data_Set_General
        data0 = self.PDF_Latencyyy(data_0,True)

            
        #print('2')  
        Budgets = [5/100*(i+1)+0.1 for i in range(6)]   
        data_out = {}
        for typ in ['RIPE', 'NYM']:
 
            FCP_R_V0 = []
            FCP_G_V0 = []

            FCP_R_P0 = []
            FCP_G_P0 = []
   
            for i in range(len(Budgets)):
                #print('iiii',i)
                
                FCP_R_V = []
                FCP_G_V = []
                Ent_R_V = []
                Ent_G_V = []
                
                FCP_R_P = []
                FCP_G_P = []
                Ent_R_P = []
                Ent_G_P = []                
                
                for It in range(Iterations):
                    #print('Ittt',It)
                    #FCP for Vanilla
                    List1 , FCP1 = self.Adv_Random(data0['It'+str(It+1)][typ][method][0][2]['PP'],int(Budgets[i]*self.L*self.WW[typ]))
                    #print('no')
                    List2 , FCP2 = self.Adv_Fun(data0['It'+str(It+1)][typ][method][0][2]['PP'],int(Budgets[i]*self.L*self.WW[typ]))
                    #print('yes')
                    FCP_R_V.append(FCP1)
                    FCP_G_V.append(FCP2)


                    #FCP for PAD
                    List1 , FCP1 = self.Adv_Random(data0['It'+str(It+1)][typ][method][1][2]['PP'],int(Budgets[i]*self.L*self.WW[typ]))
                    List2 , FCP2 = self.Adv_Fun(data0['It'+str(It+1)][typ][method][1][2]['PP'],int(Budgets[i]*self.L*self.WW[typ]))
                    FCP_R_P.append(FCP1)
                    FCP_G_P.append(FCP2)

        
                FCP_R_V0.append(FCP_R_V)
                FCP_G_V0.append(FCP_G_V)

                
                FCP_R_P0.append(FCP_R_P)
                FCP_G_P0.append(FCP_G_P)
 
            
            x1 =  Medd(To_list((np.matrix(FCP_R_V0))))
            x2 =  Medd(To_list((np.matrix(FCP_G_V0))))
            x3 =  Medd(To_list((np.matrix(FCP_R_P0))))
            x4 =  Medd(To_list((np.matrix(FCP_G_P0))))     
            
            data_out[typ] = {'FCP_R_V': x1, 'FCP_R_P': x3, 'FCP_G_V': x2, 'FCP_G_P': x4}

        #with open(File_Path +'/'+method+'Budget_EXP.pkl','wb') as file:

            #pickle.dump(data_out, file)
                  
        return data_out
    
    
    
    
    
    
    
    
    def FCP_Latency_(self,nn,Iterations,method):
        #print('1')
        import numpy as np
        import pickle
        from Routing import Routing
        import json
        data_0 = self.Data_Set_General
        data0 = self.PDF_Latencyyy(data_0,True)
        data_out = {}
        for typ in ['RIPE', 'NYM']:
 
            FCP_R_V0 = []
            FCP_G_V0 = []
            Ent_R_V0 = []
            Ent_G_V0 = []
            
            FCP_R_P0 = []
            FCP_G_P0 = []
            Ent_R_P0 = []
            Ent_G_P0 = [] 
            #for i in [2]:
            for i in range(len(data0['It1']['NYM']['LC'][0])):
                
                FCP_R_V = []
                FCP_G_V = []
                Ent_R_V = []
                Ent_G_V = []
                
                FCP_R_P = []
                FCP_G_P = []
                #Ent_R_P = []
                #Ent_G_P = []                
                
                for It in range(Iterations):
                    #FCP for Vanilla
                    List1 , FCP1 = self.Adv_Random(data0['It'+str(It+1)][typ][method][0][i]['PP'],int(0.25*self.L*self.WW[typ]))
                    List2 , FCP2 = self.Adv_Fun(data0['It'+str(It+1)][typ][method][0][i]['PP'],int(0.25*self.L*self.WW[typ]))
                    FCP_R_V.append(FCP1)
                    FCP_G_V.append(FCP2)
                    Mix_dict = {'First':data0['It'+str(It+1)][typ][method][0][i]['P']}
                    Mix_dict['Routing'] = [To_list(data0['It'+str(It+1)][typ][method][0][i]['G1']),To_list(data0['It'+str(It+1)][typ][method][0][i]['G2'])]
                  
                    ent1 = self.Simulation_ADV(self.Corruption_ADV(List1,self.L*self.WW[typ]),Mix_dict,nn,self.WW[typ])                   
                    Ent_R_V += ent1

                    ent2 = self.Simulation_ADV(self.Corruption_ADV(List2,self.L*self.WW[typ]),Mix_dict,nn,self.WW[typ])                   
                    Ent_G_V += ent2 

                    #FCP for PAD
                    List1 , FCP1 = self.Adv_Random(data0['It'+str(It+1)][typ][method][1][i]['PP'],int(0.25*self.L*self.WW[typ]))
                    List2 , FCP2 = self.Adv_Fun(data0['It'+str(It+1)][typ][method][1][i]['PP'],int(0.25*self.L*self.WW[typ]))
                    FCP_R_P.append(FCP1)
                    FCP_G_P.append(FCP2)
                    Mix_dict = {'First':data0['It'+str(It+1)][typ][method][1][i]['P']}
                    Mix_dict['Routing'] = [To_list(data0['It'+str(It+1)][typ][method][1][i]['G1']),To_list(data0['It'+str(It+1)][typ][method][1][i]['G2'])]
                  
                    #ent1 = self.Simulation_ADV(self.Corruption_ADV(List1,self.L*self.WW[typ]),Mix_dict,nn,self.WW[typ])                   
                    #Ent_R_P += ent1

                    #ent2 = self.Simulation_ADV(self.Corruption_ADV(List2,self.L*self.WW[typ]),Mix_dict,nn,self.WW[typ])                   
                    #Ent_G_P += ent2 
                    #print('one')
                    
                FCP_R_V0.append(FCP_R_V)
                FCP_G_V0.append(FCP_G_V)
                Ent_R_V0.append(Ent_R_V)
                Ent_G_V0.append(Ent_G_V)
                
                FCP_R_P0.append(FCP_R_P)
                FCP_G_P0.append(FCP_G_P)
                #Ent_R_P0.append(Ent_R_P)
                #Ent_G_P0.append(Ent_G_P)  
            
            x1 =  Medd(To_list((np.matrix(FCP_R_V0))))
            x2 =  Medd(To_list((np.matrix(FCP_G_V0))))
            x3 =  Medd(To_list((np.matrix(FCP_R_P0))))
            x4 =  Medd(To_list((np.matrix(FCP_G_P0))))     
            
            data_out[typ] = {'FCP_R_V': x1, 'FCP_R_P': x3, 'FCP_G_V': x2, 'FCP_G_P': x4}
            data_out[typ]['Ent_R_V'] = (To_list((np.matrix(Ent_R_V0))))
            data_out[typ]['Ent_G_V'] = (To_list((np.matrix(Ent_G_V0))))                
            #data_out[typ]['Ent_R_P'] = (To_list((np.matrix(Ent_R_P0))))
            #data_out[typ]['Ent_G_P'] = (To_list((np.matrix(Ent_G_P0))))                  





            
            
        #with open(File_Path +'/'+method+'FCP_EXP1.pkl','wb') as file:

            #pickle.dump(data_out, file)   
        return data_out
            







    def FCP_Latency_3(self,nn,Iterations,method):

        import numpy as np
        import pickle
        from Routing import Routing
        import json
        data_0 = self.Data_Set_General
        data0 = self.PDF_Latencyyy(data_0,True)
        data_out = {'RIPE':{},'NYM':{}}
        
        
        for typ in ['RIPE', 'NYM']:
 

            Ent_R_V0 = []
            Ent_G_V0 = []
            

            for i in range(len(data0['It1']['NYM']['LC'][0])):

                Ent_R_V = []
                Ent_G_V = []
                
        
                
                for It in range(Iterations):
                    #FCP for Vanilla
                    List1 , FCP1 = self.Adv_Random(data0['It'+str(It+1)][typ][method][0][i]['PP'],int(0.25*self.L*self.WW[typ]))
                    List2 , FCP2 = self.Adv_Fun(data0['It'+str(It+1)][typ][method][0][i]['PP'],int(0.25*self.L*self.WW[typ]))

                    Mix_dict = {'First':data0['It'+str(It+1)][typ][method][0][i]['P']}
                    Mix_dict['Routing'] = [To_list(data0['It'+str(It+1)][typ][method][0][i]['G1']),To_list(data0['It'+str(It+1)][typ][method][0][i]['G2'])]
                  
                    ent1 = self.Simulation_ADV(self.Corruption_ADV(List1,self.L*self.WW[typ]),Mix_dict,nn,self.WW[typ])                   
                    Ent_R_V += ent1

                    ent2 = self.Simulation_ADV(self.Corruption_ADV(List2,self.L*self.WW[typ]),Mix_dict,nn,self.WW[typ])                   
                    Ent_G_V += ent2 

                    #FCP for PAD
                    List1 , FCP1 = self.Adv_Random(data0['It'+str(It+1)][typ][method][1][i]['PP'],int(0.25*self.L*self.WW[typ]))
                    List2 , FCP2 = self.Adv_Fun(data0['It'+str(It+1)][typ][method][1][i]['PP'],int(0.25*self.L*self.WW[typ]))
                    #FCP_R_P.append(FCP1)
                    #FCP_G_P.append(FCP2)
                    Mix_dict = {'First':data0['It'+str(It+1)][typ][method][1][i]['P']}
                    Mix_dict['Routing'] = [To_list(data0['It'+str(It+1)][typ][method][1][i]['G1']),To_list(data0['It'+str(It+1)][typ][method][1][i]['G2'])]
                  

                Ent_R_V0.append(Ent_R_V)
                Ent_G_V0.append(Ent_G_V)
                

            data_out[typ]['Ent_R_V'] = (To_list((np.matrix(Ent_R_V0))))
            data_out[typ]['Ent_G_V'] = (To_list((np.matrix(Ent_G_V0))))                

        
        return data_out
            












    
#with open('data0_.pkl','rb') as pkl_file:
#    data0 = pickle.load(pkl_file)
    
#data1 = {}    
#data2 = {}
#for i in range(20):
#    data1['It'+str(i+1)] = data0['NYM']['It'+str(i+1)]
#    data2['It'+str(i+1)] = data0['RIPE']['It'+str(i+1)]    
#data00 = {'NYM':data1,'RIPE':data2}
    
#with open('dataset.pkl','wb') as pkl_file:
#    pickle.dump(data00,pkl_file)
    
        
        
        
        
        
        
        
