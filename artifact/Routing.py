
# -*- coding: utf-8 -*-
"""
This function helps to model the routing approaches in DP-Mix.
"""
from math import exp
from scipy import constants

# Import library for making the simulation, making random choices,
#creating exponential delays, and defining matrixes.
from scipy.stats import expon
import simpy
import random
import numpy  as np
import pickle
import math
from scipy.spatial.distance import euclidean
from scipy.stats import entropy  # Computes KL divergence
def normalize_rows_by_sum(matrix):
    row_sums = matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    return matrix / row_sums
def compute_conditionals(probs_):
    probs = probs_.copy()
    W = round(len(probs) ** (1/3))
    A = np.zeros((W, W, W), dtype=float)
    
    for i in range(W):
        for j in range(W):
            start = i * (W ** 2) + j * W
            end = start + W
            #print(len(probs))
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
def compute_metrics(p1, p2):
    """
    Compute Euclidean distance and Kullback–Leibler (KL) divergence between P1 and P2.

    Parameters:
    p1 (array-like): First probability distribution (P1).
    p2 (array-like): Second probability distribution (P2).

    Returns:
    tuple: (Euclidean distance, KL divergence)
    """
    # Ensure P1 and P2 are numpy arrays
    p1 = np.array(p1)
    p2 = np.array(p2)

    # Compute Euclidean distance
    #euclidean_dist = euclidean(p1, p2)

    # Compute KL divergence (avoid division by zero by adding a small epsilon)
    epsilon = 1e-10
    p1 = np.clip(p1, epsilon, 1)  # Ensure no zero values
    p2 = np.clip(p2, epsilon, 1)  # Ensure no zero values
    kl_div = entropy(p1, p2)  # KL(P1 || P2)

    return  kl_div
'''
# Example usage
p1 = [0.0, 1, 0.0]
p2 = [0.5, 0.0, 0.5]
kl_divergence = compute_metrics(p1, p2)


print(f"KL Divergence: {kl_divergence}")
'''
def compute_cdf(D, E):
    """
    Computes the CDF for a dataset D evaluated at points in E.

    Args:
        D (list): A list of data values (numerical).
        E (list): A list of evaluation points (numerical).

    Returns:
        list: A list O, where O[i] represents the percentage of values in D less than E[i].
    """
    # Sort the data list for efficient comparison
    D_sorted = sorted(D)
    n = len(D)
    O = []

    for e in E:
        # Count the number of elements in D that are less than e
        count = sum(1 for x in D_sorted if x <= e)
        # Calculate the percentage
        percentage = count / n
        O.append(percentage)

    return O
def Normalized(List, Omega0,Co):
    Sum = np.sum([List[i]*Co[i] for i in range(len(List))])
    Sum = Sum/Omega0
    return [List[i]/Sum for i in range(len(List))]
    
    
#print(Normalized([0.5,0.1,0.1],1.5))
    
def Zero_Check(A):
    o1,o2 = np.shape(A)
    for i in range(o1):
        for j in range(o2):
            if int((10**(6))*A[i,j]) ==0:
                A[i,j] = 10**(-20)
    return A

def sort_and_recover(input_list):
    """
    Sorts the input list and returns:
    - The sorted list
    - A recovery list (indices mapping sorted list back to the original list)
    
    Args:
        input_list (list): The original list to sort.

    Returns:
        tuple: (sorted_list, recovery_list)
    """
    # Pair elements with their original indices
    indexed_list = list(enumerate(input_list))
    # Sort based on the values
    sorted_indexed_list = sorted(indexed_list, key=lambda x: x[1])
    # Extract the sorted list and the recovery indices
    sorted_list = [x[1] for x in sorted_indexed_list]
    recovery_list = [x[0] for x in sorted_indexed_list]
    return sorted_list, recovery_list

def recover_original(sorted_list, recovery_list):
    """
    Reconstructs the original list using the sorted list and recovery list.
    
    Args:
        sorted_list (list): The sorted list.
        recovery_list (list): The recovery list (indices mapping to original).

    Returns:
        list: The reconstructed original list.
    """
    # Create a placeholder list for the original
    original_list = [None] * len(sorted_list)
    # Use the recovery list to restore the original order
    for i, index in enumerate(recovery_list):
        original_list[index] = sorted_list[i]
    return original_list
#from Message_ import message

#from GateWay import GateWay
#from Mix_Node_ import Mix

#from NYM import MixNet

#from Message_Genartion_and_mix_net_processing_ import Message_Genartion_and_mix_net_processing
def To_list(List):
    import numpy as np
    List_ = List.tolist()
    if len(List_)==1:
        output = List_[0]
    else:
        output = List_
    
    return output
def subtract_lists(list1, list2):
    # Check if both lists have the same length
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length.")
    
    # Perform element-wise subtraction
    result = [a - b for a, b in zip(list1, list2)]
    for i in range(len(result)):
        if result[i] <0:
            result[i] =0
        
    
    return result

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


class Routing(object):
    def __init__(self,N,L):
        self.N = N
        self.L = L
        self.W = int(self.N/self.L)
        self.EPS = [0,0.5,1,2]
        
    def Path_2_KLD(self,Paths,Bandwidth,L):
        W = round(len(Paths)**(1/L))
        output = 0
        for i in range(W):
            for j in range(W):
                for k in range(W):
                    x = min(Bandwidth[0][i],Bandwidth[1][j],Bandwidth[2][k])
                    output += Paths[i*(W**2)+j*W+k]*x
        return output
               
        
    def Path_2_KLD_(self,Paths,Bandwidth,L):
        W = round(len(Paths)**(1/L))
        Sum = np.sum(Bandwidth[0])
        B_List1 = [Bandwidth[0][i]/Sum for i in range(W)]

        Sum = np.sum(Bandwidth[1])
        B_List2 = [Bandwidth[1][i]/Sum for i in range(W)]        

        Sum = np.sum(Bandwidth[2])
        B_List3 = [Bandwidth[2][i]/Sum for i in range(W)]
        
        List1 = []
        for i in range(W):
            List1.append(np.sum(Paths[(W**2)*i:(W**2)*(i+1)]))

        List2 = []
        for i in range(W):
            List_ = []
            for j in range(W):
                
                List_ += Paths[(W**2)*j+W*i:(W**2)*(j)+W*(i+1)]
            
            List2.append(np.sum(List_))
            
            
        List3 = []
        for i in range(W):
            
            List__ = []
            for j in range(W):
                for k in range(W):
                    List__.append(Paths[(W**2)*j+W*k+i])
            List3.append(np.sum(List__))   
            
        d1 = compute_metrics(B_List1,List1)
        d2 = compute_metrics(B_List2,List2)            
        d3 = compute_metrics(B_List3,List3)        
        return np.mean([d1,d2,d3])
        
        
        
    def Paths_2_matrix(self,Paths,L):
        #Get the transformation matrix from the paths probabilities and the P list
        W = round(len(Paths)**(1/L))
        #print(W,'ADDSAF')
        P_List = []
        for i in range(W):
            P_List.append(np.sum(Paths[i*(W**2):(i+1)*(W**2)]))
        Matrix1 = []
        Matrix_List = []
        for i in range(W):
            List = []
            for j in range(W):
                List.append((Paths[i*(W**2)+j*W:i*(W**2)+(j+1)*W]))
            List_ = To_list(np.sum(np.matrix(List),axis = 0))
            Matrix_List.append(List_)       
        for i in range(len(Matrix_List)):
            #print(np.matrix(Matrix_List[i]),'*****',P_List[i] )
            Matrix1.append(To_list(np.matrix(Matrix_List[i])/P_List[i]))
        
        return np.matrix(Matrix1),P_List

    def AL_EXP(self,Latency_List,D = False):
        Layers_Num = len(Latency_List)-1
        if not D:
            
            R_Path, Latency_ = self.EXP_Latency_C(Latency_List)
        else:
            R_Path, Latency_ = self.EXP_Latency_CD(Latency_List)
        
        Entropy_List = []
        Ave_Latency_List = []
        
        for item in R_Path:
            
            T,P = self.Paths_2_matrix(item,Layers_Num)
            
            Entropy_List.append(self.Entropy_Transformation(T,P))
            Ave_Latency_List.append(self.Ave_Latency(item,Latency_))
        
        return Entropy_List,Ave_Latency_List,R_Path,Latency_
            
            
    def Band_EXP(self,Band_List):
        Layers_Num = len(Band_List)
        R_Path= self.EXP_Band(Band_List)
        
        Entropy_List = []
        KVL_List = []
        
        for item in R_Path:
            
            T,P = self.Paths_2_matrix(item,Layers_Num)
            
            Entropy_List.append(self.Entropy_Transformation(T,P))
            KVL_List.append(self.Path_2_KLD(item,Band_List,Layers_Num))
        
        return Entropy_List,KVL_List,R_Path
    
    
    def JAR_EXP(self,List):
        R_Path= self.EXP_JAR(List)
        RR_Path = np.copy(R_Path)        
        Entropy_List = []
        for item in R_Path:  
            #print('0jljljl')
            #print(item,'kklkjllkjkl')
            #print('jkjl')
            P,GG1,GG2 = compute_conditionals(item) 
            T = GG1.dot(GG2)
            #print('mbjkkjjkjj')
            Entropy_List.append(self.Entropy_Transformation(T,P))
       
        return Entropy_List,RR_Path   
    
    
    
    
    
    
    
    
    
                
    def EXP_Latency_CD(self,Latency_List):
        EPS = self.EPS 
        Delta = (1/50) #Sensivity is 0.01 divided by 1 client
        (Nc,W) = np.shape(Latency_List[1])
        L = len(Latency_List)-2
        
        
        
        Scores = []

        Path = []#that is th elatency of paths thta are pur objects we consider one cleint 
        for i in range(W):

            for j in range(W):
                
                for k in range(W):
                    
                    Path.append(Latency_List[0][i]+Latency_List[1][i,j]+Latency_List[2][j,k]+Latency_List[3][k])
        for item in Path:
            Scores.append(-item)
            
        
        
        Paths_ = []
        for eps in EPS:
            Prob = []
            for term in Scores:
                Prob.append(math.exp((eps*term)/(2*Delta)))
            Sum = np.sum(Prob)
            dis = [Prob[i]/Sum for i in range(len(Prob))]
            Paths_.append(dis)

        return Paths_ ,Path            
    
    
    
    def EXP_Latency_C(self,Latency_List):
        EPS = self.EPS 
        Delta = (1/50) #Sensivity is 0.1 divided by 1 client
        (Nc,W) = np.shape(Latency_List[1])
        L = len(Latency_List)-2
        #print(W)
        
        Latency_des = np.mean(Latency_List[3])
        
        
        Scores = []

        Path = []#that is th elatency of paths thta are pur objects we consider one cleint 
        for i in range(W):

            for j in range(W):
                
                for k in range(W):
                    
                    Path.append(Latency_List[0][i]+Latency_List[1][i,j]+Latency_List[2][j,k]+Latency_des)
        for item in Path:
            Scores.append(-item)
            
        
        
        Paths_ = []
        for eps in EPS:
            Prob = []
            for term in Scores:
                Prob.append(math.exp((eps*term)/(2*Delta)))
            Sum = np.sum(Prob)
            dis = [Prob[i]/Sum for i in range(len(Prob))]
            Paths_.append(dis)
        #print(len(Paths_[0]),'l;lj;j;jjn;ojoj;')
        return Paths_ ,Path      
    
    
    
    
    
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



    def Entropy_Transformation(self,T,P):
        #print(len(T))
        (W,W) = np.shape(T)
        H = []
        for i in range(W):
            #print(P[i])
            List = []
            for k in range(W):
                List.append(T[i,k])
            L =[]
            for item in List:
                if item!=0:
                    L.append(item)
            l = np.sum(L)
            #print(P[i])
            for i in range(len(L)):
                L[i]=L[i]/l
            ent = 0
            #print(P[i])
            for item in L:
                ent = ent - item*(np.log(item)/np.log(2))
            #print(ent,P[i])
            H.append(ent)
        #print(H)
        return To_list(np.matrix(P).dot(H))[0]
    
    
    def Ave_Latency(self,Path,Latency_List):
        
        
        return To_list(np.matrix(Path).dot(Latency_List))[0]
        
    

    def EXP_Band(self,Bandwidth_List):
        
        
        EPS = self.EPS 
        Delta = (5/1) #Sensivity is 1 divided by 1 client
        
        L = len(Bandwidth_List)
        W = len(Bandwidth_List[0])
    
        Scores = []
    
        Path = []#that is th elatency of paths thta are pur objects we consider one cleint 
        for i in range(W):
    
            for j in range(W):
                
                for k in range(W):
                    
                    Path.append(Bandwidth_List[0][i]+Bandwidth_List[1][j]+Bandwidth_List[2][k])
        for item in Path:
            Scores.append(item)
            
        
        
        Paths_ = []
        for eps in EPS:
            Prob = []
            for term in Scores:
                Prob.append(math.exp((eps*term)/(2*Delta)))
            Sum = np.sum(Prob)
            dis = [Prob[i]/Sum for i in range(len(Prob))]
            Paths_.append(dis)
    
        return Paths_     
            
    def EXP_Band1(self,Bandwidth_List,eps):
        #EPS = self.EPS 
        Delta = (5/1) #Sensivity is 1 divided by 1 client
        
        L = len(Bandwidth_List)
        W = len(Bandwidth_List[0])
    
        Scores = []
    
        Path = []#that is th elatency of paths thta are pur objects we consider one cleint 
        for i in range(W):
    
            for j in range(W):
                
                for k in range(W):
                    
                    Path.append(Bandwidth_List[0][i]+Bandwidth_List[1][j]+Bandwidth_List[2][k])
        for item in Path:
            Scores.append(item)

        Prob = []
        for term in Scores:
            Prob.append(math.exp((eps*term)/(2*Delta)))
        Sum = np.sum(Prob)
        dis = [Prob[i]/Sum for i in range(len(Prob))]

        P_List = []
        for i in range(W):
            P_List.append(np.sum(dis[i*(W**2):(i+1)*(W**2)]))

        return dis, P_List   
    
  
    
    def EXP_Latency_CD1(self,Latency_List,eps):
        #EPS = self.EPS 
        Delta = (1/50) #Sensivity is 0.01 divided by 1 client
        (Nc,W) = np.shape(Latency_List[1])
        L = len(Latency_List)-2
        
        
        
        Scores = []

        Path = []#that is th elatency of paths thta are pur objects we consider one cleint 
        for i in range(W):

            for j in range(W):
                
                for k in range(W):
                    
                    Path.append(Latency_List[0][i]+Latency_List[1][i,j]+Latency_List[2][j,k]+Latency_List[3][k])
        for item in Path:
            Scores.append(-item)
            
        

        Prob = []
        for term in Scores:
            Prob.append(math.exp((eps*term)/(2*Delta)))
        Sum = np.sum(Prob)
        dis = [Prob[i]/Sum for i in range(len(Prob))]
        P_List = []
        for i in range(W):
            P_List.append(np.sum(dis[i*(W**2):(i+1)*(W**2)]))

        return dis, P_List        
    
    
    
    def EXP_Latency_C1(self,Latency_List,eps):
        #EPS = self.EPS 
        Delta = (1/50) #Sensivity is 0.1 divided by 1 client
        (Nc,W) = np.shape(Latency_List[1])
        L = len(Latency_List)-2
        #print(W)
        
        Latency_des = np.mean(Latency_List[3])
        
        
        Scores = []

        Path = []#that is th elatency of paths thta are pur objects we consider one cleint 
        for i in range(W):

            for j in range(W):
                
                for k in range(W):
                    
                    Path.append(Latency_List[0][i]+Latency_List[1][i,j]+Latency_List[2][j,k]+Latency_des)
        for item in Path:
            Scores.append(-item)
            
        
        
        Prob = []
        for term in Scores:
            Prob.append(math.exp((eps*term)/(2*Delta)))
        Sum = np.sum(Prob)
        dis = [Prob[i]/Sum for i in range(len(Prob))]
        P_List = []
        for i in range(W):
            P_List.append(np.sum(dis[i*(W**2):(i+1)*(W**2)]))

        return dis, P_List         
    


                          
    def EL_Analysis(self,Name_,Iteration):
        import numpy as np
        import json
        File_name = Name_        
        import os         
        if not os.path.exists(File_name):
            os.mkdir(os.path.join('', File_name))     
        Data1,Data2 = self.Routings(Iteration)
        
        latency_ = self.Analytic_Latency(Data1, Data2)
        
        entropy_ = self.Analytic_Entropy(Data2)



        from Plot import PLOT      

        X_L = r'$\tau$'
        Y_E = "Entropy (bits)"
        Y_L = 'Latency (sec)'
        D =  ['Proportional','LARMIX']    

        Alpha = self.Var

            
        Name_Entropy = File_name + '/' +'Entropy.png'
        Name_Latency = File_name + '/' + 'Latency.png'
        Name_t = File_name + '/' + 'Throughput.png'            
        PLT_E = PLOT(Alpha,entropy_,D,X_L,Y_E,Name_Entropy)
        PLT_E.scatter_line(True,7.5)
 
        PLT_L1 = PLOT(Alpha,latency_,D,X_L,Y_L,Name_Latency)

        PLT_L1.scatter_line(True,0.4)

        Throughput = np.matrix(entropy_)/np.matrix(latency_)  
        troughput = Throughput.tolist()
        PLT_T1 = PLOT(Alpha,troughput,D,X_L,'Throughput',Name_t)

        PLT_T1.scatter_line(True,400)        
        data_ = {}
        data_['Tau'] = Alpha
        i = 0
        for item in D:
            data_['Latency'+item] = latency_[i]
            data_['Entropy'+item] = entropy_[i] 
            i = i +1
        import pandas as pd        
        df = pd.DataFrame(data_)   
        
    def check_capacity(self,List, Cap):
        State = True
        for i in range(len(List)):
            if int(1000*List[i]) > int(1000*Cap[i]):
                State = False
                break
        return State
    def refine(self,List,Cap):
        
        for i in range(len(List)):
            
            if int(1000*List[i]) > int(1000*Cap[i]):
                List[i] = Cap[i]
        sum_ = np.sum(List)
        List_ = [List[i]/sum_ for i in range(len(List))]
        return List_
    
        
                
                
            
        
    def Fast_Balance(self,func,Matrix,Param):
        output = np.ones((self.W,self.W))
        Cap = [1]*self.W
        
        for i in range(self.W):
            
            List = func(To_list(Matrix[:,i]),Param)
            
            while not self.check_capacity(List,Cap):
                List = self.refine(List,Cap)
            Cap = subtract_lists(Cap, List)
            output[:,i] = List
            
        return output
    
    
    def BALD(self,Matrix_,Omega,Co,Iterations = 3):
        #print('BALD','8888888888888888888888888888','\n',Omega,Co,'&&&&&&&&&&&&&&&&')
        Matrix = Zero_Check(Matrix_)
        o1,o2 = np.shape(Matrix)
        Balanced_Matrix = np.copy(Matrix)
        
        for It in range(Iterations):
            #print(Balanced_Matrix)
            for j in range(o2):
                List_temp = Normalized(To_list(Balanced_Matrix[:,j]),Omega[j],Co)
                Balanced_Matrix[:,j] = List_temp
                
            for i in range(o1):
                List_temp = Normalized(To_list(Balanced_Matrix[i,:]),1,[1]*o1)
                Balanced_Matrix[i,:] = List_temp                
               
        #print('&&&&&&&',np.sum(Balanced_Matrix,axis=0))
        return Balanced_Matrix
    
    def Matrix_routing(self,fun,Matrix,Omega,Param):
        #fun: 'RST', 'REB', 'RLP'
        if fun == 'RLP':
            return self.Linear(Matrix,Omega,Param)
        
        else:
            Dis_Matrix = np.zeros((self.W,self.W))
            if fun == 'REB':
                for i in range(self.W):
                    List = To_list(Matrix[i,:])
                    dis = self.EXP_New(List,Omega,Param)
                    Dis_Matrix[i,:] = dis
            elif fun == 'RST':
                for i in range(self.W):
                    List = To_list(Matrix[i,:])
                    dis = self.alpha_closest(List,Omega,Param)
                    Dis_Matrix[i,:] = dis            
        return Dis_Matrix
            
###########################Latency measurements###############################
    def Latency_Measure(self,Latency_List,Routing_List,Path):
        L = 3
        n1,n2 = np.shape(Latency_List[0])
        
        x = 0
        for i in range(n1):
            for j in range(n1):
                for k in range(n2):
                    p = Path[i]*Routing_List[0][i,j]*Routing_List[1][j,k]
                    l = Latency_List[0][i,j]+Latency_List[1][j,k]
                    x+=p*l
                    #print(x,'[',i,j,k,']',p,l)
        return x
    
    
    def Bandwidth(self,List_R,Omega,P):
        w_List = []
        W = len(List_R[0])
        I = np.zeros((len(List_R[0]),len(List_R[0])))
        for i1 in range(len(List_R[0])):
            I[i1,i1] = 1  
            
        for k in range(len(List_R)+1):
            if k==0:
                for j in range(W):
                    w_List.append(round((P[j]*W)/Omega[k][j]*10)/10)
                    #print(((P[j]*W)/Omega[k][j]))
            else:
                Matrix  = np.copy(I)
                for _ in range(k):
                    Matrix = Matrix.dot(List_R[_])
                Temp = To_list(W*np.matrix(P).dot(Matrix))
                #print(Temp)
                Temp_ = []
                for j_1 in range(len(Temp)):
                    Temp_.append(round((Temp[j_1]/Omega[k][j_1])*10)/10)
                   
                
                w_List = w_List + Temp_
                    
        E = [i/10 for i in range(51)]
        
        return compute_cdf(w_List, E)
    
    def Bandwidth_(self,List_R,Omega,P):
        w_List = []
        W = len(List_R[0])
        I = np.zeros((len(List_R[0]),len(List_R[0])))
        for i1 in range(len(List_R[0])):
            I[i1,i1] = 1  
            
        for k in range(len(List_R)+1):
            if k==0:
                for j in range(W):
                    xx = round((P[j]*W)/Omega[k][j]*10)/10
                    if xx>1:
                        w_List.append(xx-1)
                    #print(((P[j]*W)/Omega[k][j]))
            else:
                Matrix  = np.copy(I)
                for _ in range(k):
                    Matrix = Matrix.dot(List_R[_])
                Temp = To_list(W*np.matrix(P).dot(Matrix))
                #print(Temp)
                Temp_ = []
                for j_1 in range(len(Temp)):
                    Temp_.append(round((Temp[j_1]/Omega[k][j_1])*10)/10)
                   
                    #print(Omega[k][j_1])
                for item in Temp_:
                    xx = item
                    if xx>1:
                        w_List.append(xx-1)                   
        
        if len(w_List)==0:
            return 0
           
        x = np.mean(w_List)
        
        return x
    
    def R1_R2_(self,Path):
        
        W = round(len(Path)**(1/3))
        P = []
        for i in range(W):
            P.append(np.sum(Path[i*(W**2):(i+1)*(W**2)]))
        R = []
        for i in range(W):
            List1 = Path[i*(W**2):(i+1)*(W**2)]
            #print(List1)
            R.append([np.sum(List1[j*W:(j+1)*W])/P[j] for j in range(W)])
        R1 = To_list(np.matrix(R))
        
        
        for i in range(W):
            for j in range(W):
                if int((10*10)*R1[i][j]) == 0:
                    R1[i][j] = 1e-10
                    

        RR = np.zeros((W,W))
        for i in range(W):
            List1 = Path[i*(W**2):(i+1)*(W**2)]
            #print(List1)
            RR += np.matrix([To_list(np.matrix(List1[j*W:(j+1)*W])/R1[i][j]) for j in range(W)])

        R2 = To_list(RR)
        
        #print(np.sum(P),np.sum(R1)/W,np.sum(R2)/W)

        return R1,R2, P   




    def R1_R2(self,Path_probability):
        W = round(len(Path_probability)**(1/3))
        Path_probability = np.array(Path_probability).reshape(W, W, W)  # Reshape into 3D array
        
        # Probability of selecting each node in Layer 1
        P_layer1 = np.sum(Path_probability, axis=(1, 2))  # Summing over Layer 2 and 3 choices
        P_layer1 /= np.sum(P_layer1)  # Normalize
        
        # Probability matrix of selecting Layer 2 node given Layer 1 node
        P_layer2_given_layer1 = np.sum(Path_probability, axis=2)  # Sum over Layer 3 choices
        P_layer2_given_layer1 /= P_layer2_given_layer1.sum(axis=1, keepdims=True)  # Normalize rows
        
        # Probability matrix of selecting Layer 3 node given Layer 2 node
        P_layer3_given_layer2 = np.sum(Path_probability, axis=0)  # Sum over Layer 1 choices
        P_layer3_given_layer2 /= P_layer3_given_layer2.sum(axis=1, keepdims=True)  # Normalize rows
        
        R1 = To_list(P_layer2_given_layer1)
        R2 = To_list(P_layer3_given_layer2)
        return R1,R2, P_layer1
    
    
    def EXP_JAR(self,List):       
        EPS = self.EPS 
        Delta = (self.L-1) #Sensivity is 1 divided by 1 client       
        W = round(len(List)**(1/self.L))    
        Scores = List.copy()       
        Paths_ = []
        for eps in EPS:
            Prob = []
            for term in Scores:
                Prob.append(math.exp((eps*term)/(2*Delta)))
            Sum = np.sum(Prob)
            dis = [Prob[i]/Sum for i in range(len(Prob))]
            Paths_.append(dis)  
        return Paths_       



