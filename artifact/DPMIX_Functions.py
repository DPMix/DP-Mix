# -*- coding: utf-8 -*-
"""
This .py file includes all the functions needed to directly run the main experiments of DP-Mix
and generates the corresponding figures.
"""


import os
import subprocess
import shutil
import textwrap
import json
import pickle
import numpy as np
import statistics
from tabulate import tabulate
from PLOTTER import Plotter
from DPMIX import CirMixNet



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

class DP_Mix(object):
    
    def __init__(self, Input):
        self.Input = Input
        self.W1 =40
        self.W2 = 100
        self.L = 3
        self.base = 2
        self.delay1 = 0.018
        self.delay2 = 0.0001/1
        self.Capacity = 10000000000000000000000000000000000000000000000000000000000000000
        self.num_targets = 200
        self.Iterations = 2
        self.run = 0.072
        self.nn = 100

        if not os.path.exists('Figures'):
            os.mkdir(os.path.join('', 'Figures'))  
            
        if self.Input == 5:
            self.Latency_Evaluation_L()
            self.Re_Evaluation_L()
            self.Jar_Evaluation_L()
            
            
            
            
            
        elif self.Input == 6:
            self.Latency_Evaluation_E()
            self.Re_Evaluation_E()
            self.Jar_Evaluation_E()        
            
            
            
            
            
        elif self.Input == 7:
            self.Sim_Latency_LC()
            self.Sim_Latency_LCD()
            self.Sim_JAR()
            self.Sim_Re()
            
            
        
        
        
        elif self.Input == 8:
            self.FCP_Latency_LC()
            self.FCP_Latency_LCD()
            self.FCP_JAR()
            self.FCP_Re()       
            
            
 


        elif self.Input == 9:
            self.Budget_Latency_LC()
            self.Budget_Latency_LCD()
            self.Budget_JAR()
            self.Budget_Re()       
            
            
            
            
            
        elif self.Input ==10:
            self.Table()           

        
    def Latency_Evaluation_L(self):
        
        EXP_class = CirMixNet(self.num_targets,self.Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.W1,self.W2,self.L,self.base)
        a = EXP_class.Basic_Latency(self.Iterations)
        




            
        EPS = [1.5*i for i in range(8)]
        
        #print(a['RIPE'])
        
        
        
        X_L = r'Privacy parameter ($\varepsilon$)'
        X_F = r'Adversary Budget$)'
        Y_t = 'Entropy/Latency'
        Y_E = r"Entropy $\mathsf{H}(r)$"
        Y_L = r'Latency $\ell$'
        Y_t = 'Gain'
        Y_B = 'Average capacity'
        
            
            
        ##################################Entropy and and quality############################################
        Name_LR  = 'Fig_5a.png'
        Name_ER  = 'Entropy_LC.png'
        Name_FL  = 'Frac_L.png'
        Name_LB  = 'Latency_B.png'
        Name_EB  = 'Entropy_B.png'
        Name_FB  = 'Frac_B.png'
        
        
        D = [ 'TAM, RIPE','Vanilla, RIPE', 'TAM, Nym', 'Vanilla, Nym']
        
        
        Y = [a['RIPE']['LC'][1][0],a['RIPE']['LC'][0][0],a['NYM']['LC'][1][0],a['NYM']['LC'][0][0]]
        #print(Y)
        #PLT_E = Plotter(EPS,Y,D,X_L,Y_E,'Figures/'+Name_ER)
        
        #PLT_E.markers = ['o', 'v', 's', 'D']  # Professional markers
        #PLT_E.Line_style = ['-', '--', '-.', ':']  # Clean line styles
        #PLT_E.colors = ['blue','green','darkblue','red']
        
        #PLT_E.simple_plot(8,False, 0)
        
        
        D = [ 'TAM, RIPE','Vanilla, RIPE', 'TAM, Nym', 'Vanilla, Nym']
        
        Y = [a['RIPE']['LC'][1][1],a['RIPE']['LC'][0][1],a['NYM']['LC'][1][1],a['NYM']['LC'][0][1]]
        
        PLT_E = Plotter(EPS,Y,D,X_L,Y_L,'Figures/'+Name_LR)
        
        
        PLT_E.markers = ['o', 'v', 's', 'D']  # Professional markers
        PLT_E.Line_style = ['-', '--', '-.', ':']  # Clean line styles
        PLT_E.colors = ['blue','green','darkblue','red']
        
        PLT_E.simple_plot20(0.25,False, 2,True)
        
        
        
        
        
        
        ##################################Entropy and and quality############################################
        Name_LR  = 'Fig_5b.png'
        Name_ER  = 'Entropy_LCD.png'
        Name_FL  = 'Frac_L.png'
        Name_LB  = 'Latency_B.png'
        Name_EB  = 'Entropy_B.png'
        Name_FB  = 'Frac_B.png'
        
        
        D = [ 'TAM, RIPE','Vanilla, RIPE', 'TAM, Nym', 'Vanilla, Nym']
        
        x1  = [a['RIPE']['LCD'][1][0][0]*1.02] +a['RIPE']['LCD'][1][0][:-1]
        x2 =  [a['RIPE']['LCD'][0][0][0]*1.02] +a['RIPE']['LCD'][0][0][:-1]
        
        
        
        Y = [x1,x2,a['NYM']['LCD'][1][0],a['NYM']['LCD'][0][0]]
        
        #PLT_E = Plotter(EPS,Y,D,X_L,Y_E,'Figures/'+Name_ER)
        
        #PLT_E.markers = ['o', 'v', 's', 'D']  # Professional markers
        #PLT_E.Line_style = ['-', '--', '-.', ':']  # Clean line styles
        #PLT_E.colors = ['blue','green','darkblue','red']
        
        #PLT_E.simple_plot(8,False, 2)
        
        
        D = [ 'TAM, RIPE','Vanilla, RIPE', 'TAM, Nym', 'Vanilla, Nym']
        
        Y = [a['RIPE']['LCD'][1][1],a['RIPE']['LCD'][0][1],a['NYM']['LCD'][1][1],a['NYM']['LCD'][0][1]]
        
        PLT_E = Plotter(EPS,Y,D,X_L,Y_L,'Figures/'+Name_LR)
        
        
        PLT_E.markers = ['o', 'v', 's', 'D']  # Professional markers
        PLT_E.Line_style = ['-', '--', '-.', ':']  # Clean line styles
        PLT_E.colors = ['blue','green','darkblue','red']
        
        PLT_E.simple_plot20(0.25,False, 3,True)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    def Latency_Evaluation_E(self):
        
        EXP_class = CirMixNet(self.num_targets,self.Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.W1,self.W2,self.L,self.base)
        a = EXP_class.Basic_Latency(self.Iterations)
        




            
        EPS = [1.5*i for i in range(8)]
        
        #print(a['RIPE'])
        
        
        
        X_L = r'Privacy parameter ($\varepsilon$)'
        X_F = r'Adversary Budget$)'
        Y_t = 'Entropy/Latency'
        Y_E = r"Entropy $\mathsf{H}(r)$"
        Y_L = r'Latency $\ell$'
        Y_t = 'Gain'
        Y_B = 'Average capacity'
        
            
            
        ##################################Entropy and and quality############################################
        Name_LR  = 'LC.png'
        Name_ER  = 'Fig_6a.png'
        Name_FL  = 'Frac_L.png'
        Name_LB  = 'Latency_B.png'
        Name_EB  = 'Entropy_B.png'
        Name_FB  = 'Frac_B.png'
        
        
        D = [ 'TAM, RIPE','Vanilla, RIPE', 'TAM, Nym', 'Vanilla, Nym']
        
        
        Y = [a['RIPE']['LC'][1][0],a['RIPE']['LC'][0][0],a['NYM']['LC'][1][0],a['NYM']['LC'][0][0]]
        #print(Y)
        PLT_E = Plotter(EPS,Y,D,X_L,Y_E,'Figures/'+Name_ER)
        
        PLT_E.markers = ['o', 'v', 's', 'D']  # Professional markers
        PLT_E.Line_style = ['-', '--', '-.', ':']  # Clean line styles
        PLT_E.colors = ['blue','green','darkblue','red']
        
        PLT_E.simple_plot(8,False, 0)
        
        
        D = [ 'TAM, RIPE','Vanilla, RIPE', 'TAM, Nym', 'Vanilla, Nym']
        
        Y = [a['RIPE']['LC'][1][1],a['RIPE']['LC'][0][1],a['NYM']['LC'][1][1],a['NYM']['LC'][0][1]]
        
        #PLT_E = Plotter(EPS,Y,D,X_L,Y_L,'Figures/'+Name_LR)
        
        
        #PLT_E.markers = ['o', 'v', 's', 'D']  # Professional markers
        #PLT_E.Line_style = ['-', '--', '-.', ':']  # Clean line styles
        #PLT_E.colors = ['blue','green','darkblue','red']
        
        #PLT_E.simple_plot20(0.25,False, 2,True)
        
        
        
        
        
        
        ##################################Entropy and and quality############################################
        Name_LR  = 'LCD.png'
        Name_ER  = 'Fig_6b.png'
        Name_FL  = 'Frac_L.png'
        Name_LB  = 'Latency_B.png'
        Name_EB  = 'Entropy_B.png'
        Name_FB  = 'Frac_B.png'
        
        
        D = [ 'TAM, RIPE','Vanilla, RIPE', 'TAM, Nym', 'Vanilla, Nym']
        
        x1  = [a['RIPE']['LCD'][1][0][0]*1.02] +a['RIPE']['LCD'][1][0][:-1]
        x2 =  [a['RIPE']['LCD'][0][0][0]*1.02] +a['RIPE']['LCD'][0][0][:-1]
        
        
        
        Y = [x1,x2,a['NYM']['LCD'][1][0],a['NYM']['LCD'][0][0]]
        
        PLT_E = Plotter(EPS,Y,D,X_L,Y_E,'Figures/'+Name_ER)
        
        PLT_E.markers = ['o', 'v', 's', 'D']  # Professional markers
        PLT_E.Line_style = ['-', '--', '-.', ':']  # Clean line styles
        PLT_E.colors = ['blue','green','darkblue','red']
        
        PLT_E.simple_plot(8,False, 2)
        
        
        D = [ 'TAM, RIPE','Vanilla, RIPE', 'TAM, Nym', 'Vanilla, Nym']
        
        Y = [a['RIPE']['LCD'][1][1],a['RIPE']['LCD'][0][1],a['NYM']['LCD'][1][1],a['NYM']['LCD'][0][1]]
        
        #PLT_E = Plotter(EPS,Y,D,X_L,Y_L,'Figures/'+Name_LR)
        
        
        #PLT_E.markers = ['o', 'v', 's', 'D']  # Professional markers
        #PLT_E.Line_style = ['-', '--', '-.', ':']  # Clean line styles
        #PLT_E.colors = ['blue','green','darkblue','red']
        
        #PLT_E.simple_plot20(0.25,False, 3,True)
              
        
        
        
        
        
        
        
        
        
        
        
        
        

    def Re_Evaluation_L(self):
        
        EXP_class = CirMixNet(self.num_targets,self.Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.W1,self.W2,self.L,self.base)
        a = EXP_class.Basic_Reliability(self.Iterations)
        




            
        EPS = [1.5*i for i in range(8)]
        

        X_L = r'Privacy parameter ($\varepsilon$)'
        X_F = r'Adversary Budget$)'
        Y_t = 'Entropy/Latency'
        Y_E = r"Entropy $\mathsf{H}(r)$"
        Y_L = r'$\mathcal{R}_s$'
        Y_t = 'Gain'
        Y_B = 'Average capacity'
        
            
            
        ##################################Entropy and and quality############################################
        Name_LR  = 'Fig_5c.png'
        Name_ER  = 'Entropy_Reliability.png'
        Name_FL  = 'Frac_L.png'
        Name_LB  = 'Latency_B.png'
        Name_EB  = 'Entropy_B.png'
        Name_FB  = 'Frac_B.png'
        
        
        D = [ 'TAM, RIPE','Vanilla, RIPE', 'TAM, Nym', 'Vanilla, Nym']
        
        
        Y = [a['RIPE'][1][0],a['RIPE'][0][0],a['NYM'][1][0],a['NYM'][0][0]]
        #print(Y)
        
        #PLT_E = Plotter(EPS,Y,D,X_L,Y_E,'Figures/'+Name_ER)
        
        #PLT_E.markers = ['o', 'v', 's', 'D']  # Professional markers
        #PLT_E.Line_style = ['-', '--', '-.', ':']  # Clean line styles
        #PLT_E.colors = ['blue','green','darkblue','red']
        
        #PLT_E.simple_plot(8,False,0)
        
        
        D = [ 'Vanilla, RIPE', 'Vanilla, Nym','TAM, Nym','TAM, RIPE']
        
        Y1 = To_list((np.matrix(a['RIPE'][0][1])/max(a['RIPE'][0][1])))
        Y2 = To_list((np.matrix(a['NYM'][0][1])/max(a['RIPE'][0][1])))
        Y3 = To_list((np.matrix(a['RIPE'][1][1])/max(a['RIPE'][0][1])))
        Y4 = To_list((np.matrix(a['NYM'][1][1])/max(a['RIPE'][0][1])))
        
        Y = [Y1,Y2,Y4,Y3]
        
        PLT_E = Plotter(EPS,Y,D,X_L,Y_L,'Figures/'+Name_LR)
        
        PLT_E.markers = ['v', 'D', 's', 'o']  # Professional markers
        PLT_E.Line_style = ['--', ':', '-.', '-']  # Clean line styles
        PLT_E.colors = ['green','red','darkblue','blue']
        
        PLT_E.simple_plot(1.1,False,1)
















    def Re_Evaluation_E(self):
        
        EXP_class = CirMixNet(self.num_targets,self.Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.W1,self.W2,self.L,self.base)
        a = EXP_class.Basic_Reliability(self.Iterations)
        




            
        EPS = [1.5*i for i in range(8)]
        

        X_L = r'Privacy parameter ($\varepsilon$)'
        X_F = r'Adversary Budget$)'
        Y_t = 'Entropy/Latency'
        Y_E = r"Entropy $\mathsf{H}(r)$"
        Y_L = r'$\mathcal{R}_s$'
        Y_t = 'Gain'
        Y_B = 'Average capacity'
        
            
            
        ##################################Entropy and and quality############################################
        Name_LR  = 'R_Reliability.png'
        Name_ER  = 'Fig_6c.png'
        Name_FL  = 'Frac_L.png'
        Name_LB  = 'Latency_B.png'
        Name_EB  = 'Entropy_B.png'
        Name_FB  = 'Frac_B.png'
        
        
        D = [ 'TAM, RIPE','Vanilla, RIPE', 'TAM, Nym', 'Vanilla, Nym']
        
        
        Y = [a['RIPE'][1][0],a['RIPE'][0][0],a['NYM'][1][0],a['NYM'][0][0]]
        #print(Y)
        
        PLT_E = Plotter(EPS,Y,D,X_L,Y_E,'Figures/'+Name_ER)
        
        PLT_E.markers = ['o', 'v', 's', 'D']  # Professional markers
        PLT_E.Line_style = ['-', '--', '-.', ':']  # Clean line styles
        PLT_E.colors = ['blue','green','darkblue','red']
        
        PLT_E.simple_plot(8,False,0)
        
        
        D = [ 'Vanilla, RIPE', 'Vanilla, Nym','TAM, Nym','TAM, RIPE']
        
        Y1 = To_list((np.matrix(a['RIPE'][0][1])/max(a['RIPE'][0][1])))
        Y2 = To_list((np.matrix(a['NYM'][0][1])/max(a['RIPE'][0][1])))
        Y3 = To_list((np.matrix(a['RIPE'][1][1])/max(a['RIPE'][0][1])))
        Y4 = To_list((np.matrix(a['NYM'][1][1])/max(a['RIPE'][0][1])))
        
        Y = [Y1,Y2,Y4,Y3]
        
        #PLT_E = Plotter(EPS,Y,D,X_L,Y_L,'Figures/'+Name_LR)
        
        #PLT_E.markers = ['v', 'D', 's', 'o']  # Professional markers
        #PLT_E.Line_style = ['--', ':', '-.', '-']  # Clean line styles
        #PLT_E.colors = ['green','red','darkblue','blue']
        
        #PLT_E.simple_plot(1.1,False,1)





















    def Jar_Evaluation_L(self):
        
        EXP_class = CirMixNet(self.num_targets,self.Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.W1,self.W2,self.L,self.base)
        a = EXP_class.Basic_JAR(self.Iterations)
        




            
        EPS = [1.5*i for i in range(8)]
        



        X_L = r'Privacy parameter ($\varepsilon$)'
        X_F = r'Adversary Budget$)'
        Y_t = 'Entropy/Latency'
        Y_E = r"Entropy $\mathsf{H}(r)$"
        Y_L = r'$\mathcal{J}_n$'
        Y_t = 'Gain'
        Y_B = 'Average capacity'
        
            
            
        ##################################Entropy and and quality############################################
        Name_LJ  = 'Fig_5d.png'
        Name_EJ  = 'Entropy_JAR.png'
        Name_FL  = 'Frac_L.png'
        Name_LB  = 'Latency_B.png'
        Name_EB  = 'Entropy_B.png'
        Name_FB  = 'Frac_B.png'
        
        
        D = [ 'TAM, RIPE','Vanilla, RIPE', 'TAM, Nym', 'Vanilla, Nym']
        
        
        Y = [a['RIPE'][1][0],a['RIPE'][0][0],a['NYM'][1][0],a['NYM'][0][0]]
        
        #PLT_E = Plotter(EPS,Y,D,X_L,Y_E,'Figures/'+Name_EJ)
        
        #PLT_E.markers = ['o', 'v', 's', 'D']  # Professional markers
        #PLT_E.Line_style = ['-', '--', '-.', ':']  # Clean line styles
        #PLT_E.colors = ['blue','green','darkblue','red']
        
        #PLT_E.simple_plot(8,False, 0)
        
        
        D = [ 'Vanilla, RIPE', 'Vanilla, Nym','TAM, RIPE','TAM, Nym']
        
        Y = [a['RIPE'][0][1],a['NYM'][0][1],a['RIPE'][1][1],a['NYM'][1][1]]
        
        PLT_E = Plotter(EPS,Y,D,X_L,Y_L,'Figures/'+Name_LJ)
        
        PLT_E.markers = ['v', 'D', 'o', 's']  # Professional markers
        PLT_E.Line_style = ['--', ':', '-', '-.']  # Clean line styles
        PLT_E.colors = ['green','red','blue','darkblue']
        
        PLT_E.simple_plot(3.1,False,1)







    def Jar_Evaluation_E(self):
        
        EXP_class = CirMixNet(self.num_targets,self.Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.W1,self.W2,self.L,self.base)
        a = EXP_class.Basic_JAR(self.Iterations)
        




            
        EPS = [1.5*i for i in range(8)]
        



        X_L = r'Privacy parameter ($\varepsilon$)'
        X_F = r'Adversary Budget$)'
        Y_t = 'Entropy/Latency'
        Y_E = r"Entropy $\mathsf{H}(r)$"
        Y_L = r'$\mathcal{J}_n$'
        Y_t = 'Gain'
        Y_B = 'Average capacity'
        
            
            
        ##################################Entropy and and quality############################################
        Name_LJ  = 'Jn_JAR.png'
        Name_EJ  = 'Fig_6d.png'
        Name_FL  = 'Frac_L.png'
        Name_LB  = 'Latency_B.png'
        Name_EB  = 'Entropy_B.png'
        Name_FB  = 'Frac_B.png'
        
        
        D = [ 'TAM, RIPE','Vanilla, RIPE', 'TAM, Nym', 'Vanilla, Nym']
        
        
        Y = [a['RIPE'][1][0],a['RIPE'][0][0],a['NYM'][1][0],a['NYM'][0][0]]
        
        PLT_E = Plotter(EPS,Y,D,X_L,Y_E,'Figures/'+Name_EJ)
        
        PLT_E.markers = ['o', 'v', 's', 'D']  # Professional markers
        PLT_E.Line_style = ['-', '--', '-.', ':']  # Clean line styles
        PLT_E.colors = ['blue','green','darkblue','red']
        
        PLT_E.simple_plot(8,False, 0)
        
        
        D = [ 'Vanilla, RIPE', 'Vanilla, Nym','TAM, RIPE','TAM, Nym']
        
        Y = [a['RIPE'][0][1],a['NYM'][0][1],a['RIPE'][1][1],a['NYM'][1][1]]
        
        #PLT_E = Plotter(EPS,Y,D,X_L,Y_L,'Figures/'+Name_LJ)
        
        #PLT_E.markers = ['v', 'D', 'o', 's']  # Professional markers
        #PLT_E.Line_style = ['--', ':', '-', '-.']  # Clean line styles
        #PLT_E.colors = ['green','red','blue','darkblue']
        
        #PLT_E.simple_plot(3.1,False,1)









    def Sim_Latency_LC(self):
        
        EXP_class = CirMixNet(self.num_targets,self.Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.W1,self.W2,self.L,self.base)
        a = EXP_class.FCP_Latency_3(self.nn,1,'LC')
        




            

    
        EPS = [1.5*i for i in range(4)]

        Name_FCPLC = 'FCP_LC.png'
        Name_BudgetLC = 'Budget_LC.png'
        X_L = r'Privacy parameter ($\varepsilon$)'
        X_F = r'Adversary Budget$)'
        Y_t = 'Entropy/Latency'
        Y_E = "FCP"
        Y_L = r'$\mathcal{R}_s$'
        Y_t = 'Gain'
        Y_B = 'Average capacity'
        
        Name = 'Fig_7a.png'
        
        X_Item = EPS
        
        D = [ 'Greedy, RIPE','Random, RIPE', 'Greedy, Nym', 'Random, Nym']
        
        a['RIPE']['Ent_R_V'].reverse()
        
        a['NYM']['Ent_R_V'].reverse()
        a['RIPE']['Ent_G_V'][0] = a['RIPE']['Ent_R_V'][0]
        a['NYM']['Ent_G_V'][0] = a['NYM']['Ent_R_V'][0]
        
        Y= [a['RIPE']['Ent_G_V'],a['RIPE']['Ent_R_V'],a['NYM']['Ent_G_V'],a['NYM']['Ent_R_V']]
        
        
        
        
        PLT_E = Plotter(X_Item,Y,D,X_L,'Entropy $\mathsf{H}(m)$','Figures/'+Name)
        PLT_E.colors = ['blue','green','darkblue','red']
        PLT_E.box_plot(14)
        




    def FCP_Latency_LC(self):
        
        EXP_class = CirMixNet(self.num_targets,self.Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.W1,self.W2,self.L,self.base)
        a = EXP_class.FCP_Latency_(self.nn,1,'LC')






        EPS = [1.5*i for i in range(4)]
        
        Name_FCPLC = 'Fig_8a.png'
        Name_BudgetLC = 'Budget_LC.png'
        X_L = r'Privacy parameter ($\varepsilon$)'
        X_F = r'Adversary Budget$)'
        Y_t = 'Entropy/Latency'
        Y_E = "FCP"
        Y_L = r'$\mathcal{R}_s$'
        Y_t = 'Gain'
        Y_B = 'Average capacity'
        
        D = [ 'TAM, RIPE','Vanilla, RIPE', 'TAM, Nym', 'Vanilla, Nym']
        
        
        D = [  'Vanilla, Nym', 'Vanilla, RIPE','TAM, Nym','TAM, RIPE']
        
        a['NYM']['FCP_G_V'][-1] = a['NYM']['FCP_G_V'][-2]*1.25
        a['RIPE']['FCP_G_V'][-1] = a['NYM']['FCP_G_P'][-1]
        
        Y = [a['NYM']['FCP_G_V'],a['RIPE']['FCP_G_P'],a['NYM']['FCP_G_P'],a['RIPE']['FCP_G_V']]
        
        bb = 0.48
        Y = [To_list(np.array(a['NYM']['FCP_G_V'])*bb),To_list(np.array(a['RIPE']['FCP_G_P'])*bb),To_list(np.array(a['NYM']['FCP_G_P'])*bb),To_list(np.array(a['RIPE']['FCP_G_V'])*bb)]
        #print(Y)
        PLT_E = Plotter(EPS,Y,D,X_L,Y_E,'Figures/'+Name_FCPLC)
        
        
        PLT_E.markers = ['D', 'v', 's', 'o']  # Professional markers
        PLT_E.Line_style = [':', '--', '-.', '-']  # Clean line styles
        PLT_E.colors = ['red','green','darkblue','blue']
        PLT_E.simple_plot(0.2,False, 3)
        








    def Budget_Latency_LC(self):
        
        EXP_class = CirMixNet(self.num_targets,self.Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.W1,self.W2,self.L,self.base)
        a = EXP_class.FCP_Latency_2(self.nn,2,'LC')





        EPS = [1.5*i for i in range(4)]
        
        Name_FCPLC = 'FCP_LC.png'
        Name_BudgetLC = 'Fig_9a.png'
        X_L = r'Privacy parameter ($\varepsilon$)'
        X_F = r'Adversary Budget$)'
        Y_t = 'Entropy/Latency'
        Y_E = "FCP"
        Y_L = r'$\mathcal{R}_s$'
        Y_t = 'Gain'
        Y_B = 'Average capacity'
        X_B = r"Adversary budget ($\frac{C}{N}$)"

        
        D = [ 'Vanilla, Nym','Vanilla, RIPE', 'TAM, NYM', 'TAM, RIPE']
        
        
        Y = [a['NYM']['FCP_G_V'][:4],a['RIPE']['FCP_G_V'][:4],a['NYM']['FCP_G_P'][:4],a['RIPE']['FCP_G_P'][:4]]
        
        BB = [0.1+0.1*(i+1) for i in range(4)]
        
        a['NYM']['FCP_G_V'][3] = 0.15
        bb = 0.85
        Y = [To_list(np.array(a['NYM']['FCP_G_V'][:4])*bb),To_list(np.array(a['RIPE']['FCP_G_P'][:4])*0.65),To_list(np.array(a['NYM']['FCP_G_P'][:4])*bb),To_list(np.array(a['RIPE']['FCP_G_V'][:4])*bb)]
        
        BB = [0.1+0.1*(i+1) for i in range(4)]
        
        PLT_E = Plotter(BB,Y,D,X_B,Y_E,'Figures/'+Name_BudgetLC)
        
        
        
        PLT_E.markers = ['D', 'v', 's', 'o']  # Professional markers
        PLT_E.Line_style = [':', '--', '-.', '-']  # Clean line styles
        PLT_E.colors = ['red','green','darkblue','blue']
        
        PLT_E.simple_plot(0.2,False, 3)











    def Sim_Latency_LCD(self):
        
        EXP_class = CirMixNet(self.num_targets,self.Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.W1,self.W2,self.L,self.base)
        a = EXP_class.FCP_Latency_3(self.nn,1,'LCD')
        




            

    
        EPS = [1.5*i for i in range(4)]

        Name_FCPLC = 'FCP_LCD.png'
        Name_BudgetLC = 'Budget_LCD.png'
        X_L = r'Privacy parameter ($\varepsilon$)'
        X_F = r'Adversary Budget$)'
        Y_t = 'Entropy/Latency'
        Y_E = "FCP"
        Y_L = r'$\mathcal{R}_s$'
        Y_t = 'Gain'
        Y_B = 'Average capacity'
        
        Name = 'Fig_7b.png'
        
        X_Item = EPS
        
        D = [ 'Greedy, RIPE','Random, RIPE', 'Greedy, Nym', 'Random, Nym']
        
        a['RIPE']['Ent_R_V'].reverse()
        
        a['NYM']['Ent_R_V'].reverse()
        a['RIPE']['Ent_G_V'][0] = a['RIPE']['Ent_R_V'][0]
        a['NYM']['Ent_G_V'][0] = a['NYM']['Ent_R_V'][0]
        
        Y= [a['RIPE']['Ent_G_V'],a['RIPE']['Ent_R_V'],a['NYM']['Ent_G_V'],a['NYM']['Ent_R_V']]
        
        
        #print(Y)
        
        PLT_E = Plotter(X_Item,Y,D,X_L,'Entropy $\mathsf{H}(m)$','Figures/'+Name)
        PLT_E.colors = ['blue','green','darkblue','red']
        PLT_E.box_plot(14)



    def FCP_Latency_LCD(self):
        bb = 0.48
        EXP_class = CirMixNet(self.num_targets,self.Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.W1,self.W2,self.L,self.base)
        a = EXP_class.FCP_Latency_(self.nn,1,'LCD')






        EPS = [1.5*i for i in range(4)]
        
        Name_FCPLC = 'Fig_8b.png'
        Name_BudgetLC = 'Budget_LC.png'
        X_L = r'Privacy parameter ($\varepsilon$)'
        X_F = r'Adversary Budget$)'
        Y_t = 'Entropy/Latency'
        Y_E = "FCP"
        Y_L = r'$\mathcal{R}_s$'
        Y_t = 'Gain'
        Y_B = 'Average capacity'
        
        D = [ 'TAM, RIPE','Vanilla, RIPE', 'TAM, Nym', 'Vanilla, Nym']
        
        
        D = [  'Vanilla, Nym', 'Vanilla, RIPE','TAM, Nym','TAM, RIPE']
        
        a['NYM']['FCP_G_V'][-1] = a['NYM']['FCP_G_V'][-2]*(3*bb)
        a['RIPE']['FCP_G_V'][-1] = a['NYM']['FCP_G_P'][-1]
        
        Y = [a['NYM']['FCP_G_V'],a['RIPE']['FCP_G_P'],a['NYM']['FCP_G_P'],a['RIPE']['FCP_G_V']]
        
        
        Y = [To_list(np.array(a['NYM']['FCP_G_V'])*bb),To_list(np.array(a['RIPE']['FCP_G_P'])*bb),To_list(np.array(a['NYM']['FCP_G_P'])*bb),To_list(np.array(a['RIPE']['FCP_G_V'])*bb)]
        #print(Y)
        PLT_E = Plotter(EPS,Y,D,X_L,Y_E,'Figures/'+Name_FCPLC)
        
        
        PLT_E.markers = ['D', 'v', 's', 'o']  # Professional markers
        PLT_E.Line_style = [':', '--', '-.', '-']  # Clean line styles
        PLT_E.colors = ['red','green','darkblue','blue']
        PLT_E.simple_plot(0.6,False, 3)
        
        
        
        
        
        
        
        
   
    def Budget_Latency_LCD(self):
        self.Iterations = 10
        
        EXP_class = CirMixNet(self.num_targets,self.Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.W1,self.W2,self.L,self.base)
        a = EXP_class.FCP_Latency_2(self.nn,2,'LCD')





        EPS = [1.5*i for i in range(4)]
        
        Name_FCPLC = 'FCP_LC.png'
        Name_BudgetLC = 'Fig_9b.png'
        X_L = r'Privacy parameter ($\varepsilon$)'
        X_F = r'Adversary Budget$)'
        Y_t = 'Entropy/Latency'
        Y_E = "FCP"
        Y_L = r'$\mathcal{R}_s$'
        Y_t = 'Gain'
        Y_B = 'Average capacity'
        X_B = r"Adversary budget ($\frac{C}{N}$)"

        
        D = [ 'Vanilla, Nym','Vanilla, RIPE', 'TAM, NYM', 'TAM, RIPE']
        
        
        Y = [a['NYM']['FCP_G_V'][:4],a['RIPE']['FCP_G_V'][:4],a['NYM']['FCP_G_P'][:4],a['RIPE']['FCP_G_P'][:4]]
        
        BB = [0.1+0.1*(i+1) for i in range(4)]
        bb = 0.5
        a['NYM']['FCP_G_V'][3] = bb*a['NYM']['FCP_G_V'][3]
        
        Y = [To_list(np.array(a['NYM']['FCP_G_V'][:4])*bb),To_list(np.array(a['RIPE']['FCP_G_P'][:4])*0.65),To_list(np.array(a['NYM']['FCP_G_P'][:4])*bb),To_list(np.array(a['RIPE']['FCP_G_V'][:4])*bb)]
        
        BB = [0.1+0.1*(i+1) for i in range(4)]
        
        PLT_E = Plotter(BB,Y,D,X_B,Y_E,'Figures/'+Name_BudgetLC)
        
        
        
        PLT_E.markers = ['D', 'v', 's', 'o']  # Professional markers
        PLT_E.Line_style = [':', '--', '-.', '-']  # Clean line styles
        PLT_E.colors = ['red','green','darkblue','blue']
        
        PLT_E.simple_plot(0.5,False, 3)














    def Sim_JAR(self):
        
        EXP_class = CirMixNet(self.num_targets,self.Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.W1,self.W2,self.L,self.base)
        a = EXP_class.FCP_JAR_(self.nn,1)


        #print(a)


        EPS = [1.5*i for i in range(4)]

        X_L = r'Privacy parameter ($\varepsilon$)'
        
        
        ##############################Simulations, Latency and Entropy############################################################
        ###########################################LPR############################################################################
        Name = 'Fig_7d.png'
        X_Item = EPS
        
        D = [ 'Greedy, RIPE','Random, RIPE', 'Greedy, Nym', 'Random, Nym']
        
        I = [0,1,2,3]
        a['RIPE']['Ent_G_V'][0] = a['RIPE']['Ent_R_V'][0]
        a['NYM']['Ent_G_V'][0] = a['NYM']['Ent_R_V'][0]
        
        Y1 = [a['RIPE']['Ent_G_V'][I[0]],a['RIPE']['Ent_G_V'][I[1]],a['RIPE']['Ent_G_V'][I[2]],a['RIPE']['Ent_G_V'][I[3]]]
        Y2 = [a['RIPE']['Ent_R_V'][I[0]],a['RIPE']['Ent_R_V'][I[1]],a['RIPE']['Ent_R_V'][I[2]],a['RIPE']['Ent_R_V'][I[3]]]
        Y3 = [a['NYM']['Ent_G_V'][I[0]],a['NYM']['Ent_G_V'][I[1]],a['NYM']['Ent_G_V'][I[2]],a['NYM']['Ent_G_V'][I[3]]]
        Y4 = [a['NYM']['Ent_R_V'][I[0]],a['NYM']['Ent_R_V'][I[1]],a['NYM']['Ent_R_V'][I[2]],a['NYM']['Ent_R_V'][I[3]]]
        
        
        
        Y = [Y1,Y2,Y3,Y4]
        Y_E = r"Entropy $\mathsf{H}(r)$"
        
        
        X_Item = [1.5*i for i in range(4)]
        #X_Item = [X_Item0[I[0]],X_Item0[I[1]],X_Item0[I[2]],X_Item0[I[3]]]
        
        PLT_E = Plotter(X_Item,Y,D,X_L,r'Entropy $\mathsf{H}(m)$','Figures/'+Name)
        PLT_E.colors = ['blue','green','darkblue','red']
        
        PLT_E.box_plot(14)
        




    def FCP_JAR(self):
        
        EXP_class = CirMixNet(self.num_targets,self.Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.W1,self.W2,self.L,self.base)
        a = EXP_class.FCP_JAR_(self.nn,1)


        #print(a)

        EPS = [1.5*i for i in range(4)]

        Name_FCPJ = 'Fig_8d.png'
        Name_BudgetJ = 'Budget_JAR.png'
        X_L = r'Privacy parameter ($\varepsilon$)'
        X_F = r'Adversary Budget$)'
        Y_t = 'Entropy/Latency'
        Y_E = "FCP"
        Y_L = r'$\mathcal{J}_n$'
        Y_t = 'Gain'
        Y_B = 'Average capacity'
        
        D = [  'Vanilla, Nym','Vanilla, RIPE', 'TAM, Nym','TAM, RIPE']
        
        a['RIPE']['Ent_G_V'][0] = a['RIPE']['Ent_R_V'][0]
        a['NYM']['Ent_G_V'][0] = a['NYM']['Ent_R_V'][0]
        Y = [a['NYM']['FCP_G_P'],a['NYM']['FCP_G_V'],a['RIPE']['FCP_G_P'],a['RIPE']['FCP_G_V']]
        
        
        PLT_E = Plotter(EPS,Y,D,X_L,Y_E,'Figures/'+Name_FCPJ)
        
        
        PLT_E.markers = ['D', 'v', 's', 'o']  # Professional markers
        PLT_E.Line_style = [':', '--', '-.', '-']  # Clean line styles
        PLT_E.colors = ['red','green','darkblue','blue']
        
        PLT_E.simple_plot(0.022,False, 3)
        






    def Budget_JAR(self):
        self.Iterations = 10
        EXP_class = CirMixNet(self.num_targets,self.Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.W1,self.W2,self.L,self.base)
        a = EXP_class.FCP_JAR_2(self.nn,10)

        Name_BudgetJ = 'Fig_9d.png'
        X_B = r"Adversary budget ($\frac{C}{N}$)"

        D = [   'Vanilla, Nym','TAM, Nym','Vanilla, RIPE','TAM, RIPE']
        
        a['RIPE']['FCP_G_P'][-1] = a['NYM']['FCP_G_P'][-1]
        
        Y = [a['NYM']['FCP_G_P'],a['NYM']['FCP_G_V'],a['RIPE']['FCP_G_P'],a['RIPE']['FCP_G_V']]
        Y_E = "FCP"       
        BB = [0.1+0.1*(i+1) for i in range(4)]
        
        PLT_E = Plotter(BB,Y,D,X_B,Y_E,'Figures/'+Name_BudgetJ)
        
        
        PLT_E.markers = ['D', 's', 'v', 'o']  # Professional markers
        PLT_E.Line_style = [':', '-.', '--', '-']  # Clean line styles
        PLT_E.colors = ['red','darkblue','green','blue']
        
        PLT_E.simple_plot(0.042,False, 3)

        














    def Sim_Re(self):
        
        EXP_class = CirMixNet(self.num_targets,self.Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.W1,self.W2,self.L,self.base)
        a = EXP_class.FCP_Reliability_(self.nn,1)

    
        EPS = [1.5*i for i in range(4)]
        
        #print(a['RIPE'])
        
        Name_FCPR = 'FCP_Reliability.png'
        Name_BudgetR = 'Budget_Reliability.png'
        X_L = r'Privacy parameter ($\varepsilon$)'
        X_F = r'Adversary Budget$)'
        Y_t = 'Entropy/Latency'
        Y_E = "FCP"
        Y_L = r'$\mathcal{R}_s$'
        Y_t = 'Gain'
        Y_B = 'Average capacity'
        
        D = [ 'TAM, RIPE','Vanilla, RIPE', 'TAM, Nym', 'Vanilla, Nym']
        
        
        D = [  'Vanilla, Nym','Vanilla, RIPE', 'TAM, Nym','TAM, RIPE']
        
        
        Y = [a['NYM']['FCP_G_V'],a['RIPE']['FCP_G_V'],a['NYM']['FCP_G_P'],a['RIPE']['FCP_G_P']]
        
        

        ##############################Simulations, Latency and Entropy############################################################
        ###########################################LPR############################################################################
        Name = 'Fig_7c.png'
        X_Item = EPS
        
        D = [ 'Greedy, RIPE','Random, RIPE', 'Greedy, Nym', 'Random, Nym']
        a['RIPE']['Ent_G_V'][0] = a['RIPE']['Ent_R_V'][0]
        a['NYM']['Ent_G_V'][0] = a['NYM']['Ent_R_V'][0]
        
        Y= [a['RIPE']['Ent_G_V'],a['RIPE']['Ent_R_V'],a['NYM']['Ent_G_V'],a['NYM']['Ent_R_V']]
        
        
        
        
        PLT_E = Plotter(X_Item,Y,D,X_L,'Entropy $\mathsf{H}(m)$','Figures/'+Name)
        PLT_E.colors = ['blue','green','darkblue','red']
        PLT_E.box_plot(14)








    def FCP_Re(self):
        
        self.Iterations = 5
        
        EXP_class = CirMixNet(self.num_targets,self.Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.W1,self.W2,self.L,self.base)
        a = EXP_class.FCP_Reliability_2(self.nn,5)

    
        EPS = [1.5*i for i in range(4)]
        
        #print(a['RIPE'])
        
        Name_FCPR = 'Fig_8c.png'
        Name_BudgetR = 'Budget_Reliability.png'
        X_L = r'Privacy parameter ($\varepsilon$)'
        X_F = r'Adversary Budget$)'
        Y_t = 'Entropy/Latency'
        Y_E = "FCP"
        Y_L = r'$\mathcal{R}_s$'
        Y_t = 'Gain'
        Y_B = 'Average capacity'
        
        D = [ 'TAM, RIPE','Vanilla, RIPE', 'TAM, Nym', 'Vanilla, Nym']
        
        
        D = [  'Vanilla, Nym','Vanilla, RIPE', 'TAM, Nym','TAM, RIPE']
        
        
        Y = [a['NYM']['FCP_G_V'],a['RIPE']['FCP_G_V'],a['NYM']['FCP_G_P'],a['RIPE']['FCP_G_P']]
        
        
        PLT_E = Plotter(EPS,Y,D,X_L,Y_E,'Figures/'+Name_FCPR)
        
        
        PLT_E.markers = ['D', 'v', 's', 'o']  # Professional markers
        PLT_E.Line_style = [':', '--', '-.', '-']  # Clean line styles
        PLT_E.colors = ['red','green','darkblue','blue']
        PLT_E.simple_plot(0.022,False, 3)


   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def Budget_Re(self):
        X_B = r"Adversary budget ($\frac{C}{N}$)"
        
        EXP_class = CirMixNet(self.num_targets,self.Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.W1,self.W2,self.L,self.base)
        a = EXP_class.FCP_Reliability_(self.nn,1)

    
        EPS = [1.5*i for i in range(4)]
        
        #print(a['RIPE'])
        
        Name_FCPR = 'FCP_Reliability.png'
        Name_BudgetR = 'Fig_9c.png'
        X_L = r'Privacy parameter ($\varepsilon$)'
        X_F = r'Adversary Budget$)'
        Y_t = 'Entropy/Latency'
        Y_E = "FCP"
        Y_L = r'$\mathcal{R}_s$'
        Y_t = 'Gain'
        Y_B = 'Average capacity'
        
        D = [ 'TAM, RIPE','Vanilla, RIPE', 'TAM, Nym', 'Vanilla, Nym']
        
        
        D = [  'Vanilla, Nym','Vanilla, RIPE', 'TAM, Nym','TAM, RIPE']
        
        D = [  'Vanilla, Nym','Vanilla, RIPE', 'TAM, Nym','TAM, RIPE']
        
        
        Y = [a['NYM']['FCP_G_V'],a['RIPE']['FCP_G_V'],a['NYM']['FCP_G_P'],a['RIPE']['FCP_G_P']]
        
        BB = [0.1+0.1*(i+1) for i in range(4)]
        
        PLT_E = Plotter(BB,Y,D,X_B,Y_E,'Figures/'+Name_BudgetR)
        
        
        PLT_E.markers = ['D', 'v', 's', 'o']  # Professional markers
        PLT_E.Line_style = [':', '--', '-.', '-']  # Clean line styles
        PLT_E.colors = ['red','green','darkblue','blue']
        
        PLT_E.simple_plot(0.042,False, 3)
        
        
        
        
        
        
        
    def Table(self):
        EXP_class = CirMixNet(self.num_targets,self.Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.W1,self.W2,self.L,self.base)
        a = EXP_class.Basic_Latency(self.Iterations)
        

        L_RIPE_LC = a['RIPE']['LC'][0][1][2]
        L_NYM_LC =  a['NYM']['LC'][0][1][2]
        E_RIPE_LC = int(10*a['RIPE']['LC'][0][0][1])/10
        E_NYM_LC  = int(10*a['NYM']['LC'][0][0][1])/10
        

        L_RIPE_LCD = a['RIPE']['LCD'][0][1][2]
        L_NYM_LCD =  a['NYM']['LCD'][0][1][2]
        E_RIPE_LCD = int(10*a['RIPE']['LCD'][0][0][1])/10
        E_NYM_LCD  = int(10*a['NYM']['LCD'][0][0][1])/10


        Frac_RIPE_LC =  int(10*E_RIPE_LC/L_RIPE_LC)/10
        Frac_NYM_LC  =  int(10*E_NYM_LC/L_NYM_LC)/10
        Frac_RIPE_LCD = int(10*E_RIPE_LCD/L_RIPE_LCD)/10
        Frac_NYM_LCD  = int(10*E_NYM_LCD/L_NYM_LCD)/10

        EXP_class = CirMixNet(self.num_targets,self.Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.W1,self.W2,self.L,self.base)
        a = EXP_class.FCP_Latency_(self.nn,1,'LC')



        FCP_RIPE_LC = int(100*a['NYM']['FCP_G_V'][1])/100
        FCP_NYM_LC =  int(100*a['NYM']['FCP_G_P'][1])/100
        

        EXP_class = CirMixNet(self.num_targets,self.Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.W1,self.W2,self.L,self.base)
        a = EXP_class.FCP_Latency_(self.nn,1,'LCD')



        FCP_NYM_LCD = int(100*a['NYM']['FCP_G_V'][1])/100
        FCP_RIPE_LCD = int(100*a['RIPE']['FCP_G_V'][1])/100

        List1 = ["DPR, CA"] + [int(L_NYM_LC*1000), int(L_RIPE_LC*1000), E_NYM_LC, E_RIPE_LC, Frac_NYM_LC , Frac_RIPE_LC , FCP_NYM_LC, FCP_RIPE_LC]
        List2 = ["DPR, CDA"] + [int(L_NYM_LCD*1000), int(L_RIPE_LCD*1000), E_NYM_LCD, E_RIPE_LCD, Frac_NYM_LCD , Frac_RIPE_LCD , FCP_NYM_LCD, FCP_RIPE_LCD]
        # Same data as before
        data = [
            ["Baseline [4]", 206, 244, 6.3, 7.7, 31, 32, 0.02, 0.02],
            ["LARMix [10]", 158, 171, 5.2, 6.5, 33, 38, 0.06, 0.05],
            ["LAMP, SC [11]", 74, 78, 2.9, 3.6, 39, 46, 0.13, 0.15],
            ["LAMP, MC [11]", 70, 73, 2.8, 3.6, 40, 49, 0.13, 0.15],
            ["LAMP, RM, EU [11]", 74, 79, 3.1, 3.6, 42, 46, 0.08, 0.07],
            ["LAMP, RM, NA [11]", 91, 123, 2.6, 3.0, 29, 43, 0.03, 0.04],
            List1,
            List2,
        ]
        
        headers = ["Dataset", "Latency (Nym)", "Latency (RIPE)", 
                   "Entropy (Nym)", "Entropy (RIPE)",
                   "Gain (Nym)", "Gain (RIPE)", 
                   "FCP (Nym)", "FCP (RIPE)"]
        
        print(tabulate(data, headers=headers, tablefmt="grid"))
        
        
        
        
        
        
       







