#import itertools

import numpy as np
import random as rd
import networkx as nx
import matplotlib.pyplot as plt
#from tabulate import tabulate


###################################################################################
#different function  for the time
####################################################################################

#function unitx_y(t) allow us to have 1 when we are between the time [x,y] with our discretization of time dt  
def unit7_9(t,dt):
    nb_interval_hours=60/dt
    if t>=0 and t<=nb_interval_hours*2-1:
        return(1)
    return(0)
def unit9_12(t,dt):
    nb_interval_hours=60/dt
    if t>=nb_interval_hours*2 and t<=nb_interval_hours*5-1:
        return(1)
    return(0)
def unit12_14(t,dt):
    nb_interval_hours=60/dt
    if t>=nb_interval_hours*5 and t<=nb_interval_hours*7-1:
        return(1)
    return(0)
def unit14_17(t,dt):
    nb_interval_hours=60/dt
    if t>=nb_interval_hours*7 and t<=nb_interval_hours*10-1:
        return(1)
    return(0)
def unit17_21(t,dt):
    nb_interval_hours=60/dt
    if t>=nb_interval_hours*10 and t<=nb_interval_hours*14:
        return(1)
    return(0)

def affichetemps(t,dt):
    minutes=int(7*60+t*dt)
    hours=int(minutes/60)
    minutes-=int(hours*60)
    return(str(hours)+"h"+str(minutes)+"min")
    

###################################################################################
# def of the class
####################################################################################
class Graph:
    def __init__(self, personnes, trotinette, nodes_name, edeges_length,personnes_moove_w_tr):
        self.personnes = personnes #list of numbers of people at each place (lenght is the same as numbers of places)
        self.trotinette = trotinette #list of numbers of trotinette at each place (lenght is the same as numbers of places)
        self.nodes_name = nodes_name #list of places names (lenght is the same as numbers of places)
        self.Adj = np.ones((len(self.nodes_name), len(self.nodes_name))) - np.identity(len(self.nodes_name)) #adjacence matrix of the graph
        self.edges_length = edeges_length #matrix of lenght beetween each place 
        self.personnes_moove_w_tr = personnes_moove_w_tr  #list of numbers of people who wants to moove with trotinette at each place (lenght is the same as numbers of places)
    def trac_graph(self,t,dt):
        nb_endroit = len(self.nodes_name)
        G = nx.Graph()
        for k in range(len(self.nodes_name)):
            G.add_node(self.nodes_name[k],weight=self.personnes_moove_w_tr[k] - self.trotinette[k])
        A = self.Adj
        for i in range(nb_endroit):
            for j in range(i):
                if A[i, j] == 1:
                    G.add_edge(self.nodes_name[i],self.nodes_name[j],weight=self.edges_length[i][j])
        pos = nx.spring_layout(G, seed=7)

        nodes_saturate = [(u) for (u, d) in G.nodes(data=True) if d['weight'] >= 0]
        nodes_not_saturate = [(u) for (u, d) in G.nodes(data=True) if d['weight'] <= 0]

        ax = plt.gca()
        ax.set_title("Graph instant "+str(affichetemps(t, dt)))
        # nodes
        nx.draw_networkx_nodes(G, pos, nodelist=nodes_saturate, node_color="r", node_size=1500, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=nodes_not_saturate, node_color="g", node_size=1500, ax=ax)
        # edges
        nx.draw_networkx_edges(G, pos, width=2, ax=ax)
        # node labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif", font_color="b", ax=ax)
        # edge weight labels
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        data =[]
        for k in range (len(self.nodes_name)):
            diff=0
            if (self.personnes_moove_w_tr[k] - self.trotinette[k] >= 0):
                diff=self.personnes_moove_w_tr[k] - self.trotinette[k]
            data+=[[self.nodes_name[k],self.personnes[k],self.trotinette[k],self.personnes_moove_w_tr[k],diff]]
        #print(tabulate(data, headers=["Location", "numbers of people", "numbers of trotinette", "nb_p want to moove","people who can't take trotinettes"]))
        #print()
        return(data)

class Agent:
    def __init__(self,n_pos,type_habitant):
        self.n_pos = n_pos #n_pos corespond to the number who caracterize the place where we are (int between 0 or 3)
        self.type_habitant = type_habitant #type_habitant int 0 or 1,0 Paris,1 Champs-sur-Marne
    
    def proba_moove(self,t,dt):
        s=0
        U=rd.uniform(0,1)
        a,b,c,d,e=unit7_9(t,dt),unit9_12(t,dt),unit12_14(t,dt),unit14_17(t,dt),unit17_21(t,dt)
        if self.type_habitant==0:
            Proba=[[a+b+0.6*c+0.25*d,0.3*c+0.25*d+0.45*e,0.25*d+0.25*e,0.1*c+0.25*d+0.25*e],
                   [0.8*a+0.25*b+0.8*c+0.25*d+0.1*e,0.2*a+0.25*b+0.2*c+0.25*d+0.4*e,0.25*b+0.25*d+0.25*e,0.25*b+0.25*d+0.25*e],
                   [a+b+c+d+e,a+b+c+d+e,a+b+c+d+e,a+b+c+d+e],
                   [a+b+c+d+e,a+b+c+d+e,a+b+c+d+e,a+b+c+d+e]]
            for k in range(4):
                s+=Proba[self.n_pos][k]
                if (U<s):
                    return(k)
        else:
            Proba=[[a+b+c+d+e,a+b+c+d+e,a+b+c+d+e,a+b+c+d+e],
                   [a+b+c+d+e,a+b+c+d+e,a+b+c+d+e,a+b+c+d+e],
                   [a+b+c+d+e,a+b+c+d+e,a+b+c+d+e,a+b+c+d+e],
                   [a+b+c+d+e,a+b+c+d+e,a+b+c+d+e,a+b+c+d+e]]
                        for k in range(4):
            s+=Proba[self.n_pos][k]
            if (U<s):
                return(k)
        


####################################################################################
# def variables
####################################################################################

nodes_name = ["Ponts", "Résidence", "Gare", "Super U"]
edeges_length = [[0, 0, 0, 0],
                 [0.6, 0, 0, 0],
                 [0.35, 1.1, 0, 0],
                 [0.9, 1.5, 0.6, 0.3]]

def init_personnes(nb,p):# nb is numbber of agent in our simulation , p is the proportion of people who are agent 0  (vision simpliste de l'initialisation a voir comment la faireevoluer )
    personnes = [0,0,0,0]
    agent=[]
    nb_agent0=int(nb*p)
    personnes[2]=nb_agent0
    personnes[0]=nb-nb_agent0
    for k in range(nb_agent0):
        a=Agent(2,0)
        agent+=[a]
    for k in range(nb_agent0,nb):
        a=Agent(0,1)
        agent+=[a]
    return (agent,personnes)


def init_trotinette(nb, nodes_priority):# a faire 
    trotinettes = []
    for k in nodes_priority:
        trotinettes += [int(k * (nb + 1))]
    return (trotinettes)


def simulation_of_day(nodes_name, edeges_length, dt,p):
    data=[]
    
    
    nodes_priority=[1-p,0,p,0]
    trotinettes = init_trotinette(40, nodes_priority)
    
    agent,personnes = init_personnes(100,p)
    personnes_moove_w_tr=[0,0,0,0]
    
    nb_interval_hours=int(60/dt)
    nb_intervals=int((21-7)*nb_interval_hours)
    
    for t in range(0,nb_intervals+1):
        
        
        
        
        print(affichetemps(t, dt))
        G = Graph(personnes, trotinettes, nodes_name, edeges_length,personnes_moove_w_tr)
        data+=[G.trac_graph(t,dt)]
    
    return (data)

simulation_of_day(nodes_name, edeges_length, 20,0.3)

































