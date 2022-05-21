#import itertools

import numpy as np
import random as rd
import networkx as nx
import matplotlib.pyplot as plt
from tabulate import tabulate


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
    def __init__(self, personnes, trotinette, nodes_name, edeges_length,personnes_w_moove_w_tr):
        self.personnes = personnes #list of numbers of people at each place (lenght is the same as numbers of places)
        self.trotinette = trotinette #list of numbers of trotinette at each place (lenght is the same as numbers of places)
        self.nodes_name = nodes_name #list of places names (lenght is the same as numbers of places)
        self.Adj = np.ones((len(self.nodes_name), len(self.nodes_name))) - np.identity(len(self.nodes_name)) #adjacence matrix of the graph
        self.edges_length = edeges_length #matrix of lenght beetween each place 
        self.personnes_w_moove_w_tr = personnes_w_moove_w_tr  #list of numbers of people who wants to moove with trotinette at each place (lenght is the same as numbers of places)
    def trac_graph(self,t,dt):
        nb_endroit = len(self.nodes_name)
        G = nx.Graph()
        for k in range(len(self.nodes_name)):
            G.add_node(self.nodes_name[k],weight=self.personnes_w_moove_w_tr[k] - self.trotinette[k])
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
            angry_people=0
            trotinette_not_use=0
            if (self.personnes_w_moove_w_tr[k] - self.trotinette[k] >= 0):
                angry_people=self.personnes_w_moove_w_tr[k] - self.trotinette[k]
            else: 
                trotinette_not_use=self.trotinette[k]-self.personnes_w_moove_w_tr[k] 
            data+=[[self.nodes_name[k],self.personnes[k],self.trotinette[k],self.personnes_w_moove_w_tr[k],angry_people,trotinette_not_use]]
        print(tabulate(data, headers=["Location", "numbers of people", "numbers of trotinette", "nb_p want to moove","people who can't take trotinettes","trotinette_not_use"]))
        return(data)

class Agent:
    def __init__(self,n_pos,type_habitant,n_pos_apres,w_or_wo):
        
        self.n_pos = n_pos #n_pos corespond to the number who caracterize the place where we are (int between 0 or 3)
        self.type_habitant = type_habitant #type_habitant int 0 or 1, 0 Champs-sur-Marne ,1 Paris
        self.n_pos_apres = n_pos_apres # the position the instant after 
        self.w_or_wo = w_or_wo # 0 if we moove without trotinette -1 if we don't moove 1 if we moove with trotinette
        
    def proba_moove(self,t,dt):
        s=0
        U=rd.uniform(0,1)
        a,b,c,d,e=unit7_9(t,dt),unit9_12(t,dt),unit12_14(t,dt),unit14_17(t,dt),unit17_21(t,dt)
        if self.type_habitant==0: #matrice for Champs-sur-Marne student
            Proba=[[a+b+(1/2)*c+d+(1/4)*e,(2/6)*c+(1/4)*e,(1/4)*e,(1/6)*c+(1/4)*e],
                   [(3/4)*a+b+(2/3)*d,(1/4)*a+(2/3)*c+(2/3)*e,(1/4)*e,(1/3)*c+(1/3)*d+(1/12)*e],
                   [a+b+c+(1/3)*d,(1/6)*d+(1/3)*e,(1/3)*d+(2/3)*e,(1/6)*d],
                   [(1/3)*(a+b)+(1/4)*d,(1/3)*(a+b)+(2/3)*c+(1/4)*d+(3/4)*e,(1/4)*(c+d)+(1/4)*e,(1/3)*(a+b)+(1/3)*c+(1/4)*d]]
            for k in range(4):
                s+=Proba[self.n_pos][k]
                if (U<s):
                    return(k)
        else:#matrice for Paris student
            Proba=[[a+b+(1/2)*c+(2/3)*d+(1/4)*e,(1/4)*e,(1/4)*c+(1/3)*d+(1/2)*e,(1/4)*c],
                   [(1/2)*(a+b)+(2/5)*d,(1/2)*(a+b+c),(1/4)*c+(2/5)*d+e,(1/4)*c+(1/5)*d],
                   [a+(2/3)*b+(1/4)*c+(1/2)*d+(1/8)*e,(1/8)*e,(1/3)*b+(1/2)*(c+d)+(3/4)*e,(1/4)*c],
                   [(1/3)*(a+b)+(1/2)*d,(1/8)*c+(1/4)*e,(1/2)*(a+b)+(3/8)*c+(1/2)*d+(3/4)*e,(1/3)*(a+b)+(1/2)*c+(1/4)*d]]
            for k in range(4):
                s+=Proba[self.n_pos][k]
                if (U<s):
                    return(k)
        

####################################################################################
# def fonction utilité
####################################################################################

def utility(data):
    sum_people_who_cant_take_trotinettes=0
    sum_trotinettes_not_use=0
    for k in range(len(data)):
        for j in range (len(data[k])):
            sum_trotinettes_not_use+=data[k][j][-1]
            sum_people_who_cant_take_trotinettes+=data[k][j][-2]
    return(sum_people_who_cant_take_trotinettes,sum_trotinettes_not_use,sum_people_who_cant_take_trotinettes+sum_trotinettes_not_use)
        
####################################################################################
# def variables
####################################################################################

nodes_name = ["Ponts", "Résidence", "Gare", "Super U"]
edeges_length = [[0, 0, 0, 0],
                 [0.6, 0, 0, 0],
                 [0.35, 1.1, 0, 0],
                 [0.9, 1.5, 0.6, 0.3]]

def init_personnes(nb,p): # nb is numbber of agent in our simulation , p is the proportion of people who are agent 0
    personnes = [0,0,0,0]
    agent=[]
    nb_agent0=int(nb*p)
    personnes[1]=nb_agent0
    personnes[2]=nb-nb_agent0
    for k in range(nb_agent0):
        a=Agent(1,0,1,-1)
        agent+=[a]
    for k in range(nb_agent0,nb):
        a=Agent(2,1,2,-1)
        agent+=[a]
    return (agent,personnes)


def init_trotinette(nb, nodes_priority):# a faire 
    trotinettes = []
    for k in nodes_priority:
        trotinettes += [int(k * (nb + 1))]
    return (trotinettes)


def simulation_of_day(nodes_name, edeges_length, dt,p):
    data=[]
    
    
    nodes_priority=[0,p,1-p,0]
    trotinettes = init_trotinette(40, nodes_priority)
    
    agent,personnes = init_personnes(100,p)
    personnes_w_moove_w_tr=[0,0,0,0]
    print(personnes)
    
    nb_interval_hours=int(60/dt)
    nb_intervals=int((21-7)*nb_interval_hours)
    
    for t in range(0,nb_intervals+1):
        trotinettes_moove=[0,0,0,0]
        personnes_w_moove_w_tr=[0,0,0,0]
        for k in range (len(agent)):
            personnes[agent[k].n_pos]-=1
            personnes[agent[k].n_pos_apres]+=1
            if (agent[k].w_or_wo==1 and trotinettes[agent[k].n_pos]!=0 ):
                trotinettes[agent[k].n_pos]-=1
                trotinettes[agent[k].n_pos_apres]+=1
            agent[k].n_pos=agent[k].n_pos_apres
            i=agent[k].proba_moove(t, dt)
            if (i!=agent[k].n_pos):
                personnes_w_moove_w_tr[agent[k].n_pos]+=1
                agent[k].n_pos_apres=i
                if (trotinettes[agent[k].n_pos]-trotinettes_moove[agent[k].n_pos]!=0):
                    trotinettes_moove[agent[k].n_pos]+=1
                    agent[k].w_or_wo=1
                else:
                    agent[k].w_or_wo=0
            else:
                agent[k].n_pos_apres=agent[k].n_pos
                agent[k].w_or_wo=-1
                
                    
        print(affichetemps(t, dt))
        G = Graph(personnes, trotinettes, nodes_name, edeges_length,personnes_w_moove_w_tr)
        data+=[G.trac_graph(t,dt)]
    
    return (data)

data=simulation_of_day(nodes_name, edeges_length, 30,0.5)
U=utility(data)
print(U)

































