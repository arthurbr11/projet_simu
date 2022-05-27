#import itertools
import numpy as np
import random as rd
import networkx as nx
import matplotlib.pyplot as plt
from tabulate import tabulate

####################################################################################
# We fix the constant of the problem
####################################################################################

vm=83 #speed by foot in m/min
vt=332 #speed by trotinette

#Constants multiplicatives
alpha=1.5
beta=3
gamma=1


# names of the place of our modelisation
nodes_name = ["Ponts", "Résidence", "Gare", "Super U"]

#Matrix length in m
d_ER = 1100 
d_EG = 1250
d_EC = 950
d_RG = 1800
d_RC = 650
d_GC = 2000
edeges_length = [[0, d_ER,d_EG,d_EC],
                 [d_ER, 0, d_RG, d_RC],
                 [d_EG, d_RG, 0, d_GC],
                 [d_EC, d_RC, d_GC, 0]]

####################################################################################
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
# def of the class for the GRAPH
####################################################################################
class Graph:
    
    def __init__(self, personnes, trotinette, nodes_name, edeges_length,personnes_w_moove_w_tr):# function to initialize our GRAPH
        self.personnes = personnes #list of numbers of people at each place (lenght is the same as numbers of places)
        self.trotinette = trotinette #list of numbers of trotinette at each place (lenght is the same as numbers of places)
        self.nodes_name = nodes_name #list of places names (lenght is the same as numbers of places)
        self.Adj = np.ones((len(self.nodes_name), len(self.nodes_name))) - np.identity(len(self.nodes_name)) #adjacence matrix of the graph
        self.edges_length = edeges_length #matrix of lenght beetween each place 
        self.personnes_w_moove_w_tr = personnes_w_moove_w_tr  #list of numbers of people who wants to moove with trotinette at each place (lenght is the same as numbers of places)
    
    def trac_graph(self,t,dt):#function to trac our GRAPH and modulate our data_base
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

###################################################################################
# def of the class for the AGENT
####################################################################################

class Agent:
    
    def __init__(self,n_pos,type_habitant,n_pos_apres,w_or_wo):# function to initialize our AGENT
        
        self.n_pos = n_pos #n_pos corespond to the number who caracterize the place where we are (int between 0 or 3)
        self.type_habitant = type_habitant #type_habitant int 0 or 1, 0 Champs-sur-Marne ,1 Paris
        self.n_pos_apres = n_pos_apres # the position the instant after 
        self.w_or_wo = w_or_wo # 0 if we moove without trotinette -1 if we don't moove 1 if we moove with trotinette
        
    def proba_moove(self,t,dt): # function which return the place where the agent will go the next  t+1 (it can be the place where he is at t)
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
# def utility function
####################################################################################

def utility(data,dt,T):
    sum_trotinettes_not_use=0
    for k in range(len(data)):
        for j in range (len(data[k])):
            sum_trotinettes_not_use+=data[k][j][-1]
    L=sum_trotinettes_not_use*dt #Total time where the trotinettes are not use
    return(-(beta*T+gamma*L))

####################################################################################
# def function of initialization 
####################################################################################

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


def init_trotinette(nb, p,typ):# typ==0 (equal repart) typ==1 (1/2 residence 1/2 gare) typ==2 (p residence 1-p gare)
    trotinettes = [0,0,0,0]
    
    if (typ==0):
        n=int(1/4*nb)
        trotinettes =[n,n,n,n]
    elif(typ==1):
        n=int(1/2*nb)
        trotinettes =[0,n,n,0]
    else:
        n=int(p*nb)
        trotinettes =[0,n,nb-n,0]
    return (trotinettes)

####################################################################################
# def function of the main loop
####################################################################################

def simulation_of_day(nodes_name,dt,p,f,typ,nb_trot):
    
    rd.seed(30) # We fix the seed of the module random in order to have all the time the same sequence of pick 

    data=[] #initialization of our database
    T=0 #Time total of travel

    trotinettes = init_trotinette(nb_trot,p,typ) #initialization of our trotinette
    
    agent,personnes = init_personnes(100,p) #initialization of our agent
    personnes_w_moove_w_tr=[0,0,0,0]

    
    nb_interval_hours=int(60/dt)
    nb_intervals=int((21-7)*nb_interval_hours)
    
    for t in range(0,nb_intervals+1):
        
        trotinettes_moove=[0,0,0,0]
        personnes_w_moove_w_tr=[0,0,0,0]
        
        for k in range (len(agent)):
            #we update the number of people in the place
            personnes[agent[k].n_pos]-=1
            personnes[agent[k].n_pos_apres]+=1
            
            if (agent[k].w_or_wo==1 and trotinettes[agent[k].n_pos]!=0 ):#we update the number of people in the trotinette if the personn moove and take the trotinette
                trotinettes[agent[k].n_pos]-=1
                trotinettes[agent[k].n_pos_apres]+=1
            
            agent[k].n_pos=agent[k].n_pos_apres #updtae the position of our agent
            
            i=agent[k].proba_moove(t, dt) #compute the next place where he will go
            
            if (i!=agent[k].n_pos):# if he don't stay in the same place
                personnes_w_moove_w_tr[agent[k].n_pos]+=1
                U=rd.uniform(0,1)
                if (trotinettes[agent[k].n_pos]-trotinettes_moove[agent[k].n_pos]!=0):#update when he moove and he have trotinette to do it 
                    trotinettes_moove[agent[k].n_pos]+=1
                    agent[k].w_or_wo=1
                    agent[k].n_pos_apres=i
                    T+=edeges_length[i][agent[k].n_pos]/vt
                elif (U<f):# update when he moove despite he don't have trotinette and choose to not wait
                    agent[k].w_or_wo=0
                    agent[k].n_pos_apres=i
                    T+=alpha*edeges_length[i][agent[k].n_pos]/vm
                else:# update when he stay in the same place bc he don't have trotinette and choose to wait
                    agent[k].n_pos_apres=agent[k].n_pos
                    agent[k].w_or_wo=-1
                    T+=dt
                    
                    
            else:# update when he stay in the same place
                agent[k].n_pos_apres=agent[k].n_pos
                agent[k].w_or_wo=-1
                
                    
        print(affichetemps(t, dt))
        G = Graph(personnes, trotinettes, nodes_name,edeges_length,personnes_w_moove_w_tr)
        data+=[G.trac_graph(t,dt)]
    
    return (data,T)


####################################################################################
# THE MAIN LOOP UTILISATION
####################################################################################
dt=30
p=0.5
f=1 #if f=1 never wait if f=0 wait all the time
typ=2
nb_trot=40

data,T=simulation_of_day(nodes_name,dt,p,f,typ,nb_trot)
U=utility(data,dt,T)
print(U)


































