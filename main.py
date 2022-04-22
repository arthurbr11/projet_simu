import itertools

import numpy as np
import random as rd
import networkx as nx
import matplotlib.pyplot as plt
from tabulate import tabulate


###################################################################################
# def of the class
####################################################################################
class Graph:
    def __init__(self, personnes, trotinette, nodes_name, edeges_length,personnes_moove_w_tr):
        self.personnes = personnes
        self.trotinette = trotinette
        self.nodes_name = nodes_name
        self.Adj = np.ones((len(self.nodes_name), len(self.nodes_name))) - np.identity(len(self.nodes_name))
        self.edges_length = edeges_length
        self.personnes_moove_w_tr = personnes_moove_w_tr
    def trac_graph(self):
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

        ax.set_title('Random graph')
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
        print(tabulate(data, headers=["Location", "numbers of people", "numbers of trotinette", "nb_p want to moove","people who can't take trotinettes"]))

class Agent:
    def __init__(self,n_pos,type_habitant):
        self.n_pos = n_pos #n_pos corespond au nombre qui caractérise l'endroit ou on est (entier entre 0 et 3)
        self.type_habitant = type_habitant #type_habitant entier 0 ou 1,0 Paris,1 Champs-sur-Marne
    
    def proba_bouger(self,t):
        if self.type_habitant==0:
            
            1
        else:
            1
        



####################################################################################
# tests
####################################################################################

nodes_name = ["Ponts", "Résidence", "Gare", "Super U"]
nodes_priority = [0, 0.7, 0.3, 0]
personnes = [0, 10, 10, 0]
trotinettes = [2, 8, 11, 1]
edeges_length = [[0, 0, 0, 0],
                 [0.6, 0, 0, 0],
                 [0.35, 1.1, 0, 0],
                 [0.9, 1.5, 0.6, 0.3]]

def init_personnes(nb, nodes_priority):
    personnes = []
    for k in nodes_priority:
        personnes += [int(k * (nb + 1))]
    return (personnes)


def init_trotinette(nb, nodes_priority):
    trotinettes = []
    for k in nodes_priority:
        trotinettes += [int(k * (nb + 1))]
    return (trotinettes)


def simulation_of_day(nodes_name, edeges_length, dt):
    personnes = init_personnes(100, nodes_priority)
    trotinettes = init_trotinette(40, nodes_priority)
    personnes_moove_w_tr=[personnes[k]*0.5 for k in range(len(personnes))]
    G = Graph(personnes, trotinettes, nodes_name, edeges_length,personnes_moove_w_tr)
    G.trac_graph()
    return ()


simulation_of_day(nodes_name, edeges_length, 3)
