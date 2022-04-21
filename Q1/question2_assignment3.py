#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import nltk
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
import networkx as nx


# In[2]:


def importFile(path):
    content = []
    with open(path) as f:
        content = f.readlines()
    
    info = content[:3]
    content = content[3::]
    for i, line in enumerate(content):
        content[i] = nltk.RegexpTokenizer(r"\w+").tokenize(line)
    info[-1] = nltk.RegexpTokenizer(r"\w+").tokenize(info[-1])
    
    df = pd.DataFrame(content[1::], columns = content[0])
    df["FromNodeId"] = df["FromNodeId"].map(int)
    df["ToNodeId"] = df["ToNodeId"].map(int)
    
    return info, df


# In[ ]:


def createGraph(content):
    G = nx.from_pandas_edgelist(content,source = "FromNodeId", target = "ToNodeId")
    return G.to_directed()


# In[ ]:


def PageRank(G, d = 0.85, max_iters = 100):
    n = G.number_of_nodes()
    rank = {}
    node_val = {}
    error = []
    for node in G.nodes():
        rank[node] = 1/n
        node_val[node] = 0
        
    for _ in range(max_iters):            
        for node in G.nodes():
            out = G.out_edges(node)
            n_edges = len(out)
            for edge in out:
                node_val[edge[1]] += rank[edge[0]]/n_edges

        for node in G.nodes():
            node_val[node] = (1-d)/n + d*node_val[node]
        
        e = mse(list(rank.values()),list(node_val.values()))
        error.append(e)
        
        for node in G.nodes():
            rank[node] = node_val[node]
            node_val[node] = 0
        
        if e < 1e-15:
            break
        
    return error, rank


# In[ ]:


def updation(G, a, b, norm, type_):
    for node in G.nodes():
        if type_ == "authority":
            edges = G.in_edges(node)
        else:
            edges = G.out_edges(node)
        n_edges = len(edges)
        a[node] = 0
        for edge in edges:
            if type_ == "authority":
                a[node] += b[edge[0]]
            else:
                a[node] += b[edge[1]]
                
        norm += a[node]**2
    norm = np.sqrt(norm)
    for node in G.nodes():
        a[node] /= norm
        
def HITS(G, max_iters = 100):
    n = G.number_of_nodes()
    hub = {}
    authority = {}
    e1 = []
    e2 = []
    
    for node in G.nodes():
        hub[node] = 1
        authority[node] = 1
    
    cur_auth = authority.copy()
    cur_hub = hub.copy()
    
    for _ in range(max_iters):
        updation(G,authority,hub,0,"authority")
        updation(G,hub,authority,0,"hub")
        
        e1.append(mse(list(authority.values()),list(cur_auth.values())))
        e2.append(mse(list(hub.values()),list(cur_hub.values())))

        cur_auth = authority.copy()
        cur_hub = hub.copy()
        
        if e1[-1] < 1e-20 and e2[-1] < 1e-20:
            break
            
    return authority, hub, e1[1::], e2[1::]


# In[ ]:


def sortDict(d, title):
    d = {key : value for key, value in sorted(d.items(), key=lambda item: item[1], reverse = True)}
    i = 0
    print("top 10 " + title + " scores :")
    for key, val in d.items():
        if i<10:
            print(key,"\t",val)
        i+=1
    print()
    return d


# In[ ]:


path = "dataset/Wiki-Vote.txt"
meta, content = importFile(path)
G = createGraph(content)


# In[ ]:


error, rank = PageRank(G)


# In[ ]:


plt.plot(error)


# In[ ]:


authority, hub, e1, e2 = HITS(G)


# In[ ]:


plt.plot(e1,'r',e2, 'g')


# In[ ]:


rank = sortDict(rank, "rank")
authority = sortDict(authority, "authority")
hub = sortDict(hub, "hub")


# In[ ]:


meta


# In[ ]:




