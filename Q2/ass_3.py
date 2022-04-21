#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[1]:


path = "C:/Users/gupta/Downloads/Wiki-Vote.txt"
file1 = open(path, 'r')
lines = file1.readlines()


# In[2]:


lines = lines[4:]


# In[ ]:





# In[3]:


edges = []
for line in lines:
    edges.append([int(line.split()[0]), int(line.split()[1])])


# In[4]:


node_set = set()
for edge in edges:
    node_set.add(edge[0])
    node_set.add(edge[1])


# In[5]:


len(node_set)


# In[6]:


node_set = list(node_set)


# In[7]:


max = node_set[0]
for node in node_set:
    if node > max:
        max = node
print(max)


# In[8]:


len(edges)


# In[74]:


#Number of Nodes as described in datasets are 7115
n = max
adj_mat = [[0]*(n+1)]*(n + 1)


# In[71]:


edges[1]


# In[75]:


#Adjacency Matrix
print(adj_mat[3][3])

for i in range(len(edges)):
    n1 = edges[i][0]
    n2 = edges[i][1]
    adj_mat[n1][n2] = 1

    
# print(adj_mat[3][3])


# In[12]:


class adjNode:
    def __init__(self, data):
        self.vertex = data
        self.next = None


# In[13]:


#Adjacency list python
graph = [0]*(max + 1)
for edge in edges:
    n1 = edge[0]
    n2 = edge[1]

    new_node = adjNode(n2)
    new_node.next = graph[n1]
    graph[n1] = new_node


# In[14]:


temp = graph[3]
while temp:
    print(temp.vertex)
    temp = temp.next


# In[15]:


#Number of Nodes
len(node_set)


# In[16]:


#Number of Edges
len(edges)


# In[17]:


len(graph)


# In[18]:


a = [1,2,4,5,6,7,8,9,3]


# In[19]:


for i in range(len(a)):
    print(a[i])


# In[20]:


#Average In-out degree
out_ = [0]*(max + 1)
in_ = [0]*(max + 1)
x = []
for ele in range(len(graph)):
    out_node = 0
    temp = graph[ele]
    while temp:
        out_node += 1
        in_[temp.vertex] += 1
        temp = temp.next
    out_[ele] = out_node


# In[ ]:





# In[21]:


len(in_)


# In[22]:


sum(out_)/len(node_set)


# In[23]:


sum(in_)/len(node_set)


# In[24]:


in_idx = 0
max_in = in_[0]
for i in range(len(in_)):
    if(in_[i] > max_in):
        in_idx = i
        max_in = in_[i]


# In[25]:


out_idx = 0
max_out = out_[0]
for i in range(len(out_)):
    if(out_[i] > max_out):
        out_idx = i
        max_out = out_[i]


# In[26]:


in_idx


# In[27]:


out_idx


# In[28]:


in_[in_idx]


# In[29]:


out_[out_idx]


# In[ ]:





# In[30]:


#Graph density
nodes = len(node_set)
max_nodes = nodes*(nodes-1)
print(max_nodes)


# In[31]:


density = len(edges)/max_nodes


# In[32]:


density


# In[33]:


max_in


# In[34]:


y = [i for i in range(len(in_))]


# In[35]:


len(y)


# In[36]:


class Node:
    def __init__(self,idx,data):
        self.index = idx
        self.data = data


# In[37]:


for i,v in enumerate(in_):
    print(i,v)


# In[38]:


out_[24]


# In[39]:


in_arr = [Node(i,v) for i,v in enumerate(in_) if v!= 0 or out_[i] != 0]


# In[40]:


out_arr = [Node(i,v) for i,v in enumerate(out_) if v!= 0 or in_[i] != 0]


# In[41]:


len(out_arr)


# In[42]:


# for ele in in_arr:
#     print(ele.index,ele.data)


# In[43]:


from matplotlib import pyplot as plt
x = [ele.data for ele in out_arr]
y = [ele.data for ele in in_arr]

plt.hist(x)


# In[44]:


plt.hist(y)


# In[45]:


len(graph)


# In[46]:


temp_1 = graph[3]
while temp_1:
    print(temp_1.vertex)
    temp_1 = temp_1.next


# In[47]:


# for i in range(0,max+1):
#     if adj_mat[i][i] == 1:
#         print(i)


# In[48]:


adj_mat[6][28]


# In[ ]:





# In[76]:


#Local clustering coefficient
#check the neighbour of each node and count the edges between the neighbour nodes
#divide it by the nn(nn-1) where nn is the count neighbour nodes of each node
local_cluster = []
for node in in_arr:
    temp = graph[node.index]
    nn = []
    while temp:
        nn.append(temp.vertex)
        temp = temp.next
    nn_edge = 0
    for i in range(len(nn)):
        for j in range(len(nn)):
            temp_list = []
            temp_ele = graph[nn[i]]
            while temp_ele:
                temp_list.append(temp_ele.vertex)
                temp_ele = temp_ele.next

            for k in temp_list:
                if k == nn[j]:
                    nn_edge += 1
#     print(nn_edge)
    if(out_[node.index] > 1):
        local_cluster.append(nn_edge/(out_[node.index]*(out_[node.index]-1)))
    
    
        
        
    


# In[77]:


x = local_cluster
y = [i for i in range(len(local_cluster))]

plt.hist(x)


# In[ ]:




