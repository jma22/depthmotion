import pickle



def knn_graph_edges(k):
    dist_path = '../data/dist_dict.pickle'
#     dist_path = '../data/monkey_dist_dict.pickle'
    with open(dist_path,"rb") as f:
      data = pickle.load(f)
    edge_list = []
#     for i in range(968):
    for i in range(588):
        for node,dist in data[i][0:k]:
            edge_list.append((i,node))
    return edge_list
            






    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#### create dist dictionary:
# def compute_dist(triple1,triple2):
#     return ((triple1[0]-triple2[0])**2+(triple1[1]-triple2[1])**2+(triple1[2]-triple2[2])**2)**0.5
# dist_path = '../data/monkey_data.pickle'
# with open(dist_path,"rb") as f:
#       data = pickle.load(f)
# allpoints = data[0]['coor']
# dist_dict = {i:[] for i in range(allpoints.shape[0])}
# for i in range(allpoints.shape[0]):
#     print(i)
#     for j in range(allpoints.shape[0]):
#         if j==i:
#             continue
#         else:
#             dist_dict[i].append((j,compute_dist(allpoints[i,:],allpoints[j,:])))
            
            
# for i in range(allpoints.shape[0]):
#     dist_dict[i].sort(key=lambda x:x[1])
# print(dist_dict[0])

# dist_path = '../data/monkey_dist_dict.pickle'
# with open(dist_path,"wb") as f:
#   data = pickle.dump(dist_dict,f)
                  
