import pickle



def knn_graph_edges(k,frames=1):
    dist_path = '../data/dist_dict.pickle'
    with open(dist_path,"rb") as f:
      data = pickle.load(f)
    edge_list = []
    for j in range(frames):
        for i in range(588):
            for node,dist in data[i][0:k]:
                edge_list.append((j*588+i,j*588+node))
    return edge_list
            


def temporal_edges(frames):
    edge_list = []
    #forward
    for j in range(588):
        for i in range(frames-1):
            edge_list.append((j+i*588,j+(i+1)*588))
    #backward
    for j in range(588):
        for i in range(frames-1,0,-1):
            edge_list.append((j+i*588,j+(i-1)*588))
    return edge_list

                             


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#### create dist dictionary:
# def compute_dist(triple1,triple2):
#     return ((triple1[0]-triple2[0])**2+(triple1[1]-triple2[1])**2+(triple1[2]-triple2[2])**2)**0.5

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

# dist_path = '../data/dist_dict.pickle'
# with open(dist_path,"wb") as f:
#   data = pickle.dump(dist_dict,f)
                  
