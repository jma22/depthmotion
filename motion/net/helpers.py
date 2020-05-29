import numpy as np
def make_shuffled_idx(end,start=0):
    """
    output: iterater
    """
    return np.random.permutation(list(range(start,end)))

def get_data_tensor(idx, tensor = True,device = device):
    """
    output: full_data, obscured data
    """
    pass

def knn_graph(k,frames,num_points,dist_dict_name):
    pass

def temp_graph(frames,num_points):
    pass

def get_loss(idx,net,k,frames,device,train,sg = None,tg=None):
    """
    output: loss tensor
    """
    
    # full_data, obscured_data = get_data_tensor(idx, tensor = True,device = device)   ##TODO: Make       
    # net.train() 
    # output = net(spatial_graph, temporal_graph, obscured_data)
    # loss = F.mse_loss(output,full_data) + F.l1_loss(output,full_data)
    #loss += 

def evaluate(net,test_start,test_end):
    pass
    #get_loss