import trimesh
import numpy as np
import pickle
from camera import Camera
import matplotlib.pyplot as plt
import torch
from torch.utils import data
import pickle



def normalize(vector):
    norm = np.linalg.norm(vector)
    return vector/norm

def global_to_camera(coordinate,camera_coordinate):
    """
    global_to_camera(np.array([[1,1,1],[2,2,2]]),np.array([2,2,2]))
    """
    x = np.array([-camera_coordinate[0],-camera_coordinate[1],-camera_coordinate[2]])
    y = np.array([-camera_coordinate[0],-camera_coordinate[1],camera_coordinate[0]**2/camera_coordinate[2]+camera_coordinate[1]**2/camera_coordinate[2]])
    z = np.cross(x,y)
    x = normalize(x)
    y = normalize(y)
    z = normalize(z)
    coor_mat = np.array([[x[0],x[1],x[2],0],
                        [y[0],y[1],y[2],0],
                        [z[0],z[1],z[2],0],
                        [camera_coordinate[0],camera_coordinate[1],camera_coordinate[2],1]])
    coordinate = np.concatenate([coordinate,np.ones([coordinate.shape[0],1])],axis=1)
    res = np.matmul(coordinate,np.linalg.inv(coor_mat))
    return res[...,:-1]

def genId(mesh,cam_pos):
    """
    @input: tuple of x y z of camera
    @output set of visible indexes
    """
    camera = Camera()
    print(cam_pos)
    # print(np.array([0,0,0]))
    camera.lookat(cam_pos,np.array([0,0,0]),np.array([0,0,1]))
    o,d,end = camera.generate_rays() # generates vectors and a bunch of rays
    inc=trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    o = o.reshape(-1,3)
    d = d.reshape(-1,3)
    fid = inc.intersects_first(o,d) # list of indices of hit triangles
    # fid = fid.reshape([600,600])
    alist = set([x for x in fid if x!=-1])
    return alist

def scene_to_data(cam_pos,mesh):
    """
    takes in a camera position as a 3 size vector
    returns x and edge index (visible)
    """
    edge_index = []
    point_positions = mesh.triangles_center
    point_adjacencies = mesh.face_adjacency
    idset = genId(mesh,cam_pos)
    print(len(idset))
    oldToNew = {old:new for new,old in enumerate(idset)}
    global_pos = np.array([point_positions[i]for i in oldToNew.keys()])
    x = global_to_camera(global_pos,cam_pos)
    ##edge code
    for edgepair in point_adjacencies:
        if edgepair[0] in idset and edgepair[1] in idset:
            edge_index.append([oldToNew[edgepair[0]],oldToNew[edgepair[1]]])
    return x, np.asarray(edge_index).T
    
def scene_to_x(cam_pos,mesh):
    """
    takes in a camera position as a 3 size vector
    returns x with global indexing with seen set and maxid
    """
    eps = 1e-10
    edge_index = []
    point_positions = mesh.triangles_center
    maxid = len(point_positions)-1
    point_adjacencies = mesh.face_adjacency
    idset = genId(mesh,cam_pos)

    global_pos = np.array([point_positions[idx] if idx in idset else cam_pos for idx in range(len(point_positions))])
    x = global_to_camera(global_pos,cam_pos)
    x[np.abs(x) < eps] = 0
 
    return x, idset, maxid
    
def genVideo(mesh,frames):
    """
    Returns frames length array of id seen at each time step of a video
    """
    t = np.linspace(0,2*np.pi,frames)
    sine = np.sin(t)
    cosine = np.cos(t)
    data = []
    for s,c in zip(sine,cosine):
        data.append(genId(mesh,[2*s+2*c,-2*s+2*c,-2*c+2*s])) #do calculus
    #1,-1,-1 and 1,1,1 for wobble
    return data


def getCurrent(data_path,truth_path):
    try:
        with open(data_path,"rb") as f:
            data = pickle.load(f)
    except:
        data = {}
            
    try:
        with open(truth_path,"rb") as f:
            truth = pickle.load(f)
    except:
        truth = {}
    return data,truth,len(data.keys())

def saveCurrent(data_path,truth_path,data,truth):
    with open(data_path,"wb") as f:
        pickle.dump(data, f)
    with open(truth_path,"wb") as f:
        pickle.dump(truth, f)


if __name__ == "__main__":
    cube_path = '/data/vision/billf/scratch/ztzhang/data/non-rigid/3d_models/cube_simple.obj'
    mesh = trimesh.load(cube_path)  
    data_path = './data2/cube_data.pickle'
    truth_path = './data2/cube_truth.pickle'
    data_path_short = './data2/cube_data_short.pickle'
    truth_path_short = './data2/cube_truth_short.pickle'
    point_positions = mesh.triangles_center

    data,truth,currnum = getCurrent(data_path,truth_path)
    # saveCurrent(data_path_short,truth_path_short,dict((k,data[k]) for k in range(0,1000)),dict((k,truth[k]) for k in range(0,1000)))

    ###codefor making new examples
    newadj = makeUndirect(mesh)
    # data,truth,currnum = getCurrent(data_path,truth_path)
    for i in range(currnum,currnum+100000):
        print(i)
        edge, coordinates, answer, campos = messageData(mesh,newadj,point_positions)
        # print(type(edge[0]))
        data[i] = {}
        data[i]['edge'] = edge
        data[i]['coor'] = coordinates
        data[i]['cam'] = campos
        truth[i] = answer
        if i %50000==0: 
            saveCurrent(data_path,truth_path,data,truth)
            # print(len(data))
            





    
    


    # with open(edges_path,"wb+") as f:
    #     pickle.dump(graph, f)

    # tr(labels, f)
    
    # print(global_to_camera(np.array([[1,1,1],[2,2,2]]),np.array([2,2,2])))
    # x,idset,maxid = genOne(mesh,1)
    # newx, edge, label = removePoints(x,['a'],idset,maxid,1)
    # print(newx,x)

  
