import trimesh
import numpy as np
import pickle
from camera import Camera
import matplotlib.pyplot as plt
import torch
from torch.utils import data
import pickle
import random
import math



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


def getCurrent(data_path):
    try:
        with open(data_path,"rb") as f:
            data = pickle.load(f)
    except:
        data = {}
            
    
    return data,len(data.keys())

def saveCurrent(data_path,data):
    with open(data_path,"wb") as f:
        pickle.dump(data, f)
 

def makeUndirect(mesh):
#### message passing
 ## make edge array [2xn]
 ## make coordinate array [6xn]
    point_adjacencies = mesh.face_adjacency
    point_adjacencies = point_adjacencies.tolist()
    newadj = []
    for pair in point_adjacencies:
        newadj.append((pair[1],pair[0]))
        newadj.append((pair[0],pair[1]))
    return newadj

if __name__ == "__main__":
    cube_path = '/data/vision/billf/scratch/ztzhang/data/non-rigid/3d_models/cube_simple.obj'
    mesh = trimesh.load(cube_path)  
    data_path = '../data/move_cube_data.pickle'
    point_positions = mesh.triangles_center

    data,currnum = getCurrent(data_path)
    # saveCurrent(data_path_short,truth_path_short,dict((k,data[k]) for k in range(0,1000)),dict((k,truth[k]) for k in range(0,1000)))

    ###codefor making new examples
    newadj = makeUndirect(mesh)
    data,currnum = getCurrent(data_path)
    for i in range(currnum,currnum+100001):
        theta = random.random()*2*math.pi
        phi = math.acos(2*random.random()-1)
        rho = random.random()*2+3
        first_xyz = np.array([math.cos(phi) * math.cos(theta) * rho,math.cos(phi) * math.sin(theta) * rho,math.sin(phi) * rho])
        # lsit of frame number ofdata
        features = []
        cam = []
        seen = []
        print(i)
        for frame in range(20):
            features.append(global_to_camera(point_positions,first_xyz))
            cam.append(first_xyz)
            seen.append(genId(mesh,first_xyz))
            theta += (2*random.random()-1)*2*math.pi/18
            phi += (2*random.random()-1)*2*math.pi/18
            rho += (2*random.random()-1)*0.2
            first_xyz = np.array([math.cos(phi) * math.cos(theta) * rho,math.cos(phi) * math.sin(theta) * rho,math.sin(phi) * rho])
            
            
        data[i] = {}
        #incomplete data
        
        data[i]['visible'] = seen
#         complete data
        data[i]['edge'] = newadj
        data[i]['coor'] = features
        data[i]['cam'] = cam
        if i %500==0: 
            saveCurrent(data_path,data)
            # print(len(data))
            





    
    


    # with open(edges_path,"wb+") as f:
    #     pickle.dump(graph, f)

    # tr(labels, f)
    
    # print(global_to_camera(np.array([[1,1,1],[2,2,2]]),np.array([2,2,2])))
    # x,idset,maxid = genOne(mesh,1)
    # newx, edge, label = removePoints(x,['a'],idset,maxid,1)
    # print(newx,x)

  
