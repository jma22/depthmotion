import numpy as np
import trimesh
from trimesh import viewer
import numpy as np
import pickle
from camera import Camera
import matplotlib.pyplot as plt
import torch
from torch.utils import data
import random
import math
import matplotlib.pyplot as plt
import networkx as nx
# from classes import *
from torch.utils.data import DataLoader
import pickle
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from dgl import DGLGraph
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import sys
sys.path.append('..')
from helpers import *
import os

    
    
###############################
### neural network training
##############################
def make_shuffled_idx(end,start=0):
    """
    output: iterater
    """
    return np.random.permutation(list(range(start,end)))

def get_data_tensor(idx,device, tensor = True):
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
    
###############################
### Visualize
##############################
def get_xyzcolor(point_positions,observed_set=None):
    X = point_positions[:,0]
    Y = point_positions[:,1]
    Z = point_positions[:,2]
    temp = np.array([X+Y+Z])
    temp = (temp-temp.min())/(temp.max()-temp.min())
    colors = np.hstack((temp.T,np.zeros((point_positions.shape[0],2))+0.5))
    if observed_set is not None:
        for idx in range(len(point_positions)):
            if idx not in observed_set:
                colors[idx] = np.array([0,1,0])
    return X,Y,Z,colors
    
def show_obj_points(obj_path,elev=-30,azim=125,show=True):
    mesh = trimesh.load(obj_path)  
    point_positions = mesh.triangles_center
    fig = plt.figure(figsize=(15,7))
    ax = fig.add_subplot(121, projection='3d')
    X,Y,Z,colors = get_xyzcolor(point_positions)
    ax.view_init(elev=elev, azim=azim)
    ax.scatter3D(X, Y, Z, c=colors )
    if show:
        plt.show()
    
def show_obj_mesh(obj_path):
    ##need to copy directly into notebook
    mesh = trimesh.load(obj_path) 
    viewer.notebook.scene_to_notebook(mesh, height=500)


def animate_data(pickle_path):
    pass



def animate_one_frame(obj_path,elev=0,save_dir = None):
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-1.5,1.5)
    ax.set_ylim(-1.5,1.5)
    ax.set_zlim(-1.5,1.5)
    mesh = trimesh.load(obj_path)  
    point_positions = mesh.triangles_center
    X,Y,Z,colors = get_xyzcolor(point_positions)
    def init():
        ax.scatter3D(X, Y, Z, c=colors )
        return fig

    def animate(i):
        ax.view_init(elev=elev, azim=i*4)
        return fig

    # Animate
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=90, interval=20)
    if save_dir is not None:
        anim.save(save_dir, writer = "ffmpeg",fps=30, extra_args=['-vcodec', 'libx264'])
    return anim

def animate_folder(folder_path,frames=90,elev=-30,azim=125,save_dir=None):
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-1.5,1.5)
    ax.set_ylim(-1.5,1.5)
    ax.set_zlim(-1.5,1.5)
    filename = os.path.join(folder_path,"frame_{}.obj".format("1".zfill(6)))
    mesh = trimesh.load(filename)  
    point_positions = mesh.triangles_center
    X,Y,Z,colors = get_xyzcolor(point_positions)
    def init():
        ax.scatter3D(X, Y, Z, c=colors )
        ax.view_init(elev=elev, azim=azim)
        return fig
    def animate(i):
        print(i)
        plt.cla()
        filename = os.path.join(folder_path,"frame_{}.obj".format(str(i+1).zfill(6)))
        mesh = trimesh.load(filename)  
        point_positions = mesh.triangles_center
        X,Y,Z,colors = get_xyzcolor(point_positions)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-1.5,1.5)
        ax.set_ylim(-1.5,1.5)
        ax.set_zlim(-1.5,1.5)
        ax.scatter3D(X, Y, Z, c=colors )
        ax.view_init(elev=elev, azim=azim)
        return fig
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=frames, interval=20)
    if save_dir is not None:
        anim.save(save_dir, writer = "ffmpeg",fps=30, extra_args=['-vcodec', 'libx264'])
    return anim
                            
            
    
    

###############################
### Data generation
##############################

def obj_to_num(obj_path,camera_coordinate):
    """
    @input: mesh_path, camera coordinate as 3x0 array
    @output: all points,set of visible ids
    """
    mesh = trimesh.load(obj_path)  
    point_positions = mesh.triangles_center
    features=global_to_camera(point_positions,first_xyz)
    seen =genId(mesh,camera_coordinate)
    return features, seen
    

def folder_to_num():
    pass

def make_dist_dict(obj_path):
    """
    @output: dict of lsit of closect node ids
    """
    def compute_dist(triple1,triple2):
        return ((triple1[0]-triple2[0])**2+(triple1[1]-triple2[1])**2+(triple1[2]-triple2[2])**2)**0.5
    mesh = trimesh.load(obj_path) 
    allpoints = mesh.triangles_center
    dist_dict = {i:[] for i in range(allpoints.shape[0])}
    for i in range(allpoints.shape[0]):
        print(i)
        for j in range(allpoints.shape[0]):
            if j==i:
                continue
            else:
                dist_dict[i].append((j,compute_dist(allpoints[i,:],allpoints[j,:])))      
    for i in range(allpoints.shape[0]):
        dist_dict[i].sort(key=lambda x:x[1])
    print(dist_dict[0])

    return dist_dict
                  



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


def normalize(vector):
    norm = np.linalg.norm(vector)
    return vector/norm

def global_to_camera(coordinate,camera_coordinate):
    """
    convert coordinates to camera coordinate
    """
    x = np.array([-camera_coordinate[0],-camera_coordinate[1],-camera_coordinate[2]])
    y = np.array([-camera_coordinate[0],-camera_coordinate[1], camera_coordinate[0]**2/camera_coordinate[2] + camera_coordinate[1]**2/camera_coordinate[2]])
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
    camera.lookat(cam_pos,np.array([0,0,0]),np.array([0,0,1]))
    o,d,end = camera.generate_rays() # generates vectors and a bunch of rays
    inc=trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    o = o.reshape(-1,3)
    d = d.reshape(-1,3)
    fid = inc.intersects_first(o,d) # list of indices of hit triangles
    alist = set([x for x in fid if x!=-1])
    return alist
