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

