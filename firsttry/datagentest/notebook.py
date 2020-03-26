
# coding: utf-8

# In[103]:


import trimesh
import numpy as np

class Camera():
    # camera coordinates: y up, z forward, x right.
    # consistent with blender definitions.
    # res = [w,h]
    def __init__(self):
        self.position = np.array([1.6, 0, 0])
        self.rx = np.array([0, 1, 0])
        self.ry = np.array([0, 0, 1])
        self.rz = np.array([1, 0, 0])
        self.focal_length = 0.05
        self.res = [600,600]
        # set the diagnal to be 35mm film's diagnal
        self.set_diagal((0.036**2 + 0.024**2)**0.5)

    def rotate(self, rot_mat):
        self.rx = rot_mat[:, 0]
        self.ry = rot_mat[:, 1]
        self.rz = rot_mat[:, 2]

    def move_cam(self, new_pos):
        self.position = new_pos

    def set_pose(self, inward, up):
        # print(inward)
        # print(up)
        self.rx = np.cross(up, inward)
        self.ry = np.array(up)
        self.rz = np.array(inward)
        self.rx = self.rx / np.linalg.norm(self.rx)
        self.ry = self.ry/np.linalg.norm(self.ry)
        self.rz = self.rz/np.linalg.norm(self.rz)

    def set_diagal(self, diag):
        h_relative = self.res[1] / self.res[0]
        self.sensor_width = np.sqrt(diag**2 / (1 + h_relative**2))

    def lookat(self, orig, target, up):
        self.position = np.array(orig)
        target = np.array(target)
        inward = self.position - target
        right = np.cross(up, inward)
        up = np.cross(inward, right)
        self.set_pose(inward, up)
        
    def generate_rays(self):
        orig = np.zeros([self.res[0],self.res[1],3])
        orig[:,:,0] = self.position[0]
        orig[:,:,1] = self.position[1]
        orig[:,:,2] = self.position[2]
        w = self.sensor_width
        h_linspace = np.linspace(-w / 2, w / 2, self.res[0])                        
        w_linspace = np.linspace(-w / 2, w / 2, self.res[1])
        H,W = np.meshgrid(h_linspace,w_linspace)
        H = H[...,None]
        W = W[..., None]
        ends = (self.position-self.rz*self.focal_length)+H*self.ry[None,None,:] + W*self.rx[None,None,:]
        direction = (ends-orig)/np.linalg.norm(ends-orig,axis=2,keepdims=True)
        return orig,direction,ends

cube_path = '/data/vision/billf/scratch/ztzhang/data/non-rigid/3d_models/cube_simple.obj'
mesh = trimesh.load(cube_path)


# In[104]:


# use faces as elements:
point_positions = mesh.triangles_center
point_adjacencies = mesh.face_adjacency


# In[108]:


point_adjacencies.shape


# In[123]:


camera = Camera()
camera.lookat(np.array([5,5,5]),np.array([0,0,0]),np.array([0,0,1]))
o,d,end = camera.generate_rays()
inc=trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
o = o.reshape(-1,3)
d = d.reshape(-1,3)
fid = inc.intersects_first(o,d)


# In[119]:


(point_positions[0,:]+1)/2


# In[127]:


camera.position

