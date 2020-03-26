
# coding: utf-8

# In[29]:


import trimesh
import numpy as np
cube_path = '/data/vision/billf/scratch/ztzhang/data/non-rigid/3d_models/cube_simple.obj'
mesh = trimesh.load(cube_path)


# In[21]:


# use faces as elements:
point_positions = mesh.triangles_center
point_adjacencies = mesh.face_adjacency


# In[42]:


# calculate visibility:
s=trimesh.scene.scene.Scene(mesh)
s.set_camera(center=(5,5,5),angles=(55*np.pi/180, 0, 140*np.pi/180))
s


# In[43]:


dir(s)

