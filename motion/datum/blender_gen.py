import bpy
from random import random
import os
import sys
sys.path.append('..')
# from helpers import * # dont do this

## to run blender source .bashrc in depthmotion and module load bpy, venv36

def random_euler():
    return (random()*2*3.14,
            random()*2*3.14,
            random()*2*3.14)
def make_new_folder(name,where = ""):
    ## returns path of newfolder
    current = os.getcwd()
    path = os.path.join(current, where) 
    path = os.path.join(path, name) 
    os.mkdir(path)
    return path
    
def genone_rotate(folder_name,frames = 90): 
    """generate 90 objs in folder under "objs" """
    ##delete all
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.select = True
        else:
            obj.select = False
        bpy.ops.object.delete()

    # make torus
    bpy.ops.mesh.primitive_torus_add(
        location=[0,0,0],
        rotation=random_euler())
    ob = bpy.context.object
    frame_num = 0
    
    # add 4 keyframes 30 frames apart to animate
    for i in range(0,4):
        bpy.context.scene.frame_set(frame_num)
        ob.rotation_euler = random_euler()
        ob.keyframe_insert(data_path="rotation_euler",index=-1)
        frame_num+=30
    # cap frames at 90 and save
    bpy.context.scene.frame_end = frames
    path = make_new_folder(folder_name,"objs")
    path  = os.path.join(path,"frame.obj")
    bpy.ops.export_scene.obj(filepath=path, check_existing=False, use_animation=True,use_materials=False)
    
    
if __name__ == "__main__":
    for num in range(10):
        genone_rotate("eg{}".format(num))