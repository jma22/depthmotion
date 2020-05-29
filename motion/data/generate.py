import bpy
from random import random

def random_euler():
    return (random()*2*3.14,
            random()*2*3.14,
            random()*2*3.14)
            
            
bpy.ops.mesh.primitive_cube_add(
    location=[0,0,0],
    rotation=random_euler())
ob = bpy.context.object
frame_num = 0

for i in range(0,4):
    bpy.context.scene.frame_set(frame_num)
    ob.rotation_euler = random_euler()
    ob.keyframe_insert(data_path="rotation_euler",index=-1)
    frame_num+=30
    
bpy.context.scene.frame_end = 90
bpy.ops.export_scene.obj(filepath="./test.obj", check_existing=True, use_animation=False,use_materials=False)