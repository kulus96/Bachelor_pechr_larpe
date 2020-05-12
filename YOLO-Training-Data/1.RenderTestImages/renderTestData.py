"""Examples of using pyrender for viewing and offscreen rendering.
"""
import pyglet
pyglet.options['shadow_window'] = False
#import os
#os.environ['PYOPENGL_PLATFORM'] == 'egl'
from PIL import Image
import numpy as np
import trimesh
import cv2
import random
import progressbar
from pyrender import PerspectiveCamera,\
                     DirectionalLight, SpotLight, PointLight,\
                     Mesh, Scene,\
                     Viewer, OffscreenRenderer, material


# Fuze trimesh
random.seed(a=None)
def renderObj(objPath,xL=int(random.uniform(0, 360)),yL=int(random.uniform(0, 360)),zL = 0,xObj=int(random.uniform(0, 360)),yObj=int(random.uniform(0, 360)),zObj=int(random.uniform(0, 360)),xCam=0,yCam=0,zCam=0,lColorR=random.uniform(0, 100)*0.01,lColorG=random.uniform(0, 100)*0.01,lColorB=random.uniform(0, 100)*0.01):
    fuze_trimesh = trimesh.load(objPath)
    mat = material.MetallicRoughnessMaterial(baseColorFactor=[ 1.000, 0.766, 0.336, 1.0 ], metallicFactor=0.7,roughnessFactor= 0.4)
    fuze_mesh = Mesh.from_trimesh(fuze_trimesh, material=mat,smooth=False)
    
    #fuze_mesh = Mesh.from_trimesh(fuze_trimesh,smooth=False)
    #fuze_trimesh = trimesh.load(objPath)
    
    direc_l = DirectionalLight(color=[lColorR,lColorG,lColorB], intensity=3.0)
    direc_l2 = DirectionalLight(color=[lColorB,lColorG,lColorR], intensity=3.0)
    
    cam = PerspectiveCamera(yfov=(np.pi / 3.0))
    
    cam_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 200.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
       
    aX = xL
    aY = yL
    aZ = zL
                
    rotLight = (np.array([
        [np.cos(aY)*np.cos(aZ),  -np.cos(aY)*np.sin(aZ), np.sin(aY),   0.0],
        [np.cos(aX)*np.sin(aZ)+np.sin(aX)*np.sin(aY)*np.cos(aZ),  np.cos(aX)*np.cos(aZ)-np.sin(aX)*np.sin(aY)*np.sin(aZ), -np.sin(aX)*np.cos(aY),   0.0],
        [np.sin(aX)*np.sin(aZ)-np.cos(aX)*np.sin(aY)*np.cos(aZ),  np.sin(aX)*np.cos(aZ)+np.cos(aX)*np.sin(aY)*np.sin(aZ), np.cos(aX)*np.cos(aY),   0.0],
        [0.0,  0.0, 0.0,   1.0], 
    ]))
    
    aX = xObj
    aY = yObj
    aZ = zObj
    rotObj = (np.array([
        [np.cos(aY)*np.cos(aZ),  -np.cos(aY)*np.sin(aZ), np.sin(aY),   0.0],
        [np.cos(aX)*np.sin(aZ)+np.sin(aX)*np.sin(aY)*np.cos(aZ),  np.cos(aX)*np.cos(aZ)-np.sin(aX)*np.sin(aY)*np.sin(aZ), -np.sin(aX)*np.cos(aY),   0.0],
        [np.sin(aX)*np.sin(aZ)-np.cos(aX)*np.sin(aY)*np.cos(aZ),  np.sin(aX)*np.cos(aZ)+np.cos(aX)*np.sin(aY)*np.sin(aZ), np.cos(aX)*np.cos(aY),   0.0],
        [0.0,  0.0, 0.0,   1.0], 
    ]))

    aX = xCam
    aY = yCam
    aZ = zCam
    
    scene = Scene(ambient_light=[0.01, 0.01, 0.01],bg_color=[0, 0, 0])
    #scene = Scene()#scene = Scene(ambient_light=np.array([0.01, 0.01, 0.01, 1.0]))
    scene.add(fuze_mesh,pose=rotObj)       
                
    scene.add(direc_l, pose=rotLight)
    scene.add(direc_l2, pose=cam_pose)
    
    #v = Viewer(scene, shadows=False)
    scene.add(cam, pose=cam_pose)
                
    #==============================================================================
    # Rendering offscreen from that camera
    #===========================================================================
    #Viewer(scene, use_raymond_lighting=True)
    w = 250
    h = 250
    r = OffscreenRenderer(viewport_width=w, viewport_height=h)
    color, depth = r.render(scene)
    mask = np.zeros((h,w,1), np.uint8)
    for x in range(w):
        for y in range(h):
            if depth[x,y] > 0:
                mask[x,y] = 1
            else:
                mask[x,y] = 0
    r.delete()
    name = "cubev2"+"_"+str(xObj)+"_"+str(yObj)+"_"+str(zObj)+".png"
    cv2.imwrite("Images/"+name,color)
    
    return color, mask
if __name__== "__main__":
    objPath = 'Cube_center.obj'
    for progress in progressbar.progressbar(range(100)):
	    renderObj(objPath,int(random.uniform(0,360)),int(random.uniform(0,360)),0,int(random.uniform(0,360)),int(random.uniform(0,360)),int(random.uniform(0,360)),0,0,0,random.uniform(0,100)*0.01,random.uniform(0, 100)*0.01,random.uniform(0,100)*0.01)
print("DONE")
