import trimesh
import trimesh.visual
import torch
import numpy as np
from psbody.mesh import Mesh
import os.path as osp
from scipy.io import loadmat
from PIL import Image

base_path = './models/gat/datas'
mesh = Mesh(filename=osp.join(base_path, 'BFM/mu.obj'))
model = loadmat(osp.join('./BFM', 'BFM_model_front.mat'))

vertices = model['meanshape'].reshape(-1, 3)
faces = model['tri'].reshape(-1, 3) - 1

a = trimesh.Trimesh(vertices, faces)

a.export('./BFM/mu.obj')
a.visual = trimesh.visual.create_visual(vertex_colors=model['meantex'].reshape(-1, 3).astype(np.uint8))
texture_image = np.zeros((256, 256), dtype=np.uint8)
c = Image.fromarray(texture_image)
b = a.unwrap(c)

import cv2

b.export('./BFM/temp2.obj')
cv2.imwrite('./BFM/temp.jpg', np.array(c))
