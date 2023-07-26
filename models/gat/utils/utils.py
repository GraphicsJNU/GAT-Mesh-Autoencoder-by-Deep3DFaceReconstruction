import torch
import os
import numpy as np
from typing import List

import os.path as osp
import pickle
from models.gat.utils import mesh_sampling


def makedirs(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def to_sparse(spmat):
    return torch.sparse.FloatTensor(
        torch.from_numpy(np.array([spmat.tocoo().row, spmat.tocoo().col], dtype=np.int64)),
        torch.FloatTensor(spmat.tocoo().data), torch.Size(spmat.tocoo().shape)
    )


def to_edge_index(mat):
    return torch.LongTensor(np.vstack(mat.nonzero()))


def get_texture_info(obj_lines: List[str], img: np.ndarray) -> torch.Tensor:
    h, w, c = img.shape

    # vertex 정보 추출
    vertex_cnt = 0
    for line in obj_lines:
        if line.startswith("v "):
            vertex_cnt += 1

    vt_pos_info = torch.zeros((vertex_cnt, 2), dtype=torch.float32)

    # 정규화된 uv pos 정보 추출
    vt_info = []
    for line in obj_lines:
        if line.startswith("vt "):  # vt 0.301980 0.598360
            vt_info.append([float(info) for info in line[3:-1].split(" ")])

    vt_info = torch.tensor(vt_info)

    # face 정보 추출
    for line in obj_lines:
        if line.startswith("f "):  # f 1/1/0 5197/2/0 10101/3/0 2851/4/0
            mesh = [[int(j) - 1 for j in i.split("/")] for i in line[2:-1].split(" ")]
            for i in range(len(mesh)):
                vt_pos_info[mesh[i][0], :] = vt_info[mesh[i][1], :]

    vt_pos_info[:, 1] = 1 - vt_pos_info[:, 1]

    uv_info = (vt_pos_info * torch.tensor([[w, h]])).type(torch.int64)
    texture_info = img[uv_info[:, 1], uv_info[:, 0], ::-1].astype(np.float32) / 255

    return torch.from_numpy(texture_info)


def get_graph_info(mesh, data_fp):
    transform_fp = osp.join(data_fp, 'transform.pkl')
    if not os.path.exists(transform_fp):
        print('Generating transform matrices...')
        ds_factors = [4, 4, 4, 4]

        with open(transform_fp, 'wb') as fp:
            pickle.dump(mesh_sampling.generate_transform_matrices(mesh, ds_factors), fp)
        print('Done!')
        print('Transform matrices are saved in \'{}\''.format(transform_fp))

    with open(transform_fp, 'rb') as f:
        M, A, D, U, F = pickle.load(f)

    return M, A, D, U, F


def save_obj(vertices, model_path, path):
    with open(model_path, 'r') as f:
        lines = f.readlines()
        info = []
        for line in lines:
            if not line.startswith('v '):
                info.append(line)

    with open(path, "w") as f:
        for vertex in vertices:
            f.write("v {:.6f} {:.6f} {:.6f}\n".format(vertex[0], vertex[1], vertex[2]))

        for line in info:
            f.write(line)
