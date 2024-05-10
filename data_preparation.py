"""This script is the data preparation script for Deep3DFaceRecon_pytorch
"""

import os
import numpy as np
import argparse
from util.detect_lm68 import detect_68p, load_lm_graph
from util.skin_mask import get_skin_mask
from util.generate_list import check_list, write_list
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='datasets', help='root directory for training data')
parser.add_argument('--img_folder', nargs="+", required=True, help='folders of training images')
parser.add_argument('--mode', type=str, default='train', help='train or val')
opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def data_prepare(folder_list, mode):
    lm_sess, input_op, output_op = load_lm_graph(
        './checkpoints/lm_model/68lm_detector.pb')  # load a tensorflow version 68-landmark detector

    print("Detect landmarks")
    for img_folder in folder_list:
        detect_68p(img_folder, lm_sess, input_op, output_op)  # detect landmarks for images
        get_skin_mask(img_folder)  # generate skin attention mask for images

    print("create files that record path to all training data")
    msks_list = []
    for img_folder in folder_list:
        path = os.path.join(img_folder, 'mask')
        msks_list += ['/'.join([img_folder, 'mask', i]) for i in sorted(os.listdir(path)) if 'jpg' in i or
                      'png' in i or 'jpeg' in i or 'PNG' in i]

    imgs_list = [i.replace('mask/', '') for i in msks_list]
    lms_list = [i.replace('mask', 'landmarks') for i in msks_list]
    lms_list = ['.'.join(i.split('.')[:-1]) + '.txt' for i in lms_list]

    lms_list_final, imgs_list_final, msks_list_final = check_list(lms_list, imgs_list,
                                                                  msks_list)  # check if the path is valid
    write_list(lms_list_final, imgs_list_final, msks_list_final, mode=mode)  # save files


if __name__ == '__main__':
    print('Datasets:', opt.img_folder)

    import shutil
    import cv2

    # CelebA 데이터셋 조정
    # with open('./datasets/CelebA/detection/external/list_landmarks_celeba.txt') as f:
    #     for line in tqdm(f.readlines()[2:]):
    #         for i in range(10, 1, -1):
    #             line = line.replace(' ' * i, ' ')
    #         filename = line[:10]
    #         coordinates = [int(i) for i in line[11:].split(" ")]
    #         with open(f'./datasets/CelebA/detection/{filename[:-4]}.txt', 'w') as f2:
    #             for i in range(0, len(coordinates), 2):
    #                 f2.write(f'{coordinates[i]} {coordinates[i + 1]}\n')
    # for i in range(1, 202600):
    #     shutil.move('./datasets/CelebA/remove/%06d.jpg' % i, './datasets/CelebA/%06d.jpg' % i)

    # Facewarehouse 데이터셋 조정
    '''
    from mtcnn import MTCNN
    
    for i in range(20):
        for j in range(1, 151):
            if not os.path.exists(f'./datasets/Facewarehouse/detections/{j}_{i}.txt') or True:
                img = cv2.cvtColor(cv2.imread(f'./datasets/Facewarehouse/{j}_{i}.png'), cv2.COLOR_BGR2RGB)
                detector = MTCNN()
                result = detector.detect_faces(img)
                if len(result) > 1:
                    result.sort(key=lambda x:x['box'][2] * x['box'][3], reverse=True)
                elif len(result) == 0:
                    print(j, i)
                    exit(0)
                lms = result[0]['keypoints']
                seq = [lms['left_eye'], lms['right_eye'], lms['nose'], lms['mouth_left'], lms['mouth_right']]
                with open(f'./datasets/Facewarehouse/detections/{j}_{i}.txt', 'w') as f:
                    for lm in seq:
                        f.write(f'{lm[0]} {lm[1]}\n')
    '''

    # from glob import glob
    # for a in glob('./datasets/test/*.png'):
    #     img = cv2.cvtColor(cv2.imread(a), cv2.COLOR_BGR2RGB)
    #     detector = MTCNN()
    #     path = a[:a.rfind(os.path.sep)]
    #     file_name = a[a.rfind(os.path.sep) + 1:a.rfind(".")]
    #     result = detector.detect_faces(img)
    #     if len(result) > 1:
    #         result.sort(key=lambda x: x['box'][2] * x['box'][3], reverse=True)
    #     elif len(result) == 0:
    #         continue
    #
    #     lms = result[0]['keypoints']
    #     seq = [lms['left_eye'], lms['right_eye'], lms['nose'], lms['mouth_left'], lms['mouth_right']]
    #     with open(f'{path}/detections/{a[a.rfind(os.path.sep) + 1:a.rfind(".")]}.txt', 'w') as f:
    #         for lm in seq:
    #             f.write(f'{lm[0]} {lm[1]}\n')

    # facewarehouse_model_info.mat 변경
    # from scipy.io import loadmat, savemat
    #
    # bfm_folder = './BFM'
    # default_name = 'facewarehouse_model_info.mat'
    # save_name = 'facewarehouse_model_front.mat'
    # facewarehouse_model_info = loadmat(os.path.join(bfm_folder, default_name))
    # data = {'meanshape': facewarehouse_model_info['meanshape'], 'meantex': facewarehouse_model_info['meantex'],
    #         'idBase': facewarehouse_model_info['idBase_PCA'], 'exBase': facewarehouse_model_info['exBase_PCA'],
    #         'texBase': facewarehouse_model_info['texBase_PCA'], 'tri': facewarehouse_model_info['tri'],
    #         'point_buf': facewarehouse_model_info['point_buf'], 'keypoints': facewarehouse_model_info['keypoints_68']}
    #
    # savemat(os.path.join(bfm_folder, save_name), data)

    # 300W-LP 데이터셋 조정
    # from mtcnn import MTCNN
    # from facenet_pytorch import MTCNN
    # from glob import glob
    # import shutil

    '''
    folders = ['300W-LP', 'AFLW2000-3D']
    fps = glob(f'.\\datasets\\300W-3D\\*\\*.mat')
    for fp in tqdm(fps):
        shutil.copy(fp, f'.\\datasets\\300W-3D\\' + fp[fp.rfind('\\') + 1:])
    fps = glob(f'.\\datasets\\300W-3D\\*\\*.jpg')
    for fp in tqdm(fps):
        shutil.copy(fp, f'.\\datasets\\300W-3D\\' + fp[fp.rfind('\\') + 1:])
    '''

    '''
    folders = opt.img_folder
    for folder in folders:
        fps = glob(f'.\\datasets\\{folder}\\*.jpg')
        for fp in tqdm(fps):
            filename = fp[fp.rfind('\\') + 1:]
            filename = filename[:filename.rfind('.')]
            txt_file = f'.\\datasets\\{folder}\\detections\\{filename}.txt'
            if not os.path.exists(txt_file):
                img = cv2.cvtColor(cv2.imread(f'.\\datasets\\{folder}\\{filename}.jpg'), cv2.COLOR_BGR2RGB)
                # boxes, probs, points = mtcnn.detect(img, True)
                # print(points)
                # exit(0)
                # detector = MTCNN()
                result = detector.detect_faces(img)
                if len(result) > 1:
                    result.sort(key=lambda x: x['box'][2] * x['box'][3], reverse=True)
                elif len(result) == 0:
                    continue

                lms = result[0]['keypoints']
                seq = [lms['left_eye'], lms['right_eye'], lms['nose'], lms['mouth_left'], lms['mouth_right']]
                with open(txt_file, 'w') as f:
                    for lm in seq:
                        f.write(f'{lm[0]} {lm[1]}\n')
    '''

    # now 챌린지 데이터
    # from glob import glob
    # from mtcnn import MTCNN
    # detector = MTCNN()
    # for a in glob(f'./datasets/Now/*.jpg'):
    #     file_name = a[a.rfind("\\") + 1:len(a) - 4]
    #     img = cv2.cvtColor(cv2.imread(a), cv2.COLOR_BGR2RGB)
    #     result = detector.detect_faces(img)
    #     lms = result[0]['keypoints']
    #     seq = [lms['left_eye'], lms['right_eye'], lms['nose'], lms['mouth_left'], lms['mouth_right']]
    #     with open('./datasets/Now/detections/' + file_name + ".txt", 'w') as f:
    #         for lm in seq:
    #             f.write(f'{lm[0]} {lm[1]}\n')

    # from glob import glob
    # import shutil
    # import random
    # from psbody.mesh import Mesh
    # from tqdm import tqdm
    # lms_idx = []
    # with open('./datasets/Facewarehouse/landmarks_68.txt', 'r') as f:
    #     lines = f.readlines()
    #     for a in lines:
    #         lms_idx.append(int(a[:-1]) - 1)
    #
    # for a in tqdm(glob('./datasets/Facewarehouse/*.obj')):
    #     m = Mesh(filename=a)
    #     lms_vtx = m.v[lms_idx, :]
    #     with open(a[:-4] + '.txt', 'w') as f:
    #         for b in range(len(lms_idx)):
    #             f.write('{:.6f} {:.6f} {:.6f}\n'.format(lms_vtx[b, 0], lms_vtx[b, 1], lms_vtx[b, 2]))

    # datalist = [[i, j] for i in range(1, 151) for j in range(20)]
    # random.shuffle(datalist)
    # datalist = datalist[:int(len(datalist) * 0.1)]
    # base_path = './datasets/Facewarehouse'
    # now_path = f'{base_path}/Now'
    # scan_path = f'{now_path}/scans'
    # scans_lmks_onlypp_path = f'{now_path}/scans_lmks_onlypp'
    # os.mkdir(scan_path)
    # os.mkdir(scans_lmks_onlypp_path)
    # os.mkdir(f'{now_path}/img')
    # with open(f'{base_path}/Now/imagepathsvalidation.txt', 'w') as f:
    #     for i, j in datalist:
    #         f.write(f'{i}_{j}/selfie/{i}_{j}.png\n')
    #
    #         os.mkdir(f'{scan_path}/{i}_{j}')
    #         shutil.copy(f'{base_path}/{i}_{j}.obj', f'{scan_path}/{i}_{j}/{i}_{j}.obj')
    #
    #         os.mkdir(f'{scans_lmks_onlypp_path}/{i}_{j}')
    #         shutil.copy(f'{base_path}/{i}_{j}.txt', f'{scans_lmks_onlypp_path}/{i}_{j}/{i}_{j}.txt')
    #
    #         shutil.copy(f'{base_path}/{i}_{j}.png', f'{now_path}/img/{i}_{j}.png')
    # exit(0)
    # https://github.com/yfeng95/face3d
    '''
    from scipy.io import loadmat, savemat
    import util.mesh as mesh
    import os.path as osp

    def get_path(file_name):
        return osp.join('./BFM', file_name)

    def process_uv(uv_coords, uv_h=256, uv_w=256):
        uv_coords[:, 0] = uv_coords[:, 0] * (uv_w - 1)
        uv_coords[:, 1] = uv_coords[:, 1] * (uv_h - 1)
        uv_coords[:, 1] = uv_h - uv_coords[:, 1] - 1
        uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1))))  # add z

        return uv_coords

    size = 224
    data = loadmat(get_path('BFM_model_for_300W_3D.mat'))
    index_shape = loadmat(get_path('BFM_exp_idx.mat'))['trimIndex'].astype(np.int32) - 1  # 0 ~ 53490
    front_index_shape = loadmat(get_path('BFM_front_idx.mat'))['idx'].astype(np.int32) - 1  # 0 ~ 53215
    index_shape = index_shape[front_index_shape].reshape(-1)
    # https://github.com/anilbas/3DMMasSTN
    uv_coords = loadmat(get_path('BFM_UV.mat'))['UV']
    uv_coords = uv_coords[index_shape]
    uv_coords = process_uv(uv_coords, size, size)

    faces = data['tri'].astype(np.int32) - 1
    colors = data['meantex'].reshape(-1, 3)
    uv_mean_texture_map = mesh.render.render_colors(uv_coords, faces, colors, size, size, c=3)
    cv2.imwrite('./models/gat/datas/texture.png', cv2.cvtColor(uv_mean_texture_map, cv2.COLOR_BGR2RGB))

    uv_texture_base = np.zeros((size, size, 3, data['texBase'].shape[1]))
    texBase = data['texBase'].reshape(-1, 3, 199)
    for i in range(uv_texture_base.shape[3]):
        uv_texture_base[:, :, :, i] = mesh.render.render_colors(uv_coords, faces, texBase[:, :, i], size, size, c=3)

    save_data = {
        'meanshape': data['meanshape'],
        'meantex': data['meantex'],
        'meantexImg': uv_mean_texture_map.reshape(1, -1),
        'idBase': data['idBase'],
        'exBase': data['exBase'],
        'texBase': data['texBase'],
        'texBaseImg': uv_texture_base.reshape(-1, 199),
        'tri': data['tri'],
        'point_buf': data['point_buf'],
        'tri_mask2': data['tri_mask2'],
        'keypoints': data['keypoints'],
        'frontmask2_idx': data['frontmask2_idx'],
        'skinmask': data['skinmask'],
        'uvcoords': uv_coords
    }

    savemat(get_path('BFM_model_for_300W_3D.mat'), save_data)
    '''

    # with np.load('./BFM/earth.npz') as f:
    #     pos_idx, pos, uv_idx, uv, tex = f.values()

    # data = loadmat(get_path('BFM_model_for_300W_3D.mat'))
    # index_shape = loadmat(get_path('BFM_exp_idx.mat'))['trimIndex'].astype(np.int32) - 1  # 0 ~ 53490
    # front_index_shape = loadmat(get_path('BFM_front_idx.mat'))['idx'].astype(np.int32) - 1  # 0 ~ 53215
    # index_shape = index_shape[front_index_shape].reshape(-1)
    # # https://github.com/anilbas/3DMMasSTN
    # uv_coords = loadmat(get_path('BFM_UV.mat'))['UV']
    # uv_coords = uv_coords[index_shape]
    # save_lines = []
    # with open('./models/gat/datas/mu.obj') as f:
    #     lines = f.readlines()
    #     faces = []
    #     for line in lines:
    #         if line.startswith('v '):
    #             save_lines.append(line)
    #         if line.startswith('f '):
    #             faces.append(line[2:-1].split(" "))
    #
    #     for uv_coord in uv_coords:
    #         save_lines.append(f"vt {uv_coord[0]} {uv_coord[1]}\n")
    #
    #     for face in faces:
    #         save_lines.append(f"f {face[0]}/{face[0]} {face[1]}/{face[1]} {face[2]}/{face[2]}\n")
    #
    # with open('./models/gat/datas/mu2.obj', 'w') as f:
    #     for line in save_lines:
    #         f.write(line)

    # AFLW2000-3D 300W-3D CelebA Facewarehouse
    data_prepare([os.path.join(opt.data_root, folder) for folder in opt.img_folder], opt.mode)
    # data_prepare(['datasets\\Facewarehouse'], opt.mode)
