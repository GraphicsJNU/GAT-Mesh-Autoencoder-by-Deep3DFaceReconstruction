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
    # from mtcnn import MTCNN
    #
    # for i in range(20):
    #     for j in range(1, 151):
    #         if not os.path.exists(f'./datasets/Facewarehouse/detections/{j}_{i}.txt'):
    #             img = cv2.cvtColor(cv2.imread(f'./datasets/Facewarehouse/{j}_{i}.png'), cv2.COLOR_BGR2RGB)
    #             detector = MTCNN()
    #             result = detector.detect_faces(img)
    #             if len(result) > 1:
    #                 result.sort(key=lambda x:x['box'][2] * x['box'][3], reverse=True)
    #             elif len(result) == 0:
    #                 print(j, i)
    #                 exit(0)
    #             lms = result[0]['keypoints']
    #             seq = [lms['left_eye'], lms['right_eye'], lms['nose'], lms['mouth_left'], lms['mouth_right']]
    #             with open(f'./datasets/Facewarehouse/detections/{j}_{i}.txt', 'w') as f:
    #                 for lm in seq:
    #                     f.write(f'{lm[0]} {lm[1]}\n')

    data_prepare([os.path.join(opt.data_root, folder) for folder in opt.img_folder], opt.mode)
