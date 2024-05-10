"""This script is the test script for Deep3DFaceRecon_pytorch
"""

import os
import os.path as osp
import cv2
import copy
import pickle
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import MyVisualizer
from util.preprocess import align_img, estimate_norm, estimate_norm_torch
from PIL import Image
import numpy as np
from util.load_mats import load_lm3d
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from data.base_dataset import get_transform
from models.base_model import BaseModel
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_data_path(root='examples'):
    img_paths = [os.path.join(root, i) for i in sorted(os.listdir(root)) if i.endswith('png') or i.endswith('jpg')]

    paths = [path[:path.rfind(os.sep)] for path in img_paths]
    f_names = [path[path.rfind(os.sep) + 1:] for path in img_paths]
    lm68_paths = [osp.join(paths[i], 'landmarks', f_names[i][:f_names[i].rfind('.')] + '.txt') for i in
                  range(len(img_paths))]
    lm5_paths = [osp.join(paths[i], 'detections', f_names[i][:f_names[i].rfind('.')] + '.txt') for i in
                 range(len(img_paths))]
    msk_paths = [osp.join(paths[i], 'mask', f_names[i]) for i in range(len(img_paths))]

    return img_paths, lm68_paths, lm5_paths, msk_paths


def get_path_info(img_path):
    path = img_path[:img_path.rfind(os.path.sep)]
    file_name = img_path[img_path.rfind(os.path.sep) + 1:]
    extension = img_path[img_path.rfind('.') + 1:]
    img_name = file_name.replace('.' + extension, '')

    return path, file_name, img_name, extension


def parse_label(label):
    return torch.tensor(np.array(label).astype(np.float32))


def read_data(img_path, lm68_path, lm5_path, msk_path, lm3d_std, device):
    # to RGB
    raw_img = Image.open(img_path).convert('RGB')
    raw_msk = Image.open(msk_path).convert('RGB')
    raw_lm68 = np.loadtxt(lm68_path).astype(np.float32).reshape(-1, 2)
    raw_lm5 = np.loadtxt(lm5_path).astype(np.float32).reshape(-1, 2)

    # _, H = raw_img.size
    # raw_lm68[:, -1] = H - 1 - raw_lm68[:, -1]
    _, img, lm, msk = align_img(raw_img, raw_lm68, lm3d_std, raw_msk)

    _, H = raw_img.size
    lm[:, -1] = H - 1 - lm[:, -1]
    M = estimate_norm(lm, H)
    transform = get_transform()
    img_tensor = transform(img)
    lm_tensor = parse_label(lm)
    msk_tensor = transform(msk)[:1, ...]
    M_tensor = parse_label(M)

    return {
        'img': img_tensor.unsqueeze(0).to(device),
        'lm': lm_tensor.unsqueeze(0).to(device),
        'msk': msk_tensor.unsqueeze(0).to(device),
        'M': M_tensor.unsqueeze(0).to(device)
    }


def torchToImage(tensorImage, toImage):
    # torchToCvImage(model.pred_face.detach().cpu().squeeze(0), toImage)
    return cv2.cvtColor(np.array(toImage(torch.clamp(tensorImage, 0, 1))), cv2.COLOR_BGR2RGB)


def main(device, opt, name='examples'):
    if device != torch.device('cpu'):
        torch.cuda.set_device(device)

    s, e = 1, 20
    graph_data = {}
    if os.path.exists('color_loss_graph.pkl'):
        with open('color_loss_graph.pkl', 'rb') as f:
            graph_data = pickle.load(f)
    for epoch in range(s, e + 1):
        if len(graph_data.keys()) > 0 and len(graph_data[list(graph_data.keys())[0]]) < epoch:
            models = {}
            # models['proposal'] = create_model(opt)
            opts = {'proposal-cheb-cheb-True-False-500': copy.copy(opt), 'proposal-gat-cheb-True-False-500': copy.copy(opt),
                    'proposal-cheb-gat-True-False-500': copy.copy(opt), 'proposal-gat-gat-True-False-500': copy.copy(opt),
                    'proposal-gat-cheb-False-False-0': copy.copy(opt),
                    'proposal-gat-cheb-False-False-500': copy.copy(opt), 'proposal-gat-cheb-True-True-500': copy.copy(opt),
                    'cmd-gat-cheb-True-True-500': copy.copy(opt)}
            opts = {'proposal-gat-cheb-True-False-500': copy.copy(opt), 'proposal-gat-cheb-False-False-0': copy.copy(opt)}
            # opts = {}
            for key in opts.keys():
                elements = key.split('-')
                model_name, enc_conv, dec_conv, shape_fc_train, texture_fc_train, ae_pretrained_epoch = key.split('-')
                opts[key].epoch = epoch
                opts[key].model = elements[0]
                opts[key].name = elements[0]
                opts[key].enc_conv = elements[1]
                opts[key].dec_conv = elements[2]
                opts[key].ae_dim = 128 if elements[0] == 'proposal' else 256
                opts[key].shape_fc_train = elements[3] == 'True'
                if model_name != 'cmd':
                    opts[key].texture_fc_train = elements[4] == 'True'
                opts[key].ae_pretrained_epoch = int(elements[5])
                models[key] = create_model(opts[key])
                if model_name == 'cmd':
                    models[key].param_name = ('_ae_{:d}_{}_{:d}_{}_{:d}_{:.0e}_{}_{}'
                                              .format(opts[key].ae_dim,
                                                      opts[key].enc_conv, opts[key].enc_k, opts[key].dec_conv,
                                                      opts[key].dec_k,
                                                      opts[key].w_kl, opts[key].ae_pretrained_epoch,
                                                      opts[key].shape_fc_train))
                else:
                    models[key].param_name = ('_ae_{:d}_{}_{:d}_{}_{:d}_{:.0e}_{}_{}_{}'
                                              .format(opts[key].ae_dim,
                                                      opts[key].enc_conv, opts[key].enc_k, opts[key].dec_conv,
                                                      opts[key].dec_k,
                                                      opts[key].w_kl, opts[key].ae_pretrained_epoch,
                                                      opts[key].shape_fc_train, opts[key].texture_fc_train))

            # opts['proposal'] = opt
            # opts['Deep3DFaceReconstruction'] = copy.copy(opt)
            # opts['Deep3DFaceReconstruction'].epoch = 20
            # opts['Deep3DFaceReconstruction'].model = 'facerecon'
            # opts['Deep3DFaceReconstruction'].name = 'original'
            # opts['Deep3DFaceReconstruction'].bfm_model = 'BFM_model_front.mat'
            # models['Deep3DFaceReconstruction'] = create_model(opts['Deep3DFaceReconstruction'])
            # models['Deep3DFaceReconstruction'].param_name = ''

            # if opt.name == 'original':
            #     models['proposal'].param_name = ''
            # elif opt.name == 'proposal':
            #     models['proposal'].param_name = ('_ae_{:d}_{}_{:d}_{}_{:d}_{:.0e}_{}_{}_{}'
            #                                      .format(opt.ae_dim,
            #                                              opt.enc_conv, opt.enc_k, opt.dec_conv, opt.dec_k,
            #                                              opt.w_kl, opt.ae_pretrained_epoch,
            #                                              opt.shape_fc_train, opt.texture_fc_train))
            # elif opt.name == 'cmd':
            #     models['proposal'].param_name = ('_ae_{:d}_{}_{:d}_{}_{:d}_{:.0e}_{}_{}'
            #                                      .format(opt.ae_dim,
            #                                              opt.enc_conv, opt.enc_k, opt.dec_conv, opt.dec_k,
            #                                              opt.w_kl, opt.ae_pretrained_epoch,
            #                                              opt.shape_fc_train))
            # else:
            #     raise Exception('Not exist model')

            visualizer = MyVisualizer(opt)
            if not os.path.exists(
                    osp.join(visualizer.img_dir, name.split(os.path.sep)[-1], 'epoch_%s_%06d' % (opt.epoch, 0))):
                os.makedirs(osp.join(visualizer.img_dir, name.split(os.path.sep)[-1], 'epoch_%s_%06d' % (opt.epoch, 0)))
            for key in models.keys():
                models[key].setup(opts[key], models[key].param_name)
                models[key].device = device
                models[key].parallelize()
                models[key].eval()

            img_paths, lm68_paths, lm5_paths, msk_paths = get_data_path(name)
            lm3d_std = load_lm3d(opt.bfm_folder)
            # all_feat_loss, all_color_loss = {'PRNet': [], '3DDFA': []}, {'PRNet': [], '3DDFA': []}
            all_feat_loss, all_color_loss = {}, {}
            if not os.path.exists(osp.join(name, 'result')):
                os.mkdir(osp.join(name, 'result'))

            if not osp.exists(osp.join(name, 'result', 'original')):
                os.mkdir(osp.join(name, 'result', 'original'))
            if not osp.exists(osp.join(name, 'result', 'proposal')):
                os.mkdir(osp.join(name, 'result', 'proposal'))
            if not osp.exists(osp.join(name, 'result', 'Deep3DFaceReconstruction')):
                os.mkdir(osp.join(name, 'result', 'Deep3DFaceReconstruction'))

            for key in models.keys():
                all_feat_loss[key] = []
                all_color_loss[key] = []
                if not osp.exists(osp.join(name, 'result', key)):
                    os.mkdir(osp.join(name, 'result', key))

            toTensor = transforms.ToTensor()
            toImage = transforms.ToPILImage()
            for i, (img_path, lm68_path, lm5_path, msk_path) in tqdm(enumerate(zip(img_paths, lm68_paths, lm5_paths, msk_paths))):
                path, file_name, img_name, extension = get_path_info(img_path)
                save_name = copy.copy(img_name) + opt.net_type
                # print(i, img_path)

                if not os.path.isfile(lm5_path):
                    print("%s is not found !!!" % lm5_path)
                    continue

                render_img_path = os.path.join(path, 'result', 'original', img_name + '_aligned.' + extension)
                data = read_data(img_path, lm68_path, lm5_path, msk_path, lm3d_std, device)
                input_data = {
                    'imgs': data['img'],
                    'lms': data['lm'],
                    'M': data['M']
                }

                render_img_path = os.path.join(path, 'result', 'original', img_name + '_aligned.' + extension)
                if not os.path.exists(render_img_path):
                    cv2.imwrite(render_img_path, torchToImage(data['img'].detach().cpu().squeeze(0), toImage))

                for key in models.keys():
                    models[key].set_input(input_data)  # unpack data from data loader
                    models[key].test()  # run inference
                    render_img_path = os.path.join(path, 'result', key, img_name + '_aligned.' + extension)
                    face_image = torchToImage(models[key].pred_face.detach().cpu().squeeze(0), toImage)

                    cv2.imwrite(render_img_path, face_image)
                    lms = models[key].pred_lm.detach().cpu().squeeze(0).numpy()
                    # M_tensor_2 = estimate_norm_torch(models[key].pred_lm, data['img'].shape[-2]).to(device)
                    # M_tensor_2 = models[key].trans_m
                    # recog_loss = models['proposal'].get_recog_loss(data['img'], models[key].pred_face, M_tensor_2, models[key].trans_m).item()

                    np.savetxt(osp.join(path, 'result', key, img_name + '_aligned.txt'), lms)

                # models['proposal'].save_mesh(
                #     osp.join(visualizer.img_dir, name.split(os.path.sep)[-1], 'epoch_%s_%06d' % (opt.epoch, 0),
                #              save_name + '.obj'), True)  # save reconstruction meshes
                # models['proposal'].save_coeff(osp.join(visualizer.img_dir, name.split(os.path.sep)[-1], 'epoch_%s_%06d' % (opt.epoch, 0), save_name + '.mat'))  # save predicted coefficients

                # visuals = models['proposal'].get_current_visuals()  # get image results
                # visualizer.display_current_results(visuals, 0, opt.epoch, dataset=name.split(os.path.sep)[-1],
                #                                    save_results=True, count=i, name=save_name, add_image=False)

            for key in all_feat_loss.keys():
                if key not in models.keys():
                    models[key] = type('', (), {})()
                    models[key].pred_lm = None

            img_names = []
            for i, (img_path, lm68_path, lm5_path, msk_path) in tqdm(enumerate(zip(img_paths, lm68_paths, lm5_paths, msk_paths))):
                path, _, img_name, extension = get_path_info(img_path)
                img_names.append(img_name)
                # print(i, img_path)

                if not os.path.isfile(lm5_path):
                    print("%s is not found !!!" % lm5_path)
                    continue

                data = read_data(img_path, lm68_path, lm5_path, msk_path, lm3d_std, device)

                recog_losses = {}
                color_losses = {}
                recog_print_str = ''
                color_print_str = ''

                M_tensor = data['M']
                transform = get_transform()
                for model_name in all_feat_loss.keys():
                    # if not isinstance(models[model_name], BaseModel):
                    # if model_name == 'PRNet':
                    #     if osp.exists(osp.join(path, 'result', model_name, img_name + '_aligned_kpt.txt')):
                    #         kpt = np.loadtxt(osp.join(path, 'result', model_name, img_name + '_aligned_kpt.txt'))
                    #         kpt[:, 1] = data['img'].shape[-2] - 1 - kpt[:, 1]
                    #         np.savetxt(osp.join(path, 'result', model_name, img_name + '_aligned.txt'), kpt[:, :2])
                    #         os.remove(osp.join(path, 'result', model_name, img_name + '_aligned_kpt.txt'))
                    models[model_name].pred_lm = torch.from_numpy(
                        np.loadtxt(osp.join(path, 'result', model_name, img_name + '_aligned.txt'))).unsqueeze(0).to(device)

                    M_tensor_2 = estimate_norm_torch(models[model_name].pred_lm, data['img'].shape[-2])
                    M_tensor_2 = M_tensor
                    temp = 'jpg' if not isinstance(models[model_name], BaseModel) else extension

                    render_img_path = os.path.join(path, 'result', model_name, img_name + '_aligned.' + temp)
                    pred_img = transform(np.array(Image.open(render_img_path).convert('RGB'))).unsqueeze(0).to(device)
                    recog_loss = models['proposal-gat-cheb-True-False-500'].get_recog_loss(data['img'], pred_img, M_tensor_2, M_tensor).item()
                    color_loss = models['proposal-gat-cheb-True-False-500'].comupte_color_loss(pred_img, data['img'], data['msk'])
                    # recog_print_str += "{} - {:.3f}, ".format(model_name, recog_loss)
                    # color_print_str += "{} - {:.3f}, ".format(model_name, color_loss)
                    recog_losses[model_name] = recog_loss
                    color_losses[model_name] = color_loss.detach().cpu().item()

                for model_name in all_feat_loss.keys():
                    all_feat_loss[model_name].append(recog_losses[model_name])
                    all_color_loss[model_name].append(color_losses[model_name])

                # print(recog_print_str)
                # print(color_print_str)

            # # candidate = [3, 4, 5, 12, 23, 24, 25]
            label_dict = {
                'Deep3DFaceReconstruction': 'Y.Deng, et al.',
                '3DDFA': '3DDFA_V2',
                'proposal': 'Proposal',
                'proposal-gat-cheb-False-False-0': 'Not Pretrained',
                'proposal-gat-gat-True-False-500': 'GAT-GAT',
                'proposal-cheb-gat-True-False-500': 'GCN-GAT',
                'proposal-cheb-cheb-True-False-500': 'GCN-GCN',
                'proposal-gat-cheb-True-False-500': 'GAT-GCN',
                'proposal-gat-cheb-False-False-500': 'ALL-ALL',
                'proposal-gat-cheb-True-True-500': 'FC-FC',
                'cmd-gat-cheb-True-True-500': 'Single',
            }
            labels = [label_dict[key] if key in label_dict.keys() else key for key in list(all_feat_loss.keys())]
            # plt.bar(labels, [np.square(np.array(all_feat_loss[key])).mean() for key in all_feat_loss.keys()])
            # plt.ylim(0.07, 0.15)
            for i, key in enumerate(all_feat_loss.keys()):
                all_feat_loss[key] = np.sort(np.array(all_feat_loss[key]))
                feat_loss = all_feat_loss[key]

                print("{} avg recog loss median = {:.7f}, mean = {:.7f}, std = {:.7f}".format(
                    label_dict[key] if key in label_dict.keys() else key,
                    feat_loss[len(feat_loss) // 2],
                    np.square(feat_loss).mean(),
                    np.std(feat_loss)))

            print()
            for i, key in enumerate(all_feat_loss.keys()):
                all_color_loss[key] = np.sort(np.array(all_color_loss[key]))
                color_loss = all_color_loss[key]
                print("{} avg color loss median = {:.7f}, mean = {:.7f}, std = {:.7f}".format(
                    label_dict[key] if key in label_dict.keys() else key,
                    color_loss[len(color_loss) // 2],
                    np.square(color_loss).mean(),
                    np.std(color_loss)))

                if key not in graph_data.keys():
                    graph_data[key] = []

                graph_data[key].append(np.square(color_loss).mean())

    with open('color_loss_graph.pkl', 'wb') as f:
        pickle.dump(graph_data, f)

    with open('color_loss_graph.pkl', 'rb') as f:
        graph_data = pickle.load(f)

    labels = ['Not Pretrained', 'Pretrained']
    plt.plot(np.ndarray([i for i in range(s, e + 1)]), np.array(graph_data['proposal-gat-cheb-False-False-0']))
    plt.plot(np.ndarray([i for i in range(s, e + 1)]), np.array(graph_data['proposal-gat-cheb-True-False-500']))

    plt.legend(labels)

    plt.title('Pixel Loss')

    plt.show()

    # 'mesh_verts',
    # 'parm_declarations',
    # 'r',
    # 'shape',
    # 'short_name',
    # 'size',


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    device = torch.device(0)
    # device = torch.device('cpu')
    main(device, opt, opt.img_folder)
