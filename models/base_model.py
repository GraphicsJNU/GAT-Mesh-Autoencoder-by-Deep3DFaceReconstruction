"""This script defines the base network model for Deep3DFaceRecon_pytorch
"""

import os
import time
import numpy as np
import torch
from torch.nn import SyncBatchNorm, DataParallel
from torch.nn.parallel import DistributedDataParallel
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks

from util import util
from util.nvdiffrast import MeshRenderer
from util.preprocess import estimate_norm_torch

import trimesh
from scipy.io import savemat
from .losses import perceptual_loss, photo_loss, reg_loss, reflectance_loss, landmark_loss


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.isTrain = opt.isTrain
        self.device = torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.parallel_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'

    @staticmethod
    def dict_grad_hook_factory(add_func=lambda x: x):
        saved_dict = dict()

        def hook_gen(name):
            def grad_hook(grad):
                saved_vals = add_func(grad)
                saved_dict[name] = saved_vals

            return grad_hook

        return hook_gen, saved_dict

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """

        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt, param_name=''):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
            param_name (str) -- footer of epoch file
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

        if not self.isTrain:
            self.load_networks(opt.epoch, param_name)

        # self.print_networks(opt.verbose)

    def parallelize(self, convert_sync_batchnorm=True):
        if not self.opt.use_ddp:
            for name in self.parallel_names:
                if isinstance(name, str):
                    module = getattr(self, name)
                    setattr(self, name, module.to(self.device))
        else:
            for name in self.model_names:
                if isinstance(name, str):
                    module = getattr(self, name)
                    if convert_sync_batchnorm:
                        module = SyncBatchNorm.convert_sync_batchnorm(module)
                    setattr(self, name, DistributedDataParallel(module.to(self.device),
                                                                device_ids=[self.device.index],
                                                                find_unused_parameters=True,
                                                                broadcast_buffers=True))

            # DistributedDataParallel is not needed when a module doesn't have any parameter that requires a gradient.
            for name in self.parallel_names:
                if isinstance(name, str) and name not in self.model_names:
                    module = getattr(self, name)
                    setattr(self, name, module.to(self.device))

        # put state_dict of optimizer to gpu device
        if self.opt.phase != 'test':
            if self.opt.continue_train:
                for optim in self.optimizers:
                    for state in optim.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(self.device)

    def data_dependent_initialize(self, data):
        pass

    def train(self):
        """Make models train mode"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.train()

    def eval(self):
        """Make models eval mode"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()

    def test(self, is_compute_visuals=True):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            if is_compute_visuals:
                self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self, name='A'):
        """ Return image paths that are used to load current data"""
        return self.image_paths if name == 'A' else self.image_paths_B

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)[:, :3, ...]

        return visual_ret

    def get_current_losses(self):
        """Return training losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(
                    getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number

        return errors_ret

    def save_networks(self, epoch, param_name=''):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
            param_name (str) -- footer of epoch file to save
        """
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        save_filename = 'epoch_%s%s.pth' % (epoch, param_name)
        save_path = os.path.join(self.save_dir, save_filename)

        save_dict = {}
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                if isinstance(net, DataParallel) or isinstance(net, DistributedDataParallel):
                    net = net.module
                save_dict[name] = net.state_dict()
        for i, optim in enumerate(self.optimizers):
            save_dict['opt_%02d' % i] = optim.state_dict()

        for i, sched in enumerate(self.schedulers):
            save_dict['sched_%02d' % i] = sched.state_dict()

        torch.save(save_dict, save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def optimizer_to(self, optim, device):
        for param in optim.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)

    def load_networks(self, epoch, param_name=''):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
            param_name (str) -- footer of epoch file
        """
        if self.opt.isTrain and self.opt.pretrained_name is not None:
            load_dir = os.path.join(self.opt.checkpoints_dir, self.opt.pretrained_name)
        else:
            load_dir = self.save_dir
        load_filename = 'epoch_%s%s.pth' % (epoch, param_name)
        load_path = os.path.join(load_dir, load_filename)
        state_dict = torch.load(load_path, map_location=self.device)
        print('loading the model from %s' % load_path)

        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                if isinstance(net, DataParallel):
                    net = net.module
                net.load_state_dict(state_dict[name])

        if self.opt.phase != 'test':
            print('loading the optim from %s' % load_path)
            for i, optim in enumerate(self.optimizers):
                optim.load_state_dict(state_dict['opt_%02d' % i])
                self.optimizer_to(optim, self.device)
            try:
                print('loading the sched from %s' % load_path)
                for i, sched in enumerate(self.schedulers):
                    sched.load_state_dict(state_dict['sched_%02d' % i])
            except:
                print('Failed to load schedulers, set schedulers according to epoch count manually')
                for i, sched in enumerate(self.schedulers):
                    sched.last_epoch = self.opt.epoch_count - 1

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requires_grad=False for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def generate_visuals_for_evaluation(self, data, mode):
        return {}


class BaseReconModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        # net structure and parameters
        parser.add_argument('--net_recon', type=str, default='resnet50',
                            choices=['resnet18', 'resnet34', 'resnet50', 'mobilenet_v2'],
                            help='network structure')
        parser.add_argument('--init_path', type=str, default='checkpoints/init_model/resnet50-0676ba61.pth')
        parser.add_argument('--use_last_fc', type=util.str2bool, nargs='?', const=True, default=False,
                            help='zero initialize the last fc')
        parser.add_argument('--bfm_folder', type=str, default='BFM')
        parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')

        # renderer parameters
        parser.add_argument('--focal', type=float, default=1015.)
        parser.add_argument('--center', type=float, default=112.)
        parser.add_argument('--camera_d', type=float, default=10.)
        parser.add_argument('--z_near', type=float, default=5.)
        parser.add_argument('--z_far', type=float, default=15.)
        parser.add_argument('--use_opengl', type=util.str2bool, nargs='?', const=True, default=True,
                            help='use opengl context or not')

        # training parameters
        parser.add_argument('--net_recog', type=str, default='r50', choices=['r18', 'r43', 'r50'],
                            help='face recog network structure')
        parser.add_argument('--net_recog_path', type=str,
                            default='checkpoints/recog_model/ms1mv3_arcface_r50_fp16/backbone.pth')
        parser.add_argument('--use_crop_face', type=util.str2bool, nargs='?', const=True, default=False,
                            help='use crop mask for photo loss')
        parser.add_argument('--use_predef_M', type=util.str2bool, nargs='?', const=True, default=False,
                            help='use predefined M for predicted face')

        if is_train:
            # augmentation parameters
            parser.add_argument('--shift_pixs', type=float, default=10., help='shift pixels')
            parser.add_argument('--scale_delta', type=float, default=0.1, help='delta scale factor')
            parser.add_argument('--rot_angle', type=float, default=10., help='rot angles, degree')

            # loss weights
            parser.add_argument('--w_feat', type=float, default=0.2, help='weight for feat loss')
            parser.add_argument('--w_color', type=float, default=1.92, help='weight for pixel-wise loss')
            parser.add_argument('--w_reg', type=float, default=3.0e-4, help='weight for reg loss')
            parser.add_argument('--w_id', type=float, default=1.0, help='weight for id_reg loss')
            parser.add_argument('--w_exp', type=float, default=0.8, help='weight for exp_reg loss')
            parser.add_argument('--w_tex', type=float, default=1.7e-2, help='weight for tex_reg loss')
            parser.add_argument('--w_gamma', type=float, default=10.0, help='weight for gamma loss')
            parser.add_argument('--w_lm', type=float, default=1.6e-3, help='weight for lm loss')
            parser.add_argument('--w_reflc', type=float, default=5.0, help='weight for reflc loss')

        opt, _ = parser.parse_known_args()
        parser.set_defaults(focal=1015., center=112., camera_d=10., use_last_fc=False, z_near=5., z_far=15.)
        if is_train:
            parser.set_defaults(use_crop_face=True, use_predef_M=False)

        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)

        fov = 2 * np.arctan(opt.center / opt.focal) * 180 / np.pi
        self.renderer = MeshRenderer(
            rasterize_fov=fov, znear=opt.z_near, zfar=opt.z_far, rasterize_size=int(2 * opt.center),
            use_opengl=opt.use_opengl
        )

        self.net_recog = networks.define_net_recog(net_recog=opt.net_recog, pretrained_path=opt.net_recog_path)

        self.visual_names = ['output_vis']
        self.model_names = ['net_recon']
        self.parallel_names = self.model_names + ['renderer', 'net_recog']

        self.compute_feat_loss = perceptual_loss
        self.comupte_color_loss = photo_loss
        if self.isTrain:
            self.loss_names = []
            self.loss_names.extend(['all', 'feat', 'color', 'lm', 'reg', 'gamma'])  # 'reflc'

            # loss func name: (compute_%s_loss) % loss_name
            self.compute_lm_loss = landmark_loss
            self.compute_reg_loss = reg_loss
            self.compute_reflc_loss = None
            # self.compute_reflc_loss = reflectance_loss

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.input_img = input['imgs'].to(self.device)
        self.atten_mask = input['msks'].to(self.device) if 'msks' in input else None
        self.gt_lm = input['lms'].to(self.device) if 'lms' in input else None
        self.trans_m = input['M'].to(self.device) if 'M' in input else None
        self.image_paths = input['im_paths'] if 'im_paths' in input else None

    def forward(self):
        output_coeff = self.net_recon(self.input_img)

        self.face_model.to(self.device)
        self.pred_vertex, self.pred_tex, self.pred_color, self.pred_lm = \
            self.face_model.compute_for_render(output_coeff)

        self.pred_mask, _, self.pred_face = self.renderer(
            self.pred_vertex, self.face_model.face_buf, feat=self.pred_color)

        self.pred_coeffs_dict = self.face_model.split_coeff(output_coeff)

    def get_recog_loss(self, input_img, pred_face, pred_trans_m, input_trans_m):
        pred_feat = self.net_recog(pred_face, pred_trans_m)
        gt_feat = self.net_recog(input_img, input_trans_m)

        return self.compute_feat_loss(pred_feat, gt_feat)

    def compute_losses(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""

        assert self.net_recog.training == False

        trans_m = self.trans_m
        if self.opt.use_predef_M:
            trans_m = estimate_norm_torch(self.pred_lm, self.input_img.shape[-2])
        self.loss_feat = self.opt.w_feat * self.get_recog_loss(self.input_img, self.pred_face, trans_m, self.trans_m)

        face_mask = self.pred_mask
        if self.opt.use_crop_face:
            face_mask, _, _ = self.renderer(self.pred_vertex, self.facemodel.front_face_buf)

        face_mask = face_mask.detach()
        self.loss_color = self.opt.w_color * self.comupte_color_loss(self.pred_face, self.input_img, self.atten_mask * face_mask)
        # self.loss_color = self.opt.w_color * self.comupte_color_loss(self.pred_face, self.input_img, face_mask)

        loss_reg, loss_gamma = self.compute_reg_loss(self.pred_coeffs_dict, self.opt)
        self.loss_reg = self.opt.w_reg * loss_reg
        self.loss_gamma = self.opt.w_gamma * loss_gamma

        self.loss_lm = self.opt.w_lm * self.compute_lm_loss(self.pred_lm, self.gt_lm)

        if self.compute_reflc_loss is not None:
            self.loss_reflc = self.opt.w_reflc * self.compute_reflc_loss(self.pred_tex, self.facemodel.skin_mask)
        else:
            self.loss_reflc = 0

        # self.loss_all = (self.loss_feat + self.loss_color + self.loss_reg + self.loss_gamma
        #                  + self.loss_lm + self.loss_reflc)
        self.loss_all = self.loss_feat + self.loss_color + self.loss_reg + self.loss_gamma + self.loss_lm
        if self.loss_reflc != 0:
            self.loss_all += self.loss_reflc

    def optimize_parameters(self, is_train=True):
        self.forward()
        self.compute_losses()

        """Update network weights; it will be called in every training iteration."""
        if is_train:
            for optimizer in self.optimizers:
                optimizer.zero_grad()

            self.loss_all.backward()

            for optimizer in self.optimizers:
                optimizer.step()

    def compute_visuals(self):
        device = self.device
        with torch.no_grad():
            input_img_numpy = 255. * self.input_img.detach().cpu().permute(0, 2, 3, 1).numpy()
            output_vis = self.pred_face * self.pred_mask + (1 - self.pred_mask) * self.input_img
            output_vis_numpy_raw = 255. * output_vis.detach().cpu().permute(0, 2, 3, 1).numpy()

            if self.gt_lm is not None:
                gt_lm_numpy = self.gt_lm.cpu().numpy()
                pred_lm_numpy = self.pred_lm.detach().cpu().numpy()
                output_vis_numpy = util.draw_landmarks(output_vis_numpy_raw, gt_lm_numpy, 'b')
                output_vis_numpy = util.draw_landmarks(output_vis_numpy, pred_lm_numpy, 'r')

                output_vis_numpy = np.concatenate((input_img_numpy, output_vis_numpy_raw, output_vis_numpy), axis=-2)
            else:
                output_vis_numpy = np.concatenate((input_img_numpy, output_vis_numpy_raw), axis=-2)

            self.output_vis = torch.tensor(output_vis_numpy / 255., dtype=torch.float32).permute(0, 3, 1, 2).to(device)

    def save_mesh(self, name, save_lms):
        recon_shape = self.pred_vertex  # get reconstructed shape
        recon_shape[..., -1] = 10 - recon_shape[..., -1]  # from camera space to world space
        recon_shape = recon_shape.cpu().numpy()[0]
        recon_color = self.pred_color
        recon_color = recon_color.cpu().numpy()[0]
        tri = self.face_model.face_buf.cpu().numpy()
        mesh = trimesh.Trimesh(vertices=recon_shape, faces=tri,
                               vertex_colors=np.clip(255. * recon_color, 0, 255).astype(np.uint8), process=False)
        # mesh = trimesh.Trimesh(vertices=recon_shape, faces=tri, process=False)
        mesh.export(name)

        if save_lms:
            lms = recon_shape[self.face_model.keypoints.cpu(), :]
            with open(name[:name.rfind(".")] + "_lms.txt", 'w') as f:
                for i in range(lms.shape[0]):
                    lm = lms[i]
                    f.write("{:.5f} {:.5f} {:.5f}\n".format(lm[0], lm[1], lm[2]))

    def save_coeff(self, name):
        pred_coeffs = {key: self.pred_coeffs_dict[key].cpu().numpy() for key in self.pred_coeffs_dict}
        pred_lm = self.pred_lm.cpu().numpy()
        # transfer to image coordinate
        pred_lm = np.stack([pred_lm[:, :, 0], self.input_img.shape[2] - 1 - pred_lm[:, :, 1]], axis=2)
        pred_coeffs['lm68'] = pred_lm
        savemat(name, pred_coeffs)
