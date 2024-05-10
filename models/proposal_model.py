""" This script defines the proposal face reconstruction model """

import torch
from .base_model import BaseReconModel
from . import networks
from .face_model import ProposalFaceModel
from .losses import perceptual_loss, photo_loss, reg_loss, reflectance_loss, landmark_loss

import os.path as osp
from psbody.mesh import Mesh
from models.gat.utils.utils import to_edge_index, to_sparse, get_graph_info
from models.gat.networks import Decoder, VariationalAutoEncoder

import time


class ProposalModel(BaseReconModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = BaseReconModel.modify_commandline_options(parser, is_train)

        # parser.set_defaults(bfm_model='facewarehouse_model_front.mat')
        parser.set_defaults(bfm_model='BFM_model_for_300W_3D.mat')

        if is_train:
            # loss weights
            parser.set_defaults(w_feat=0.2, w_color=1.92, w_reg=3.0e-4,
                                w_id=1, w_exp=0.8, w_tex=1.7e-3,
                                w_gamma=10.0, w_lm=1.6e-3, w_reflc=5.0)

        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """

        BaseReconModel.__init__(self, opt)  # call the initialization method of BaseModel

        self.model_names.append('shape_decoder_layer_fc' if opt.shape_fc_train else 'shape_decoder_layer')
        self.parallel_names.append('shape_decoder_layer_fc' if opt.shape_fc_train else 'shape_decoder_layer')
        self.model_names.append('texture_decoder_layer_fc' if opt.texture_fc_train else 'texture_decoder_layer')
        self.parallel_names.append('texture_decoder_layer_fc' if opt.texture_fc_train else 'texture_decoder_layer')

        id_dim, tex_dim, angle_dim, gamma_dim, xy_dim, z_dim = opt.ae_dim, opt.ae_dim, 3, 27, 2, 1
        final_layers = [id_dim, tex_dim, angle_dim, gamma_dim, xy_dim, z_dim]
        fc_dim = sum(final_layers)
        self.net_recon = networks.define_net_recon(
            net_recon=opt.net_recon, use_last_fc=opt.use_last_fc, init_path=opt.init_path, fc_dim=fc_dim,
            final_layers=final_layers
        )

        base_path = './models/gat/datas'
        mesh = Mesh(filename=osp.join(base_path, 'mu.obj'))
        m, a, d, u, f = get_graph_info(mesh, './models/gat/datas')
        mean = torch.load(osp.join(base_path, 'mean_f6.pt'))
        std = torch.load(osp.join(base_path, 'standard_variation_f6.pt'))

        feature_dim = 3
        lambda_max = 2.0  # 2.3
        filters = [feature_dim, 16, 16, 16, 32, 32]
        w_kl = opt.w_kl
        enc_conv, dec_conv = opt.enc_conv, opt.dec_conv
        enc_k, dec_k = opt.enc_k, opt.dec_k
        ae_pretrained_epoch = opt.ae_pretrained_epoch

        edge_index_list = [to_edge_index(adj) for adj in a]
        up_transform_list = [to_sparse(up_transform) for up_transform in u]

        self.shape_decoder_layer = Decoder(filters, edge_index_list, up_transform_list, final_layers[0],
                                           dec_k, lambda_max=lambda_max, conv=dec_conv)
        self.shape_decoder_layer_fc = self.shape_decoder_layer.fc
        shape_autoencoder_file_name = 'variational_auto_encoder_{:d}_{}_{:d}_{}_{:d}_{:.0e}_f3_shape_{:d}.pt' \
            .format(final_layers[0], enc_conv, enc_k, dec_conv, dec_k, w_kl, ae_pretrained_epoch)
        if ae_pretrained_epoch:
            shape_decoder_path = osp.join(base_path, shape_autoencoder_file_name)
            if osp.exists(shape_decoder_path):
                print("load shape decoder")
                shape_saved_data = torch.load(shape_decoder_path)
                shape_auto_encoder = VariationalAutoEncoder(feature_dim, final_layers[0], a, d, u, enc_k, dec_k,
                                                            lambda_max, enc_conv, dec_conv, 'cpu')
                shape_auto_encoder.load_state_dict(shape_saved_data['model_state_dict'])
                self.shape_decoder_layer.load_state_dict(shape_auto_encoder.decoder.state_dict())
            else:
                raise Exception("Not shape decoder")
        self.shape_decoder_layer.eval()

        self.texture_decoder_layer = Decoder(filters, edge_index_list, up_transform_list, final_layers[1],
                                             dec_k, lambda_max=lambda_max, conv=dec_conv)
        self.texture_decoder_layer_fc = self.texture_decoder_layer.fc
        texture_autoencoder_file_name = 'variational_auto_encoder_{:d}_{}_{:d}_{}_{:d}_{:.0e}_f3_texture_{:d}.pt' \
            .format(final_layers[1], enc_conv, enc_k, dec_conv, dec_k, w_kl, ae_pretrained_epoch)
        if ae_pretrained_epoch:
            texture_decoder_path = osp.join(base_path, texture_autoencoder_file_name)
            if osp.exists(texture_decoder_path):
                print("load texture decoder")
                texture_saved_data = torch.load(texture_decoder_path)
                texture_auto_encoder = VariationalAutoEncoder(feature_dim, final_layers[1], a, d, u, enc_k, dec_k,
                                                              lambda_max, enc_conv, dec_conv, 'cpu')
                texture_auto_encoder.load_state_dict(texture_saved_data['model_state_dict'])
                self.texture_decoder_layer.load_state_dict(texture_auto_encoder.decoder.state_dict())
            else:
                raise Exception("Not texture decoder")
        self.texture_decoder_layer.eval()

        self.face_model = ProposalFaceModel(
            mean, std, self.shape_decoder_layer, self.texture_decoder_layer,
            bfm_folder=opt.bfm_folder, camera_distance=opt.camera_d, focal=opt.focal, center=opt.center,
            is_train=self.isTrain, default_name=opt.bfm_model,
            coeff_dims=final_layers
        )

        if self.isTrain:
            params = [{'params': self.net_recon.parameters()},
                      {'params': self.shape_decoder_layer_fc.parameters() if opt.shape_fc_train else self.shape_decoder_layer.parameters()},
                      {'params': self.texture_decoder_layer_fc.parameters() if opt.texture_fc_train else self.texture_decoder_layer.parameters()}]
            optimizer = torch.optim.Adam(params, lr=opt.lr)
            self.optimizers = [optimizer]
        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

