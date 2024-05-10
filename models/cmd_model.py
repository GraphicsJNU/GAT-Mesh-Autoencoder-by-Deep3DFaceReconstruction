""" This script defines the proposal face reconstruction model """

import torch
from .base_model import BaseReconModel
from . import networks
from .face_model import ColouredMeshFaceModel
from .losses import perceptual_loss, photo_loss, reg_loss, reflectance_loss, landmark_loss

import os.path as osp
from psbody.mesh import Mesh
from models.gat.utils.utils import to_edge_index, to_sparse, get_graph_info
from models.gat.networks import Decoder, VariationalAutoEncoder


class CMDModel(BaseReconModel):
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

        self.model_names.append('decoder_layer_fc' if opt.shape_fc_train else 'decoder_layer')
        self.parallel_names.append('decoder_layer_fc' if opt.shape_fc_train else 'decoder_layer')

        id_dim, angle_dim, gamma_dim, xy_dim, z_dim = opt.ae_dim, 3, 27, 2, 1
        final_layers = [id_dim, angle_dim, gamma_dim, xy_dim, z_dim]
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

        feature_dim = 6
        lambda_max = 2.0  # 2.3
        filters = [feature_dim, 16, 16, 16, 32, 32]
        w_kl = opt.w_kl
        enc_conv, dec_conv = opt.enc_conv, opt.dec_conv
        enc_k, dec_k = opt.enc_k, opt.dec_k
        ae_pretrained_epoch = opt.ae_pretrained_epoch

        edge_index_list = [to_edge_index(adj) for adj in a]
        up_transform_list = [to_sparse(up_transform) for up_transform in u]

        self.decoder_layer = Decoder(filters, edge_index_list, up_transform_list, final_layers[0],
                                     dec_k, lambda_max=lambda_max, conv=dec_conv)
        self.decoder_layer_fc = self.decoder_layer.fc
        vae_file_name = 'variational_auto_encoder_{:d}_{}_{:d}_{}_{:d}_{:.0e}_f6_{:d}.pt' \
            .format(final_layers[0], enc_conv, enc_k, dec_conv, dec_k, w_kl, ae_pretrained_epoch)
        if ae_pretrained_epoch:
            decoder_path = osp.join(base_path, vae_file_name)
            if osp.exists(decoder_path):
                print("load decoder")
                vae_saved_data = torch.load(decoder_path)
                variational_auto_encoder = VariationalAutoEncoder(feature_dim, final_layers[0], a, d, u, enc_k, dec_k,
                                                                  lambda_max, enc_conv, dec_conv, 'cpu')
                variational_auto_encoder.load_state_dict(vae_saved_data['model_state_dict'])
                self.decoder_layer.load_state_dict(variational_auto_encoder.decoder.state_dict())
            else:
                raise Exception("Not decoder")
        self.decoder_layer.eval()

        self.face_model = ColouredMeshFaceModel(
            mean, std, self.decoder_layer,
            bfm_folder=opt.bfm_folder, camera_distance=opt.camera_d, focal=opt.focal, center=opt.center,
            is_train=self.isTrain, default_name=opt.bfm_model,
            coeff_dims=final_layers
        )

        if self.isTrain:
            if opt.shape_fc_train:
                self.optimizer = torch.optim.Adam(
                    [{'params': self.net_recon.parameters()},
                     {'params': self.decoder_layer_fc.parameters()}], lr=opt.lr)
            else:
                self.optimizer = torch.optim.Adam(
                    [{'params': self.net_recon.parameters()},
                     {'params': self.decoder_layer.parameters()}], lr=opt.lr)
            self.optimizers = [self.optimizer]
        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def train(self):
        super().train()

        '''
        self.decoder_layer.eval()
        '''
