"""
This script defines the face reconstruction model for Deep3DFaceRecon_pytorch
"""

import torch
from .base_model import BaseReconModel
from . import networks
from .face_model import ParametricFaceModel
from .losses import perceptual_loss, photo_loss, reg_loss, reflectance_loss, landmark_loss


class FaceReconModel(BaseReconModel):
    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """

        BaseReconModel.__init__(self, opt)  # call the initialization method of BaseModel

        self.net_recon = networks.define_net_recon(
            net_recon=opt.net_recon, use_last_fc=opt.use_last_fc, init_path=opt.init_path, fc_dim=257
        )

        self.face_model = ParametricFaceModel(
            bfm_folder=opt.bfm_folder, camera_distance=opt.camera_d, focal=opt.focal, center=opt.center,
            is_train=self.isTrain, default_name=opt.bfm_model
        )

        if self.isTrain:
            self.optimizer = torch.optim.Adam(self.net_recon.parameters(), lr=opt.lr)
            self.optimizers = [self.optimizer]
        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks
