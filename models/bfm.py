"""This script defines the parametric 3d face model for Deep3DFaceRecon_pytorch
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from util.load_mats import transferBFM09
import os
import util.mesh as mesh


def perspective_projection(focal, center):
    # return p.T (N, 3) @ (3, 3) 
    return np.array([
        focal, 0, center,
        0, focal, center,
        0, 0, 1
    ]).reshape([3, 3]).astype(np.float32).transpose()


class SH:
    def __init__(self):
        self.a = [np.pi, 2 * np.pi / np.sqrt(3.), 2 * np.pi / np.sqrt(8.)]
        self.c = [1 / np.sqrt(4 * np.pi), np.sqrt(3.) / np.sqrt(4 * np.pi), 3 * np.sqrt(5.) / np.sqrt(12 * np.pi)]


class BaseFaceModel:
    def __init__(self,
                 bfm_folder='./BFM',
                 camera_distance=10.,
                 init_lit=np.array([
                     0.8, 0, 0, 0, 0, 0, 0, 0, 0
                 ]),
                 focal=1015.,
                 center=112.,
                 is_train=True,
                 default_name='BFM_model_front.mat'):
        if not os.path.isfile(os.path.join(bfm_folder, default_name)):
            transferBFM09(bfm_folder)
        model = loadmat(os.path.join(bfm_folder, default_name))

        # face indices for each vertex that lies in. starts from 0. [N,8]
        self.point_buf = model['point_buf'].astype(np.int64) - 1
        # vertex indices for each face. starts from 0. [F,3]
        self.face_buf = model['tri'].astype(np.int64) - 1
        # vertex indices for 68 landmarks. starts from 0. [68,1]
        self.keypoints = np.squeeze(model['keypoints']).astype(np.int64) - 1

        if is_train:
            # vertex indices for small face region to compute photometric error. starts from 0.
            if 'frontmask2_idx' in model.keys():
                self.front_mask = np.squeeze(model['frontmask2_idx']).astype(np.int64) - 1
            # vertex indices for each face from small face region. starts from 0. [f,3]
            self.front_face_buf = model['tri_mask2'].astype(np.int64) - 1
            # vertex indices for pre-defined skin region to compute reflectance loss
            self.skin_mask = np.squeeze(model['skinmask'])

        self.persc_proj = perspective_projection(focal, center)
        self.device = 'cpu'
        self.camera_distance = camera_distance
        self.SH = SH()
        self.init_lit = init_lit.reshape([1, 1, -1]).astype(np.float32)

    def to(self, device):
        self.device = device
        for key, value in self.__dict__.items():
            if type(value).__module__ == np.__name__:
                setattr(self, key, torch.tensor(value).to(device))

    def compute_norm(self, face_shape):
        """
        Return:
            vertex_norm      -- torch.tensor, size (B, N, 3)

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
        """

        v1 = face_shape[:, self.face_buf[:, 0]]
        v2 = face_shape[:, self.face_buf[:, 1]]
        v3 = face_shape[:, self.face_buf[:, 2]]
        e1 = v1 - v2
        e2 = v2 - v3
        face_norm = torch.cross(e1, e2, dim=-1)
        face_norm = F.normalize(face_norm, dim=-1, p=2)
        face_norm = torch.cat([face_norm, torch.zeros(face_norm.shape[0], 1, 3).to(self.device)], dim=1)

        vertex_norm = torch.sum(face_norm[:, self.point_buf], dim=2)
        vertex_norm = F.normalize(vertex_norm, dim=-1, p=2)

        return vertex_norm

    def compute_color(self, face_texture, face_norm, gamma):
        """
        Return:
            face_color       -- torch.tensor, size (B, N, 3), range (0, 1.)

        Parameters:
            face_texture     -- torch.tensor, size (B, N, 3), from texture model, range (0, 1.)
            face_norm        -- torch.tensor, size (B, N, 3), rotated face normal
            gamma            -- torch.tensor, size (B, 27), SH coeffs
        """
        batch_size = gamma.shape[0]
        v_num = face_texture.shape[1]
        a, c = self.SH.a, self.SH.c
        gamma = gamma.reshape([batch_size, 3, 9])
        gamma = gamma + self.init_lit
        gamma = gamma.permute(0, 2, 1)
        Y = torch.cat([
            a[0] * c[0] * torch.ones_like(face_norm[..., :1]).to(self.device),
            -a[1] * c[1] * face_norm[..., 1:2],
            a[1] * c[1] * face_norm[..., 2:],
            -a[1] * c[1] * face_norm[..., :1],
            a[2] * c[2] * face_norm[..., :1] * face_norm[..., 1:2],
            -a[2] * c[2] * face_norm[..., 1:2] * face_norm[..., 2:],
            0.5 * a[2] * c[2] / np.sqrt(3.) * (3 * face_norm[..., 2:] ** 2 - 1),
            -a[2] * c[2] * face_norm[..., :1] * face_norm[..., 2:],
            0.5 * a[2] * c[2] * (face_norm[..., :1] ** 2 - face_norm[..., 1:2] ** 2)
        ], dim=-1)
        r = Y @ gamma[..., :1]
        g = Y @ gamma[..., 1:2]
        b = Y @ gamma[..., 2:]
        face_color = torch.cat([r, g, b], dim=-1) * face_texture

        return face_color

    def compute_rotation(self, angles):
        """
        Return:
            rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat

        Parameters:
            angles           -- torch.tensor, size (B, 3), radian
        """

        batch_size = angles.shape[0]
        ones = torch.ones([batch_size, 1]).to(self.device)
        zeros = torch.zeros([batch_size, 1]).to(self.device)
        x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],

        rot_x = torch.cat([
            ones, zeros, zeros,
            zeros, torch.cos(x), -torch.sin(x),
            zeros, torch.sin(x), torch.cos(x)
        ], dim=1).reshape([batch_size, 3, 3])

        rot_y = torch.cat([
            torch.cos(y), zeros, torch.sin(y),
            zeros, ones, zeros,
            -torch.sin(y), zeros, torch.cos(y)
        ], dim=1).reshape([batch_size, 3, 3])

        rot_z = torch.cat([
            torch.cos(z), -torch.sin(z), zeros,
            torch.sin(z), torch.cos(z), zeros,
            zeros, zeros, ones
        ], dim=1).reshape([batch_size, 3, 3])

        rot = rot_z @ rot_y @ rot_x

        return rot.permute(0, 2, 1)

    def to_camera(self, face_shape):
        face_shape[..., -1] = self.camera_distance - face_shape[..., -1]

        return face_shape

    def to_image(self, face_shape):
        """
        Return:
            face_proj        -- torch.tensor, size (B, N, 2), y direction is opposite to v direction

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
        """
        # to image_plane
        face_proj = face_shape @ self.persc_proj
        face_proj = face_proj[..., :2] / face_proj[..., 2:]

        return face_proj

    def transform(self, face_shape, rot, trans):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3) pts @ rot + trans

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
            rot              -- torch.tensor, size (B, 3, 3)
            trans            -- torch.tensor, size (B, 3)
        """
        return face_shape @ rot + trans.unsqueeze(1)

    def get_landmarks(self, face_proj):
        """
        Return:
            face_lms         -- torch.tensor, size (B, 68, 2)

        Parameters:
            face_proj       -- torch.tensor, size (B, N, 2)
        """
        return face_proj[:, self.keypoints]


class ParametricFaceModel(BaseFaceModel):
    def __init__(self,
                 bfm_folder='./BFM',
                 recenter=True,
                 camera_distance=10.,
                 init_lit=np.array([
                     0.8, 0, 0, 0, 0, 0, 0, 0, 0
                 ]),
                 focal=1015.,
                 center=112.,
                 is_train=True,
                 default_name='BFM_model_front.mat'):

        BaseFaceModel.__init__(self,
                               bfm_folder,
                               camera_distance,
                               init_lit,
                               focal,
                               center,
                               is_train,
                               default_name)

        model = loadmat(os.path.join(bfm_folder, default_name))
        # mean face shape. [3 * N,1]
        self.mean_shape = model['meanshape'].astype(np.float32)
        # identity basis. [3 * N, 80]
        self.id_base = model['idBase'].astype(np.float32)
        # expression basis. [3 * N, 64]
        self.exp_base = model['exBase'].astype(np.float32)
        # mean face texture. [3 * N, 1] (0 - 255)
        self.mean_tex = model['meantex'].astype(np.float32)
        # texture basis. [3 * N, 80]
        self.tex_base = model['texBase'].astype(np.float32)

        if recenter:
            mean_shape = self.mean_shape.reshape([-1, 3])
            mean_shape = mean_shape - np.mean(mean_shape, axis=0, keepdims=True)
            self.mean_shape = mean_shape.reshape([-1, 1])

    def compute_shape(self, id_coeff, exp_coeff):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3)

        Parameters:
            id_coeff         -- torch.tensor, size (B, 80), identity coeffs
            exp_coeff        -- torch.tensor, size (B, 64), expression coeffs
        """
        batch_size = id_coeff.shape[0]
        id_part = torch.einsum('ij,aj->ai', self.id_base, id_coeff)
        exp_part = torch.einsum('ij,aj->ai', self.exp_base, exp_coeff)
        face_shape = id_part + exp_part + self.mean_shape.reshape([1, -1])

        return face_shape.reshape([batch_size, -1, 3])

    def compute_texture(self, tex_coeff, normalize=True):
        """
        Return:
            face_texture     -- torch.tensor, size (B, N, 3), in RGB order, range (0, 1.)

        Parameters:
            tex_coeff        -- torch.tensor, size (B, 80)
        """
        batch_size = tex_coeff.shape[0]
        face_texture = torch.einsum('ij,aj->ai', self.tex_base, tex_coeff) + self.mean_tex
        if normalize:
            face_texture = face_texture / 255.

        return face_texture.reshape([batch_size, -1, 3])

    def split_coeff(self, coeffs):
        """
        Return:
            coeffs_dict     -- a dict of torch.tensors

        Parameters:
            coeffs          -- torch.tensor, size (B, 256)
        """
        id_coeffs = coeffs[:, :80]
        exp_coeffs = coeffs[:, 80: 144]
        tex_coeffs = coeffs[:, 144: 224]
        angles = coeffs[:, 224: 227]
        gammas = coeffs[:, 227: 254]
        translations = coeffs[:, 254:]
        return {
            'id': id_coeffs,
            'exp': exp_coeffs,
            'tex': tex_coeffs,
            'angle': angles,
            'gamma': gammas,
            'trans': translations
        }

    def compute_for_render(self, coeffs):
        """
        Return:
            face_vertex     -- torch.tensor, size (B, N, 3), in camera coordinate
            face_color      -- torch.tensor, size (B, N, 3), in RGB order
            landmark        -- torch.tensor, size (B, 68, 2), y direction is opposite to v direction
        Parameters:
            coeffs          -- torch.tensor, size (B, 257)
        """
        coef_dict = self.split_coeff(coeffs)
        face_shape = self.compute_shape(coef_dict['id'], coef_dict['exp'])
        rotation = self.compute_rotation(coef_dict['angle'])

        face_shape_transformed = self.transform(face_shape, rotation, coef_dict['trans'])
        face_vertex = self.to_camera(face_shape_transformed)

        face_proj = self.to_image(face_vertex)
        landmark = self.get_landmarks(face_proj)

        face_texture = self.compute_texture(coef_dict['tex'])
        face_norm = self.compute_norm(face_shape)
        face_norm_roted = face_norm @ rotation
        face_color = self.compute_color(face_texture, face_norm_roted, coef_dict['gamma'])

        return face_vertex, face_texture, face_color, landmark


class ProposalFaceModel(BaseFaceModel):
    def __init__(self,
                 bfm_folder='./BFM',
                 recenter=True,
                 camera_distance=10.,
                 init_lit=np.array([
                     0.8, 0, 0, 0, 0, 0, 0, 0, 0
                 ]),
                 focal=1015.,
                 center=112.,
                 is_train=True,
                 coeff_dims=None,  # id_dim, tex_dim, angle_dim, gamma_dim, xy_dim, z_dim
                 default_name='BFM_model_for_300W_3D.mat'):

        BaseFaceModel.__init__(self,
                               bfm_folder,
                               camera_distance,
                               init_lit,
                               focal,
                               center,
                               is_train,
                               default_name)

        import os.path as osp
        from psbody.mesh import Mesh
        from models.gat.utils.utils import to_edge_index, to_sparse, get_graph_info
        from models.gat.networks import DecoderLayer, AutoEncoder, VariationalAutoEncoder, RefineAutoEncoder

        self.is_train = is_train

        base_path = './models/gat/datas'
        mesh = Mesh(filename=osp.join(base_path, 'mu.obj'))
        m, a, d, u, f = get_graph_info(mesh, './models/gat/datas')

        if coeff_dims is None:
            coeff_dims = [32, 80, 3, 27, 3]

        self.coeff_dims = coeff_dims

        feature_dim = 3
        lambda_max = 2.0  # 2.3
        self.filters = [feature_dim, 16, 16, 16, 32, 32]

        edge_index_list = [to_edge_index(adj) for adj in a]
        up_transform_list = [to_sparse(up_transform) for up_transform in u]

        k = 6
        w_kl = 1e-5
        enc_conv, dec_conv = 'gat', 'cheb'
        enc_k = k
        dec_k = k
        enc_k = k if enc_conv == 'cheb' else 1
        dec_k = k if dec_conv == 'cheb' else 1
        self.decoder_layer = DecoderLayer(self.filters, edge_index_list, up_transform_list, coeff_dims[0], dec_k,
                                          lambda_max=lambda_max, conv='cheb')

        self.mean = torch.load(osp.join(base_path, 'mean.pt'))
        self.std = torch.load(osp.join(base_path, 'standard_variation.pt'))

        variational_autoencoder_file_name = 'variational_auto_encoder_{:d}_{:d}_{:d}_{:.0e}.pt' \
            .format(coeff_dims[0], enc_k, dec_k, w_kl)
        saved_data = torch.load(osp.join(base_path, variational_autoencoder_file_name))
        # saved_data = torch.load(osp.join(base_path, 'auto_encoder.pt'))
        # saved_data = torch.load(osp.join(base_path, 'auto_encoder_100000.pt'))

        auto_encoder = VariationalAutoEncoder(feature_dim, coeff_dims[0], a, d, u, enc_k, dec_k,
                                              lambda_max, enc_conv, dec_conv, 'cpu')
        # auto_encoder = AutoEncoder(feature_dim, coeff_dims[0], a, d, u, k, lambda_max, 'gat', 'cheb', 'cpu')
        auto_encoder.load_state_dict(saved_data['model_state_dict'])

        self.decoder_layer.load_state_dict(auto_encoder.decoder.state_dict())

        self.decoder_layer.eval()

        # --------------------------------------------------------------------------------------------------------------

        model = loadmat(os.path.join(bfm_folder, default_name))
        # mean face texture. [3 * N, 1] (0 - 255)
        self.mean_tex = model['meantex'].astype(np.float32)
        # self.mean_tex = model['meantexImg'].astype(np.float32
        # texture basis. [3 * N, 80]
        self.tex_base = model['texBase'].astype(np.float32)[:, :coeff_dims[1]]
        # self.tex_base = model['texBaseImg'].astype(np.float32)[:, :coeff_dims[1]]

        # self.size = 224
        # self.uvcoords = model['uvcoords'].astype(np.float32)[:, :2]
    def to(self, device):
        super().to(device)

        self.decoder_layer = self.decoder_layer.to(device)

        for i in range(len(self.decoder_layer.edge_index_list)):
            self.decoder_layer.edge_index_list[i] = self.decoder_layer.edge_index_list[i].to(device)

        for i in range(len(self.decoder_layer.up_transform_list)):
            self.decoder_layer.up_transform_list[i] = self.decoder_layer.up_transform_list[i].to(device)

        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

        self.mean_tex = self.mean_tex.to(device)
        self.tex_base = self.tex_base.to(device)

        self.point_buf = self.point_buf.to(device)
        self.face_buf = self.face_buf.to(device)
        self.keypoints = self.keypoints.to(device)

        if self.is_train:
            self.front_mask = self.front_mask.to(device)
            self.front_face_buf = self.front_face_buf.to(device)
            self.skin_mask = self.skin_mask.to(device)

    def compute_shape(self, id_coeff):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3)

        Parameters:
            id_coeff         -- torch.tensor, size (B, 256), identity coeffs
        """

        batch_size = id_coeff.shape[0]
        feature_part = self.decoder_layer(id_coeff)
        face_shape = (feature_part * self.std.repeat(batch_size, 1, 1)) + self.mean.unsqueeze(0)

        return face_shape

    def compute_texture(self, tex_coeff, normalize=True):
        """
        Return:
            face_texture     -- torch.tensor, size (B, N, 3), in RGB order, range (0, 1.)

        Parameters:
            tex_coeff        -- torch.tensor, size (B, 80)
        """
        batch_size = tex_coeff.shape[0]
        face_texture = torch.einsum('ij,aj->ai', self.tex_base, tex_coeff) + self.mean_tex
        if normalize:
            face_texture = face_texture / 255.

        return face_texture.reshape([batch_size, -1, 3])

    def split_coeff(self, coeffs):
        """
        Return:
            coeffs_dict     -- a dict of torch.tensors

        Parameters:
            coeffs          -- torch.tensor, size (B, 369)
        """
        s, e = 0, self.coeff_dims[0]
        id_coeffs = coeffs[:, s:e]  # coeffs[:, :256]  # coeffs[:, :80]
        s, e = e, e + self.coeff_dims[1]
        tex_coeffs = coeffs[:, s:e]  # coeffs[:, 256:336]  # coeffs[:, 144:224]
        s, e = e, e + self.coeff_dims[2]
        angles = coeffs[:, s:e]  # coeffs[:, 336:339]  # coeffs[:, 224:227]
        s, e = e, e + self.coeff_dims[3]
        gammas = coeffs[:, s:e]  # coeffs[:, 339:366]  # coeffs[:, 227:254]
        s, e = e, e + self.coeff_dims[4]
        translations = coeffs[:, s:]  # coeffs[:, 366:]  # coeffs[:, 254:]
        return {
            'id': id_coeffs,
            'tex': tex_coeffs,
            'angle': angles,
            'gamma': gammas,
            'trans': translations
        }

    # https://github.com/facebookresearch/pytorch3d/issues/736
    def compute_for_render(self, coeffs):
        """
        Return:
            face_vertex     -- torch.tensor, size (B, N, 3), in camera coordinate
            face_color      -- torch.tensor, size (B, N, 3), in RGB order
            landmark        -- torch.tensor, size (B, 68, 2), y direction is opposite to v direction
        Parameters:
            coeffs          -- torch.tensor, size (B, 257)
        """
        batch_size = coeffs.shape[0]

        coef_dict = self.split_coeff(coeffs)
        face_shape = self.compute_shape(coef_dict['id'])

        rotation = self.compute_rotation(coef_dict['angle'])
        face_shape_transformed = self.transform(face_shape, rotation, coef_dict['trans'])

        face_vertex = self.to_camera(face_shape_transformed)

        face_proj = self.to_image(face_vertex)
        landmark = self.get_landmarks(face_proj)

        face_texture = self.compute_texture(coef_dict['tex'])

        face_norm = self.compute_norm(face_shape)
        face_norm_roted = face_norm @ rotation
        face_color = self.compute_color(face_texture, face_norm_roted, coef_dict['gamma'])

        return face_vertex, face_texture, face_color, landmark

    def unwrap(self, colors):
        bs = colors.shape[0]
        uvcoords = self.uvcoords.detach().cpu().numpy()
        face_buf = self.face_buf.detach().cpu().numpy()
        colors = colors.detach().cpu().numpy()

        ret = torch.zeros((bs, self.size, self.size, 3), dtype=torch.float32)
        for i in range(bs):
            print(i)
            ret[i, :, :, :] = torch.from_numpy(
                mesh.render.render_colors(uvcoords, face_buf, colors[i], self.size, self.size, c=3))

        return ret
