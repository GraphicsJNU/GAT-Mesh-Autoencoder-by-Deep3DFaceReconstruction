import torch
from torch import Tensor, index_select
from torch.nn import Module, ModuleList, Linear, ELU, init, BatchNorm1d, Conv2d, Sequential, ReLU, Flatten, AvgPool2d
import torch.nn.functional as F
from models.gat.conv import ChebConv, GATConv
import torchvision.models as models
import torchvision.models._utils as _utils

from scipy.io import loadmat

import numpy as np
from torch_scatter import scatter_add

from utils import utils


def pool(x, transform, dim=1):
    row, col = transform._indices()
    value = transform._values().unsqueeze(-1)
    x = index_select(x, dim, col) * value
    x = scatter_add(x, row, dim, dim_size=transform.size(0))

    return x


class EncoderLayer(Module):
    def __init__(self, net: str, latent_dim: int, img_size: int = 224):
        super(EncoderLayer, self).__init__()

        names = ['avgpool']
        # layers = ModuleList()
        if net == 'mobilenet':
            backbone = models.mobilenet_v3_small()
            # mobilenet v3 small -> ['features', 'avgpool', 'classifier']
            coeff_size = 576
        elif 'efficientnet' in net:
            # efficientnet b0 ~ b7 -> ['features', 'avgpool', 'classifier']
            # efficientnet v2 s, m, l -> ['features', 'avgpool', 'classifier']
            if net == 'efficientnet_b0':
                backbone = models.efficientnet_b0()
                coeff_size = 1280
            elif net == 'efficientnet_b1':
                backbone = models.efficientnet_b1()
                coeff_size = 1280
            elif net == 'efficientnet_b2':
                backbone = models.efficientnet_b2()
                coeff_size = 1408
            elif net == 'efficientnet_b3':
                backbone = models.efficientnet_b3()
                coeff_size = 1536
            elif net == 'efficientnet_b4':
                backbone = models.efficientnet_b4()
                coeff_size = 1792
            elif net == 'efficientnet_b5':
                backbone = models.efficientnet_b5()
                coeff_size = 2048
            elif net == 'efficientnet_b6':
                backbone = models.efficientnet_b6()
                coeff_size = 2304
            elif net == 'efficientnet_b7':
                backbone = models.efficientnet_b7()
                coeff_size = 2560
            else:
                raise RuntimeError('Not Supported Network.')
        elif net == 'resnet50':
            from torchvision.models import ResNet50_Weights

            backbone = models.resnet50(weights=None)  # ResNet50_Weights.IMAGENET1K_V1
            # ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']
            coeff_size = 2048

            # layer_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            # # layer_names = ['relu', 'layer1']
            # names = layer_names + names
            # layer_size = len(names)
            # conv_in_feat_sizes = [64, 256, 512, 1024, 2048]
            # conv_out_feat_size = 16
            # out_img_sizes = [int(img_size / (2 ** (i + 1))) for i in range(len(layer_names))]
            # out_feat_size = 256
            # for i in range(len(layer_names)):
            #     layers.append(Sequential(
            #         Conv2d(conv_in_feat_sizes[i], conv_out_feat_size, 1, bias=False),
            #         ReLU(),
            #         Flatten(),
            #         Linear(conv_out_feat_size * out_img_sizes[i] * out_img_sizes[i], out_feat_size, bias=False),
            #         ReLU()
            #     ))
            #
            # layers.append(Sequential(Flatten(), Linear(coeff_size, out_feat_size, bias=False), ReLU()))
            # layers.append(Linear(out_feat_size * layer_size, latent_dim))
        elif net == 'resnet152':
            backbone = models.resnet152()
            # ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']
            coeff_size = 2048
        else:
            raise RuntimeError('Not Supported Network.')

        # self.layer_size = layer_size
        # self.out_feat_size = out_feat_size * layer_size
        # self.layer_names = names
        # self.layers = layers
        self.body = _utils.IntermediateLayerGetter(backbone, dict((name, name) for name in names))

        self.encoder = Sequential(
            Flatten(),
            Linear(coeff_size, 1024),
            ReLU(),
            Linear(1024, latent_dim)
        )

        self.reset_parameters()

    @staticmethod
    def reset_parameter(module: Module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                init.constant_(param, 0)
            else:
                init.xavier_uniform_(param)

    def reset_parameters(self):
        # self.reset_parameter(self.regression_layers)
        # self.reset_parameter(self.latent_estimate)

        # for layer in self.layers:
        #     self.reset_parameter(layer)

        self.reset_parameter(self.encoder)

    def forward(self, images):
        ret = self.body(images)
        # for i in range(self.layer_size):
        #     ret["feat_{}".format(self.layer_names[i])] = self.layers[i](ret[self.layer_names[i]])
        # feature = torch.cat([ret["feat_{}".format(self.layer_names[i])] for i in range(self.layer_size)], dim=1)
        # ret["feature"] = feature
        # ret["encoded_vec"] = self.layers[self.layer_size](feature)

        return self.encoder(ret["avgpool"])


class Encoder(Module):
    def __init__(self, filters: list, edge_index_list: list, down_transform_list: list,
                 latent_channels: int = 8, k: int = 6, lambda_max: float = 2.0, conv: str = 'cheb'):
        super(Encoder, self).__init__()

        self.filters = filters
        self.edge_index_list = edge_index_list
        self.down_transform_list = down_transform_list
        self.lambda_max = lambda_max
        self.gcn_cnt = len(filters) - 2
        self.conv_list = ModuleList()
        self.batchnorm_list = ModuleList()
        for i in range(self.gcn_cnt):
            s, e = i, i + 1
            if conv == 'cheb':
                self.conv_list.append(ChebConv(filters[s], filters[e], K=k))
            elif conv == 'gat':
                self.conv_list.append(GATConv(filters[s], filters[e], heads=k))
            else:
                raise RuntimeError('You can use only cheb or gat or arma.')

            self.batchnorm_list.append(BatchNorm1d(down_transform_list[i].shape[1]))

        self.activation = ELU()
        self.fc = Linear(down_transform_list[-1].size(0) * filters[-1], latent_channels)

        self.reset_parameters()

    def forward(self, x):
        for i in range(len(self.conv_list)):
            x = self.conv_list[i](x, self.edge_index_list[i], lambda_max=self.lambda_max)
            x = self.batchnorm_list[i](x)
            x = self.activation(x)
            x = pool(x, self.down_transform_list[i])

        x = x.view(x.shape[0], self.down_transform_list[-1].size(0) * self.filters[-1])
        x = self.fc(x)

        return x

    def reset_parameters(self):
        for conv in self.conv_list:
            for name, param in conv.named_parameters():
                if 'bias' in name:
                    init.constant_(param, 0)
                else:
                    init.xavier_uniform_(param)

        for name, param in self.fc.named_parameters():
            if 'bias' in name:
                init.constant_(param, 0)
            else:
                init.xavier_uniform_(param)


class DecoderLayer(Module):
    def __init__(self, filters: list, edge_index_list: list, up_transform_list: list,
                 latent_channels: int = 8, k: int = 6, lambda_max: float = 2.0, conv: str = 'cheb'):
        super(DecoderLayer, self).__init__()

        self.filters = filters
        self.edge_index_list = edge_index_list
        self.up_transform_list = up_transform_list
        self.lambda_max = lambda_max
        self.gcn_cnt = len(filters) - 2
        self.fc = Linear(latent_channels, up_transform_list[-1].size(1) * filters[-1])
        self.conv_list = ModuleList()
        self.batchnorm_list = ModuleList()
        for i in range(self.gcn_cnt):
            s, e = -i - 1, -i - 2
            if conv == 'cheb':
                self.conv_list.append(ChebConv(filters[s], filters[e], K=k))
                self.recon_conv = ChebConv(filters[1], filters[0], k)
            elif conv == 'gat':
                self.conv_list.append(GATConv(filters[s], filters[e], heads=k))
                self.recon_conv = GATConv(filters[1], filters[0], heads=k)
            else:
                raise RuntimeError('You can use only cheb or gat or arma.')

            self.batchnorm_list.append(BatchNorm1d(up_transform_list[self.gcn_cnt - i - 1].shape[0]))

        self.activation = ELU()

        self.reset_parameters()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], self.up_transform_list[-1].size(1), self.filters[-1])
        for i in range(len(self.conv_list)):
            x = pool(x, self.up_transform_list[self.gcn_cnt - i - 1])
            x = self.conv_list[i](x, self.edge_index_list[self.gcn_cnt - i - 1], lambda_max=self.lambda_max)
            x = self.batchnorm_list[i](x)
            x = self.activation(x)
        x = self.recon_conv(x, self.edge_index_list[0])

        return x

    def reset_parameters(self):
        for name, param in self.fc.named_parameters():
            if 'bias' in name:
                init.constant_(param, 0)
            else:
                init.xavier_uniform_(param)

        for conv in self.conv_list:
            for name, param in conv.named_parameters():
                if 'bias' in name:
                    init.constant_(param, 0)
                else:
                    init.xavier_uniform_(param)

        for name, param in self.recon_conv.named_parameters():
            if 'bias' in name:
                init.constant_(param, 0)
            else:
                init.xavier_uniform_(param)


class AutoEncoder(Module):
    def __init__(self, feature_channels: int, latent_channels: int, a: list,
                 d: list, u: list, k: int = 6, lambda_max: float = 2.3,
                 enc_conv='cheb', dec_conv='cheb', device='cpu'):
        super(AutoEncoder, self).__init__()

        self.filters = [feature_channels, 16, 16, 16, 32, 32]

        edge_index_list = [utils.to_edge_index(adj).to(device) for adj in a]
        down_transform_list = [utils.to_sparse(down_transform).to(device) for down_transform in d]
        up_transform_list = [utils.to_sparse(up_transform).to(device) for up_transform in u]
        # if enc_conv == 'gat':
        #     self.encoder = GATEncoder(feature_channels, edge_index_list, down_transform_list,
        #                               latent_channels, k).to(device)
        # else:
        #     self.encoder = Encoder(self.filters, edge_index_list, down_transform_list, latent_channels, k,
        #                            lambda_max=lambda_max, conv=enc_conv).to(device)
        self.encoder = Encoder(self.filters, edge_index_list, down_transform_list, latent_channels, k,
                               lambda_max=lambda_max, conv=enc_conv).to(device)
        self.decoder = DecoderLayer(self.filters, edge_index_list, up_transform_list, latent_channels, k,
                                    lambda_max=lambda_max, conv=dec_conv).to(device)

    def forward(self, x, is_vec=False):
        if x.shape[-1] != self.filters[0]:
            raise Exception('(batch_size, vertex_cnt, 3)이 아닌 데이터는 처리할 수 없습니다.')

        encoded_vec = self.encoder(x)
        pred = self.decoder(encoded_vec)

        if is_vec:
            return pred, encoded_vec
        else:
            return pred


class Reconstructor(Module):
    def __init__(self, feature_dim: int, latent_dim: int, mesh_data, a: list, u: list, k: int = 6,
                 lambda_max: float = 2.3, net: str = "resnet50", dec_conv: str = 'cheb', **kwargs):
        super(Reconstructor, self).__init__(**kwargs)
        device = self.device
        dtype = self.dtype

        '''
        pipeline: rs.pipeline = rs.pipeline()
        config: rs.config = rs.config()

        pipeline_wrapper: rs.pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile: rs.pipeline_profile = config.resolve(pipeline_wrapper)
        device: rs.device = pipeline_profile.get_device()

        found_rgb: bool = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        config.enable_stream(rs.stream.depth, img_size[1], img_size[0], rs.format.z16, 30)
        config.enable_stream(rs.stream.color, img_size[1], img_size[0], rs.format.bgr8, 30)

        pipeline.start(config)

        # ['coeffs', 'fx', 'fy', 'height', 'model', 'ppx', 'ppy', 'width']
        profile: rs.pipeline_profile = pipeline.get_active_profile()
        image_profile: rs.video_stream_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        depth_profile: rs.video_stream_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        image_intrinsics: rs.intrinsics = image_profile.get_intrinsics()
        depth_intrinsics: rs.intrinsics = depth_profile.get_intrinsics()
        # intric = profile.as_video_stream_profile().get_intrinsics()
        # print(image_intrinsics, depth_intrinsics)
        # [ 640x480  p[320.123 240.509]  f[617.598 617.986]  Inverse Brown Conrady [0 0 0 0 0] ]
        # [ 640x480  p[312.027 245.902]  f[385.041 385.041]  Brown Conrady [0 0 0 0 0] ]
        # print(dir(image_intrinsics))
        # image_intrinsics.fx
        '''
        # --------------------------------------------------------------------------------------------------------------

        self.filters = [feature_dim, 16, 16, 16, 32, 32]
        edge_index_list = [utils.to_edge_index(adj).to(device) for adj in a]
        up_transform_list = [utils.to_sparse(up_transform).to(device) for up_transform in u]
        self.n_shape, self.n_tex, self.n_pose, self.n_cam, self.n_light = latent_dim, self.tex_dims, 6, 3, 27
        # latent_dim + tex_dim + (6) + (3) + (27)
        # pose -> 회전 각도 roll, pitch, yaw, 오브젝트의 x, y, z 좌표
        # cam -> orthographic camera의 scale과 translation
        # light -> gamma
        self.encoder_layer = EncoderLayer(net, self.n_shape + self.n_tex + self.n_pose + self.n_light).to(device)
        self.decoder_layer = DecoderLayer(self.filters, edge_index_list, up_transform_list, latent_dim, k,
                                          lambda_max=lambda_max, conv=dec_conv).to(device)

        # self.cnn_auto_encoder = CnnAutoEncoder(feature_dim, latent_dim, a, u, k, lambda_max, net, dec_conv,
        #                                        device)

        self.encoder_layer.eval()
        self.decoder_layer.eval()

        self.mean = mesh_data.mean.to(device)
        self.std = mesh_data.std.to(device)
        # --------------------------------------------------------------------------------------------------------------

    def rigid_transform(self, vs, rot, trans):
        vs_r = torch.matmul(vs, rot)
        vs_t = torch.add(vs_r, trans.view(-1, 1, 3))

        return vs_t

    def render(self, trans_vertices, vertices, tex_coeff, R):
        batch_size = tex_coeff.shape[0]
        face_texture = self.get_color(tex_coeff)
        face_norm = self.compute_norm(vertices, self.tri, self.point_buf)
        face_norm_r = face_norm.bmm(R)
        face_color = self.add_illumination(face_texture, face_norm_r, self.gamma.repeat(batch_size, 1))
        face_color_tv = TexturesVertex(face_color.type(torch.float32))

        mesh = Meshes(trans_vertices, self.tri.repeat(batch_size, 1, 1), face_color_tv)
        rendered_image = self.renderer(mesh.clone())
        rendered_image = torch.clamp(rendered_image, 0, 255)

        return rendered_image

    def split_param(self, param):
        start_idx, end_idx = 0, self.n_shape
        shape_code = param[:, start_idx:end_idx]
        start_idx, end_idx = end_idx, end_idx + self.n_tex
        tex_code = param[:, start_idx:end_idx]
        start_idx, end_idx = end_idx, end_idx + self.n_pose
        pose_param = param[:, start_idx:end_idx]
        # start_idx, end_idx = end_idx, end_idx + self.n_cam
        # cam_param = param[:, start_idx:end_idx]
        start_idx, end_idx = end_idx, end_idx + self.n_light
        light_param = param[:, start_idx:end_idx]

        # return shape_code, tex_code, pose_param, cam_param, light_param
        return shape_code, tex_code, pose_param, light_param

    def get_vertices(self, shape_code):
        pred = self.decoder_layer(shape_code)
        vertices = (pred * self.std.repeat(pred.shape[0], 1, 1)) + self.mean.repeat(pred.shape[0], 1, 1)

        return vertices

    def get_projection(self, vertices, fx, fy, px, py):
        # we choose the focal length and camera position empirically
        # project 3D face onto image plane
        # input: face_shape with shape [1,N,3]
        #          rotation with shape [1,3,3]
        #         translation with shape [1,3]
        # output: face_projection with shape [1,N,2]
        #           z_buffer with shape [1,N,1]

        cam_pos = 10
        p_matrix = torch.tensor([[fx, 0.0, px], [0.0, fy, py], [0.0, 0.0, 1.0]], dtype=self.dtype).reshape([1, 3, 3])\
            .to(self.device).expand(vertices.shape[0], 3, 3)

        vertices[:, :, 2] = cam_pos - vertices[:, :, 2]
        aug_projection = vertices.bmm(p_matrix.permute(0, 2, 1))
        face_projection = aug_projection[:, :, 0:2] / aug_projection[:, :, 2:]

        z_buffer = cam_pos - aug_projection[:, :, 2:]

        return face_projection, z_buffer

    def forward(self, images, is_render=False):
        ret = {}

        batch_size = images.shape[0]
        # --------------------------------------------------------------------------------------------------------------
        # encode
        result = self.encoder_layer(images)
        # shape_code, tex_code, pose_param, cam_param, light_param = self.split_param(result)
        shape_code, tex_code, pose_param, light_param = self.split_param(result)

        ret["images"] = images
        ret["encoded_vec"] = shape_code
        # --------------------------------------------------------------------------------------------------------------
        # decode
        face_model = self.get_vertices(shape_code)
        ret["face_model"] = face_model

        # rot_mat = utils.batch_rodrigues(pose_param)
        # ret["trans_vertices"] = torch.bmm(rot_mat, ret["vertices"])
        R = self.compute_rotation_matrix(pose_param[:, :3])
        T = pose_param[:, 3:].view(-1, 1, 3)
        trans_face_model = self.rigid_transform(face_model, R, T)
        ret["trans_face_model"] = trans_face_model

        fx, fy = 1015.0, 1015.0
        px, py = 112.0, 112.0
        ret["proj_lms"], _ = self.get_projection(self.get_lms(trans_face_model), fx, fy, px, py)
        if is_render:
            img_size, dtype, device = self.img_size, self.dtype, self.device

            camera_rot = torch.eye(3).view(1, 3, 3).to(device)
            camera_rot[0, 0, 0] *= -1.0
            camera_trans = torch.zeros([1, 3]).to(device)

            half_size = (img_size - 1.0) / 2
            focal_length = torch.tensor([fx / half_size, fy / half_size], dtype=torch.float32).reshape(1, 2).to(device)
            principal_point = torch.tensor([(half_size - px) / half_size, (py - half_size) / half_size], dtype=dtype)\
                .reshape(1, 2).to(device)
            cameras = PerspectiveCameras(device=self.device, R=camera_rot, T=camera_trans, focal_length=focal_length,
                                         principal_point=principal_point)
            raster_settings = RasterizationSettings(image_size=img_size, blur_radius=0.0, faces_per_pixel=1)
            lights = PointLights(device=device,
                                 ambient_color=((1.0, 1.0, 1.0),),
                                 diffuse_color=((0.0, 0.0, 0.0),),
                                 specular_color=((0.0, 0.0, 0.0),),
                                 location=((0.0, 0.0, 1e5),))

            blend_params = BlendParams(background_color=(0.0, 0.0, 0.0))

            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings
                ),
                shader=SoftPhongShader(
                    device=device,
                    cameras=cameras,
                    lights=lights,
                    blend_params=blend_params
                )
            )

            # ----------------------------------------------------------------------------------------------------------
            batch_size = tex_code.shape[0]
            face_texture = self.get_color(tex_code)
            face_norm = self.compute_norm(face_model, self.tri, self.point_buf)
            face_norm_r = face_norm.bmm(R)
            face_color = self.add_illumination(face_texture, face_norm_r, light_param)
            face_color_tv = TexturesVertex(face_color.type(torch.float32))

            mesh = Meshes(trans_face_model, self.tri.repeat(batch_size, 1, 1), face_color_tv)
            rendered_image = renderer(mesh.clone())
            rendered_image = torch.clamp(rendered_image, 0, 255)

            ret["rendered_img"] = rendered_image

            # cameras: FoVPerspectiveCameras = self._get_cameras(eye, at, up)
            # self.renderer.rasterizer.cameras = cameras
            # self.renderer.shader.cameras = cameras

            # ret["lms_proj"] = self.project(cameras, self.get_lms(ret["vertices"]))  # 1, 68, 3
            # ret["rendered_image"] = self.render(ret["vertices"], ret["vertices"], texture_param, cameras.R)
        # --------------------------------------------------------------------------------------------------------------

        return ret
