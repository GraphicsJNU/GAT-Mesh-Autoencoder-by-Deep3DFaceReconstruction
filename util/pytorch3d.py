"""
    This script is the differentiable renderer for My Project
    Attention, antialiasing step is missing in current version.
"""

import json
import torch
from torch import nn
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.renderer.utils import (
    TensorProperties,
    convert_to_tensors_and_broadcast
)
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer as Renderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)
from pytorch3d.renderer.blending import BlendParams
import numpy as np
from typing import List


def ndc_projection(x=0.1, n=1.0, f=50.0):
    return np.array([[n / x, 0, 0, 0],
                     [0, n / -x, 0, 0],
                     [0, 0, -(f + n) / (f - n), -(2 * f * n) / (f - n)],
                     [0, 0, -1, 0]]).astype(np.float32)


# https://github.com/pomelyu/SHLight_pytorch/tree/main
class SphericalHarmonicsLights(TensorProperties):
    def __init__(
            self,
            sh_params=None,
            device="cpu",
    ):
        super().__init__(
            device=device,
            ambient_color=((0.0, 0.0, 0.0),),
            sh_params=sh_params,
        )

        if self.sh_params is None:
            with open('./models/gat/datas/face_39674.rps') as f:
                info = json.load(f)
            self.sh_params = torch.Tensor(info["environmentMap"]["coefficients"]).unsqueeze(0)
        elif self.sh_params.shape[-2:] != (9, 3):
            msg = "Expected sh_params to have shape (N, 9, 3); got %r"
            raise ValueError(msg % repr(self.sh_params.shape))

        pi = np.pi
        sqrt = np.sqrt
        att = pi * np.array([1., 2. / 3., 1. / 4.])
        sh_coeff = torch.FloatTensor([
            att[0] * (1 / 2) * (1 / sqrt(pi)),  # 1
            att[1] * (sqrt(3) / 2) * (1 / sqrt(pi)),  # Y
            att[1] * (sqrt(3) / 2) * (1 / sqrt(pi)),  # Z
            att[1] * (sqrt(3) / 2) * (1 / sqrt(pi)),  # X
            att[2] * (sqrt(15) / 2) * (1 / sqrt(pi)),  # YX
            att[2] * (sqrt(15) / 2) * (1 / sqrt(pi)),  # YZ
            att[2] * (sqrt(5) / 4) * (1 / sqrt(pi)),  # 3Z^2 - 1
            att[2] * (sqrt(15) / 2) * (1 / sqrt(pi)),  # XZ
            att[2] * (sqrt(15) / 4) * (1 / sqrt(pi)),  # X^2 - Y^2
        ])
        self.register_buffer("sh_coeff", sh_coeff[None, None, :])

    def clone(self, **kwargs):
        other = self.__class__(device=self.device)

        return super().clone(other)

    def diffuse(self, normals, points) -> torch.Tensor:
        # normals: (B, ..., 3)
        input_shape = normals.shape
        B = input_shape[0]
        normals = normals.view(B, -1, 3)
        # normals: (B, K, 3)

        sh = torch.stack([
            torch.ones_like(normals[..., 0]),  # 1
            normals[..., 1],  # Y
            normals[..., 2],  # Z
            normals[..., 0],  # X
            normals[..., 1] * normals[..., 0],  # YX
            normals[..., 1] * normals[..., 2],  # YZ
            3 * (normals[..., 2] ** 2) - 1,  # 3Z^2 - 1
            normals[..., 0] * normals[..., 2],  # XZ
            normals[..., 0] ** 2 - normals[..., 1] ** 2,  # X^2 - Y^2
        ], dim=-1)
        # sh: (B, K, 9)

        sh, sh_coeff, sh_params = convert_to_tensors_and_broadcast(sh, self.sh_coeff, self.sh_params,
                                                                   device=normals.device)

        sh = sh * sh_coeff
        # sh_params: (B, 9, 3)
        color = torch.einsum("ijk,ikl->ijl", sh, sh_params)
        color = color.view(B, *input_shape[1:-1], 3)

        return color

    def specular(self, normals, points, camera_position, shininess) -> torch.Tensor:
        return torch.zeros_like(points)


class MeshRenderer(nn.Module):
    def __init__(self,
                 rasterize_fov,
                 znear=0.1,
                 zfar=10,
                 rasterize_size=224,
                 camera_z=10.,
                 device='cpu'):
        super(MeshRenderer, self).__init__()

        x = np.tan(np.deg2rad(rasterize_fov * 0.5)) * znear
        self.ndc_proj = torch.tensor(ndc_projection(x=x, n=znear, f=zfar)).matmul(
            torch.diag(torch.tensor([1., -1, -1, 1])))

        # Initialize a camera.
        # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction.
        # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow.
        R, T = look_at_view_transform(camera_z, 0, 0)
        cameras = FoVPerspectiveCameras(
            znear=znear,
            zfar=zfar,
            fov=rasterize_fov,
            R=R,
            T=T,
            device=device
        )

        raster_settings = RasterizationSettings(
            image_size=rasterize_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        sh_lights = SphericalHarmonicsLights(device=device)

        self.rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        )
        self.shader = SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=sh_lights,
            blend_params=BlendParams(background_color=(0., 0., 0.))
        )

        self.renderer = Renderer(
            rasterizer=self.rasterizer,
            shader=self.shader
        )

    def forward(self, vtx, tri, tex=None, uv=None, uv_idx=None):
        """
        Return:
            mask               -- torch.tensor, size (B, 1, H, W)
            depth              -- torch.tensor, size (B, 1, H, W)
            features(optional) -- torch.tensor, size (B, C, H, W) if feat is not None


        Parameters:
            vtx:            -- torch.tensor, size (B, N, 3)
            tri:            -- torch.tensor, size (B, M, 3) or (M, 3), triangles
            tex:            -- torch.tensor, size (B, C, H, W), texture image
            uv:             -- torch.tensor, size (B, N, 2), uv texture image coordinates
            uv_idx:         -- torch.tensor, size (B, M, 3), uv indices
        """
        device = vtx.device
        ndc_proj = self.ndc_proj.to(device)
        # trans to homogeneous coordinates of 3d vertices, the direction of y is the same as v
        if vtx.shape[-1] == 3:
            vtx = torch.cat([vtx, torch.ones([*vtx.shape[:2], 1]).to(device)], dim=-1)
            vtx[..., 1] = -vtx[..., 1]

        vertex_ndc = vtx @ ndc_proj.t()

        # ranges = None
        # if isinstance(tri, List) or len(tri.shape) == 3:
        #     vum = vertex_ndc.shape[1]
        #     fnum = torch.tensor([f.shape[0] for f in tri]).unsqueeze(1).to(device)
        #     fstartidx = torch.cumsum(fnum, dim=0) - fnum
        #     ranges = torch.cat([fstartidx, fnum], axis=1).type(torch.int32).cpu()
        #     for i in range(tri.shape[0]):
        #         tri[i] = tri[i] + i * vum
        #     vertex_ndc = torch.cat(vertex_ndc, dim=0)
        #     tri = torch.cat(tri, dim=0)

        # for range_mode vetex: [B*N, 4], tri: [B*M, 3], for instance_mode vetex: [B, N, 4], tri: [M, 3]
        # vertex_ndc = vtx
        # tri = tri.type(torch.int32).contiguous()
        # rast_out, rast_out_db = dr.rasterize(self.ctx, vertex_ndc.contiguous(), tri,
        #                                      resolution=[rsize, rsize], ranges=ranges)
        #
        # depth, _ = dr.interpolate(vtx.reshape([-1, 4])[..., 2].unsqueeze(1).contiguous(), rast_out, tri)
        #
        # depth = depth.permute(0, 3, 1, 2)
        # mask = (rast_out[..., 3] > 0).float().unsqueeze(1)
        # depth = mask * depth

        # def photo_loss(pred_img, gt_img, img_mask):
        #     pred_img = pred_img.float()
        #     loss = torch.sqrt(torch.sum(torch.square(pred_img - gt_img), 3)) * img_mask / 255
        #     loss = torch.sum(loss, dim=(1, 2)) / torch.sum(img_mask, dim=(1, 2))
        #     loss = torch.mean(loss)
        #
        #     return loss

        vtx = vertex_ndc[..., :3]

        image = None
        if tex is None:
            meshes_world = Meshes(verts=vtx, faces=tri)
            fragments = self.rasterizer(meshes_world)
            mask = (fragments.zbuf[..., 0] > 0).float().unsqueeze(1)
        else:
            texture = TexturesUV(maps=tex, faces_uvs=uv_idx.to(device), verts_uvs=uv.to(device))
            meshes_world = Meshes(verts=vtx, faces=tri, textures=texture, verts_normals=None)
            fragments = self.rasterizer(meshes_world)
            image = self.shader(fragments, meshes_world)
            mask = (image[..., 3] > 0).float().unsqueeze(1)
            image = torch.clamp(image[..., :3].permute(0, 3, 1, 2), 0, 255)

        return mask, image
