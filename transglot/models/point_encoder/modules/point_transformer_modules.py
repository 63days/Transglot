import torch.nn as nn
import torch.nn.functional as F
import math
from transglot.models.point_encoder.modules.utils import *

class PointTransformerLayer(nn.Module):

    def __init__(self, dim, pos_hid_dim, k):
        super(PointTransformerLayer, self).__init__()
        self.k = k

        self.phi_linear = nn.Linear(dim, dim)
        self.psi_linear = nn.Linear(dim, dim)
        self.alpha_linear = nn.Linear(dim, dim)
        self.gamma_ft = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim))
        self.pos_embed = nn.Sequential(
            nn.Linear(3, pos_hid_dim),
            nn.ReLU(),
            nn.Linear(pos_hid_dim, dim))

    def forward(self, input_feature, xyz):
        '''
        Input:
            feature: [B, N, D]
            xyz: coordinates [B, N, 3]
        Output:
            new_feature: [B, N, D]
        '''
        knn_index = kNN_torch(xyz, xyz, self.k)
        knn_xyz = gather_knn(xyz, knn_index)  # BNk3
        knn_feature = gather_knn(input_feature, knn_index)  # BNkD

        pos_enc = self.pos_embed(xyz[:,:,None] - knn_xyz)

        q = self.phi_linear(input_feature)
        k = self.psi_linear(knn_feature) # BNkD
        v = self.alpha_linear(knn_feature)

        attn_weights = self.gamma_ft(q[:,:,None]-k+pos_enc)
        attn_weights = torch.softmax(attn_weights / math.sqrt(k.size(-1)), dim=2)
        # output_feature = torch.einsum('bnkd,bnkd->bnd', attn_weights, v+pos_enc)
        output_feature = torch.sum(attn_weights * (v+pos_enc), dim=2)
        return output_feature, attn_weights


class MLP(nn.Module): # To Remove

    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(True),
            nn.Conv1d(output_dim, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(True)
        )

    def forward(self, input_feature):
        output_feature = (self.mlp(input_feature.transpose(1, 2))
                          .transpose(1,2)
                          .contiguous())

        return output_feature


class PointTransformerBlock(nn.Module):

    def __init__(self, dim, k):
        super(PointTransformerBlock, self).__init__()
        self.front_linear = nn.Linear(dim, dim)
        self.attn_layer = PointTransformerLayer(dim=dim, pos_hid_dim=dim, k=k)
        self.back_linear = nn.Linear(dim, dim)

    def forward(self, input_feature, xyz):
        identity = input_feature

        output_feature = self.front_linear(input_feature)
        output_feature, attn_weights = self.attn_layer(output_feature, xyz)
        output_feature = self.back_linear(output_feature)

        return output_feature + identity, attn_weights


class TransitionDown(nn.Module):

    def __init__(self, input_dim, output_dim, sample_ratio, k):
        super(TransitionDown, self).__init__()
        self.sample_ratio = sample_ratio
        self.k = k
        self.linear = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, input_feature, xyz):
        sample_idx = furthest_point_sample(xyz, xyz.size(1) // self.sample_ratio)
        sample_xyz = (gather_operation(xyz.transpose(1, 2).contiguous(), sample_idx)
                      .transpose(1, 2)
                      .contiguous())  # BS3

        knn_idx = kNN_torch(sample_xyz, xyz, k=self.k)  # BSk
        sample_knn_feature = gather_knn(input_feature, knn_idx)  # BSkD

        sample_feature = self.linear(sample_knn_feature)
        B, S, K, D = sample_feature.shape
        sample_feature = (F.relu(self.bn(sample_feature.reshape(B, -1, D).transpose(1, 2)))
                          .transpose(1, 2)
                          .reshape(B, S, K, D)
                          .contiguous())
        sample_feature = sample_feature.max(2)[0]  # BSD

        return sample_feature, sample_xyz


class TransitionUp(nn.Module):

    def __init__(self, input_dim, output_dim, sample_ratio):
        super(TransitionUp, self).__init__()
        self.sample_ratio = sample_ratio
        self.linear1 = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(True)
        )
        self.linear2 = nn.Sequential(
            nn.Conv1d(output_dim, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(True)
        )

    def forward(self, sample_feature, sample_xyz, skip_feature, skip_xyz):
        '''
        Input:
            sample_feature: [B,S,D]
            sample_xyz: [B,S,3]
            skip_feature: [B,N,D] skip-connected feature
            skip_xyz: [B,N,3] skip-connected xyz
        Output:
            unpooled_feature: [B,N,D]
            xyz: [B,N,3]
        '''

        sample_feature = (self.linear1(sample_feature.transpose(1, 2))
                          .transpose(1, 2)
                          .contiguous())

        unpooled_feature = trilinear_interpolation(sample_feature, sample_xyz, skip_xyz)

        skip_feature = (self.linear2(skip_feature.transpose(1, 2))
                        .transpose(1, 2)
                        .contiguous())

        unpooled_feature += skip_feature

        return unpooled_feature, skip_xyz


class Up(nn.Module):

    def __init__(self, input_dim, output_dim, sample_ratio, k):
        super(Up, self).__init__()
        self.unpool = TransitionUp(input_dim, output_dim, sample_ratio)
        self.transformer = PointTransformerBlock(output_dim, k)

    def forward(self, sample_feature, sample_xyz, skip_feature, skip_xyz):
        unpooled_feature, skip_xyz = self.unpool(sample_feature, sample_xyz, skip_feature, skip_xyz)
        unpooled_feature = self.transformer(unpooled_feature, skip_xyz)

        return unpooled_feature


class Down(nn.Module):

    def __init__(self, input_dim, output_dim, sample_ratio, k):
        super(Down, self).__init__()
        self.pool = TransitionDown(input_dim, output_dim, sample_ratio, k)
        self.transformer = PointTransformerBlock(output_dim, k)

    def forward(self, input_feature, xyz):
        sample_feature, sample_xyz = self.pool(input_feature, xyz)
        output_feature = self.transformer(sample_feature, sample_xyz)

        return output_feature, sample_xyz
