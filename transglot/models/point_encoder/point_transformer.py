import torch.nn as nn
from transglot.models.point_encoder.modules.point_transformer_modules import *


class PT(nn.Module):

    def __init__(self, output_dim, dim=None, sample_ratio=4, k=16):
        super().__init__()
        self.output_dim = output_dim
        if dim == None:
            dim = [3, 32, 64, 128, 256, 512]

        self.Encoder = nn.ModuleList()
        for i in range(len(dim) - 1):
            if i == 0:
                self.Encoder.append(MLP(dim[i], dim[i + 1]))
            else:
                self.Encoder.append(TransitionDown(dim[i], dim[i + 1], sample_ratio, k))
            self.Encoder.append(PointTransformerBlock(dim[i + 1], k))

        self.Decoder = nn.ModuleList()
        for i in range(len(dim) - 1, 0, -1):
            if i == len(dim) - 1:
                self.Decoder.append(MLP(dim[i], dim[i]))
            else:
                self.Decoder.append(TransitionUp(dim[i + 1], dim[i], sample_ratio))
            self.Decoder.append(PointTransformerBlock(dim[i], k))

        self.fc_layer = nn.Sequential(
            nn.Conv1d(dim[1], 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Conv1d(128, self.output_dim, 1)
        )

    def forward(self, xyz):
        '''
        Input:
            xyz: [B*3, 2048, 3]
        '''
        B, N, _ = xyz.size()
        xyz_list, f_list = [xyz], [xyz]

        for i in range(0, len(self.Encoder), 2):
            if i == 0:
                f = self.Encoder[i](f_list[-1])
            else:
                f, xyz = self.Encoder[i](f_list[-1], xyz_list[-1])
            f, attn_weights = self.Encoder[i + 1](f, xyz)

            xyz_list.append(xyz)
            f_list.append(f)

        for i in range(0, len(self.Decoder), 2):
            if i == 0:
                f = self.Decoder[i](f_list[-1])
            else:
                f, xyz = self.Decoder[i](f, xyz, f_list[-i // 2 - 1], xyz_list[-i // 2 - 1])
            f, attn_weights = self.Decoder[i + 1](f, xyz)

        pred = self.fc_layer(f.transpose(1,2)).transpose(1,2)
        return pred