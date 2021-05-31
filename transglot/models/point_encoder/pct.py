import torch
import torch.nn as nn
import math


class PCT(nn.Module):

    def __init__(self, output_dim):
        super().__init__()
        self.input_embedding = NeighborEmbedding(3, 128, 16)
        self.sa1 = OA(128)
        self.sa2 = OA(128)
        self.sa3 = OA(128)
        self.sa4 = OA(128)
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(4*128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.convs1 = nn.Sequential(
            nn.Conv1d(1024*3, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.convs2 = nn.Sequential(
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, output_dim, 1)
        )

    def forward(self, x):
        B,N,_=x.size()
        x = self.input_embedding(x)
        x1, attn1 = self.sa1(x)
        x2, attn2 = self.sa2(x1)
        x3, attn3 = self.sa3(x2)
        x4, attn4 = self.sa4(x3)
        x = torch.cat([x1, x2, x3, x4], dim=-1)
        x = self.conv_fuse(x.transpose(1,2))
        x_max = x.max(2)[0].unsqueeze(-1).repeat(1,1,N)
        x_avg = x.mean(2).unsqueeze(-1).repeat(1,1,N)
        x = torch.cat([x, x_max, x_avg], dim=1)
        x = self.convs1(x)
        x = self.convs2(x)

        return x.transpose(1,2) #B,N,C




class NaiveEmbedding(nn.Module):

    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(True),
            nn.Conv1d(output_dim, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.mlp(x.transpose(1,2)).transpose(1,2).contiguous()


class NeighborEmbedding(NaiveEmbedding):

    def __init__(self, input_dim, output_dim=128, k=32):
        super().__init__(input_dim, output_dim//2)
        self.k = k
        self.local_op = nn.Sequential(
            nn.Conv1d(output_dim, output_dim, 1, bias=False),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Conv1d(output_dim, output_dim, 1, bias=False),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )


    def forward(self, x):
        xyz = x
        x = self.mlp(x.transpose(1,2)).transpose(1,2).contiguous() #BND
        x_repeat = x.unsqueeze(2).expand(-1,-1,self.k,-1) #BNkD
        knn_idx = kNN_torch(xyz, xyz, k=self.k)  # BNk
        knn_x = gather_knn(x, knn_idx)  # BNkD
        x = knn_x - x_repeat
        x = torch.cat([x, x_repeat], dim=-1) #B N k 2D
        B, N, k, D = x.size()
        x = x.transpose(2,3).reshape(-1, D, k) # B*N D k
        x = self.local_op(x).reshape(B, N, D, k)
        x = x.max(-1)[0]  # B N D
        return x

class SA(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.to_q = nn.Linear(dim, dim//4)
        self.to_k = nn.Linear(dim, dim//4)
        self.to_v = nn.Linear(dim, dim)
        self.lbr = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        attn_weights = torch.einsum('bnd,bmd->bnm', q, k) / math.sqrt(q.size(-1))
        attn_weights = torch.softmax(attn_weights, -1)

        attn = torch.einsum('bnm,bmd->bnd', attn_weights, v)

        return self.lbr(attn.transpose(1,2)).transpose(1,2).contiguous() + x, attn_weights


class OA(SA):

    def __init__(self, dim):
        super().__init__(dim)

    def forward(self, x):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        attn_weights = torch.einsum('bnd,bmd->bnm', q, k)
        attn_weights = torch.softmax(attn_weights, dim=1)
        attn_weights = attn_weights / (1e-9 + torch.sum(attn_weights, 2, keepdim=True))

        attn = torch.einsum('bnm,bmd->bnd', attn_weights, v)

        return self.lbr((x-attn).transpose(1, 2)).transpose(1, 2).contiguous() + x, attn_weights