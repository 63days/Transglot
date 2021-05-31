import torch
import torch.nn as nn
import torch.nn.functional as F


class PN(nn.Module):

    def __init__(self, output_dim=1024, use_tnet=False, dropout_rate=0.1):
        super().__init__()
        self.output_dim = output_dim
        self.use_tnet = use_tnet

        self.conv1 = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64))
        self.conv3 = nn.Sequential(nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64))
        self.conv4 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.conv5 = nn.Conv1d(128, 1024, 1)

        self.seg_conv1 = nn.Sequential(nn.Conv1d(1088, 256, 1), nn.BatchNorm1d(256))
        self.seg_conv2 = nn.Sequential(nn.Conv1d(256, 256, 1), nn.BatchNorm1d(256))
        self.seg_conv3 = nn.Sequential(nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128))
        #self.dp = nn.Dropout(p=dropout_rate)
        self.seg_conv4 = nn.Conv1d(128, output_dim, 1)

        if use_tnet == True:
            self.tnet3 = TNet3()
            self.tnetk = TNetK(k=64)

    def forward(self, x):
        B, N, _ = x.size()
        if self.use_tnet:
            transform = self.tnet3(x.transpose(1,2))
            x = torch.matmul(x, transform)

        x = x.transpose(1,2)
        out1 = F.relu(self.conv1(x), True)
        out2 = F.relu(self.conv2(out1), True)

        if self.use_tnet:
            self.transformk = self.tnetk(out2)
            out2 = torch.matmul(out2.transpose(1,2), self.transformk).transpose(1,2)
        out3 = F.relu(self.conv3(out2), True)
        out4 = F.relu(self.conv4(out3), True)
        out5 = self.conv5(out4)

        out_max = out5.max(2)[0].unsqueeze(-1).expand(-1, -1, N)
        concat = torch.cat([out2, out_max], dim=1)

        out = F.relu(self.seg_conv1(concat), True)
        out = F.relu(self.seg_conv2(out), True)
        out = F.relu(self.seg_conv3(out), True)
#         out = self.dp(out)
        out = self.seg_conv4(out)

        return out.transpose(1,2)


class PNCls(nn.Module):
    def __init__(self, output_dim=1024, use_tnet=False, dropout_rate=0.1):
        super().__init__()
        self.output_dim = output_dim
        self.use_tnet = use_tnet

        self.conv1 = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64))
        self.conv3 = nn.Sequential(nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64))
        self.conv4 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.conv5 = nn.Conv1d(128, 1024, 1)

        self.mlp = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(256, output_dim, 1)
        )

    def forward(self, x):
        x = x.transpose(1,2)
        x = F.relu(self.conv1(x), True)
        x = F.relu(self.conv2(x), True)
        x = F.relu(self.conv3(x), True)
        x = F.relu(self.conv4(x), True)
        x = F.relu(self.conv5(x), True)

        x = x.max(2)[0].unsqueeze(2) #[B,1024,1]
        x = self.mlp(x).squeeze(2)
        return x




class PNForTest(nn.Module):

    def __init__(self, output_dim=1024):
        super().__init__()
        self.output_dim = output_dim
        self.tnet3 = TNet3()
        self.conv1 = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64))
        self.tnetk = TNetK(k=64)
        self.conv3 = nn.Sequential(nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64))
        self.conv4 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.conv5 = nn.Conv1d(128, output_dim, 1)

    def forward(self, x):
        transform = self.tnet3(x.transpose(1,2))
        x = torch.matmul(x, transform).transpose(1,2)

        out1 = F.relu(self.conv1(x), True)
        out2 = F.relu(self.conv2(out1), True)

        self.transformk = self.tnetk(out2)
        transformk_out = torch.matmul(out2.transpose(1,2), self.transformk).transpose(1,2)
        out3 = F.relu(self.conv3(transformk_out), True)
        out4 = F.relu(self.conv4(out3), True)
        out5 = self.conv5(out4)
        return out5.transpose(1,2).contiguous() #[B,num_point,channel]

class TNet3(nn.Module):

    def __init__(self, k=3):
        super().__init__()
        assert k == 3
        self.k = k
        # mlp(64, 128, 1024) max pooling-> 2fc(512, 256)
        self.conv1 = nn.Sequential(nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))

        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

        self.weights = nn.Parameter(torch.zeros(128, 3 * 3, requires_grad=True, dtype=torch.float32, device='cuda'))
        self.bias = nn.Parameter(
            torch.zeros(3 * 3, requires_grad=True, dtype=torch.float32, device='cuda') + torch.eye(3,
                                                                                                   device='cuda').flatten())

    def forward(self, x):
        B, _, _ = x.size()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.max(2)[0]  # B, D
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))

        transform = torch.matmul(x, self.weights) + self.bias
        transform = transform.reshape(B, self.k, self.k)

        return transform


class TNetK(nn.Module):

    def __init__(self, k):
        super().__init__()
        self.k = k
        # mlp(64, 128, 1024) max pooling-> 2fc(512, 256)
        self.conv1 = nn.Sequential(nn.Conv1d(k, 256, 1), nn.BatchNorm1d(256))
        self.conv2 = nn.Sequential(nn.Conv1d(256, 1024, 1), nn.BatchNorm1d(1024))

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        self.weights = nn.Parameter(torch.zeros(256, k * k, requires_grad=True, dtype=torch.float32, device='cuda'))
        self.bias = nn.Parameter(
            torch.zeros(k * k, requires_grad=True, dtype=torch.float32, device='cuda') + torch.eye(k,
                                                                                                   device='cuda').flatten())

    def forward(self, x):
        B, _, _ = x.size()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.max(2)[0]
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))

        transform = torch.matmul(x, self.weights) + self.bias
        transform = transform.reshape(B, self.k, self.k)

        return transform