import numpy as np
from torch.utils.data import Dataset
import h5py
import os.path as osp
import torch

class ShapeglotDataset(Dataset):
    def __init__(self, np_data, shuffle_geo=False, target_last=False):
        """
        :param np_data:
        :param shuffle_geo: if True, the positions of the shapes in context are randomly swapped.
        """
        super(ShapeglotDataset, self).__init__()
        self.data = np_data
        self.shuffle_geo = shuffle_geo
        self.target_last = target_last

    def __getitem__(self, index):
        text = self.data['text'][index].astype(np.long)
        geos = self.data['in_geo'][index].astype(np.long)
        target = self.data['target'][index]
        idx = np.arange(len(geos))

        if self.shuffle_geo:
            np.random.shuffle(idx)

        geos = geos[idx]
        target = np.argmax(target[idx])

        if self.target_last:
            last = geos[-1]
            geos[-1] = geos[target]
            geos[target] = last
            target = len(geos) - 1

        return geos, target, text

    def __len__(self):
        return len(self.data)


class ShapeglotWithPCDataset(ShapeglotDataset):
    def __init__(self, np_data, num_points=2048, shuffle_geo=False, target_last=False):
        super().__init__(np_data=np_data, shuffle_geo=shuffle_geo, target_last=target_last)
        self.num_points = num_points
        self.pc_data = h5py.File('./data/shapenet_chairs_only_in_game.h5', 'r')['data'][:,:self.num_points]

    def __getitem__(self, index):
        text = self.data['text'][index].astype(np.long)
        geos_idx = self.data['in_geo'][index].astype(np.long)
        geos = self.pc_data[geos_idx]
        target = self.data['target'][index]
        idx = np.arange(len(geos))

        if self.shuffle_geo:
            np.random.shuffle(idx)
        geos = torch.from_numpy(geos[idx]).float()
        geos_idx = geos_idx[idx]
        target = np.argmax(target[idx])

        if self.target_last:
            last = geos[-1]
            last_idx = geos_idx[-1]
            geos[-1] = geos[target]
            geos_idx[-1] = geos_idx[target]
            geos[target] = last
            geos_idx[target] = last
            target = len(geos) - 1

        return geos, geos_idx, target, text




