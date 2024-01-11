import numpy as np
import lmdb, pickle
from torch.utils.data import Dataset

class dataloader(Dataset):
    def __init__(self, lmdb_root, mode, mean, std):
        # open database for reading
        self.env = lmdb.open(lmdb_root, readonly=True)
        self.cursor = list(enumerate(self.env.begin(write=False).cursor()))
        self.nSamples = self.env.stat()['entries']
        self.mode = mode
        self.mean, self.std = mean, std

    def __getitem__(self, index):
        # read image from self.cursor and convert byte data to array
        databin = pickle.loads(self.cursor[index][1][1])
        if self.mode == 'train':
                image = np.expand_dims(self.data_normaliser(databin), axis=0).astype(np.float32)
                return image
        elif self.mode == 'eval':
            reference = np.expand_dims(self.data_normaliser(databin[0]), axis=0).astype(np.float32)
            probe = np.expand_dims(self.data_normaliser(databin[1]), axis=0).astype(np.float32)
            pair = [reference, probe]
            return pair
        else:
            print('mode must be specified - train or eval')
            return -1

    def __len__(self):
        return self.nSamples

    def data_normaliser(self, data):
        data = (np.array(data) - self.mean) / self.std

        return data

def mean_and_var(root):
    env = lmdb.open(root, readonly=True)
    cursor = list(enumerate(env.begin(write=False).cursor()))
    mean_lmdb = 0.0
    var_lmdb = 0.0
    for image in cursor:
        img = pickle.loads(image[1][1])
        mean_lmdb += np.mean(img)
        var_lmdb += np.std(img)

    mean_lmdb /= env.stat()['entries']
    var_lmdb /= env.stat()['entries']

    return mean_lmdb, var_lmdb