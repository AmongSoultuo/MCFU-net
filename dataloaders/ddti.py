import json
import os

import PIL.Image as Image
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms


def make_dataset(root, seed):
    imgs = []
    seed = sorted(seed, key=lambda i: int(i.split(".")[0]))
    for img_name in seed:
        img = os.path.join(root + '/' + 'p_image/', img_name)
        mask = os.path.join(root + '/' + 'p_mask/', img_name)
        imgs.append((img, mask))
    return imgs



class DDTI(data.Dataset):
    def __init__(self, mode, transform=None, return_size=False, fold=0, test=False):
        self.mode = mode
        _BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        root = os.path.join(_BASE_DIR, 'data', 'DDTI', '2_preprocessed_data', 'stage1')
        trainval = json.load(open(os.path.join(root, 'ddti-trainval-fold' + str(fold) + '.json'), 'r'))
        if mode == 'train':
            imgs = make_dataset(root, trainval['train'])
        elif mode == 'val':
            imgs = make_dataset(root, trainval['val'])
        elif mode == 'test':
            imgs = make_dataset(root, trainval['test'])

        self.imgs = imgs
        self.transform = transform
        self.return_size = return_size
        self.test = test

    def __getitem__(self, item):
        image_path, label_path = self.imgs[item]
        assert os.path.exists(image_path), ('{} does not exist'.format(image_path))
        assert os.path.exists(label_path), ('{} does not exist'.format(label_path))
        image = Image.open(image_path).convert('RGB')
        label = np.array(Image.open(label_path).convert('L'))
        label = label / label.max()
        label = Image.fromarray(label.astype(np.uint8))
        w, h = image.size
        size = (h, w)

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)
        if self.test:
            sample['size'] = torch.tensor(size)
            label_name = os.path.basename(label_path)
            sample['label_name'] = label_name
        return sample

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    train_data = DDTI(mode='train', return_size=False, fold=1, test=False)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
    for i, data in enumerate(train_loader):
        print(i, data)
