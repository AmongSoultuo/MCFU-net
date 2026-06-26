import torch.utils.data as data
import PIL.Image as Image
import os
import json
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader


def make_dataset(root, seed, name):
    imgs = []
    seed = sorted(seed, key=lambda i: int(i.split(".")[0]))
    for img_name in seed:
        img = os.path.join(root +'/'+ name+ '-image/', img_name)
        mask = os.path.join(root +'/'+ name+ '-mask/', img_name)
        imgs.append((img, mask))
    return imgs


def make_testset(root):
    imgs = []
    img_labels = {}
    img_names = os.listdir(root +'/test-image/')
    img_names = sorted(img_names, key=lambda i: int(i.split(".")[0]))

    for img_name in img_names:
        img = os.path.join(root +'/test-image/', img_name)
        mask = os.path.join(root +'/test-mask/', img_name)
        imgs.append((img, mask))
    return imgs


class TN3K(data.Dataset):
    def __init__(self, mode, transform=None, return_size=False, fold=0, test=False):
        self.mode = mode
        _BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        root = os.path.join(_BASE_DIR, 'data', 'TN3K')
        trainval = json.load(open(os.path.join(root, 'tn3k-trainval-fold' + str(fold) + '.json'), 'r'))
        if mode == 'train':
            imgs = make_dataset(root, trainval['train'], 'trainval')
        elif mode == 'val':
            imgs = make_dataset(root, trainval['val'], 'trainval')
        elif mode == 'test':
            imgs = make_testset(root)

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
    composed_transforms_tr = transforms.Compose([
        trforms.FixedResize(size=(int(256), int(256))),
        trforms.RandomHorizontalFlip(),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()])

    composed_transforms_ts = transforms.Compose([
        trforms.FixedResize(size=(256, 256)),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()])
    train_data = TN3K(mode='train', transform=composed_transforms_tr, return_size=False, fold=0)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0, drop_last=True,
                              pin_memory=True)
    for i, data in enumerate(train_loader):
        print(i, data)
