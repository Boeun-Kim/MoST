import numpy as np
import random
import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler


def make_weighted_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))


def create_data_loader(args, cfg, mtype='content', split=None):

    dataset = MotionData(args, cfg, mtype)

    if mtype == 'content':
        sampler = make_weighted_sampler(dataset.cnt_label)
    elif mtype == 'style':
        sampler = make_weighted_sampler(dataset.sty_label)
    else:
        raise NotImplementedError

    return data.DataLoader(dataset=dataset,
                        batch_size=args.batch_size,
                        sampler=sampler,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        drop_last=True)

    return data.DataLoader(dataset=dataset, batch_size=args.batch_size)


class MotionData(data.Dataset):
    def __init__(self, args, cfg, mtype='content'):

        path = args.train_datapath

        dist_path = args.dist_datapath
        self.clips = np.load(path)['clips']
        self.foot_contact = np.load(path)['feet']
        self.cnt_label = np.load(path)['cnt']
        self.sty_label = np.load(path)['sty']
        self.mean = np.load(dist_path)['Xmean']
        self.std = np.load(dist_path)['Xstd']
        self.mtype = mtype

        # Preparing 2nd style motion
        if self.mtype == 'style':
            self.num_sty = len(cfg["style_names"])
            self.num_cnt = len(cfg["content_names"])

            self.clip_split = [] 
            for i in range(self.num_cnt):
                clip_split_sty = []
                for j in range(self.num_sty):
                    select_cnt = np.zeros(self.cnt_label.shape, dtype=bool)
                    select_sty = np.zeros(self.sty_label.shape, dtype=bool)
                    select_cnt[self.cnt_label==i] = True
                    select_sty[self.sty_label==j] = True
                    select = select_cnt * select_sty
                    clip_split_sty.append(self.clips[select])

                self.clip_split.append(clip_split_sty)

    def __len__(self):
        return len(self.clips)

    def normalize(self, x, mean, std):
        x = (x - mean) / std
        return x
    
    def rand_crop(self, x, y):
        seq_len = sum(~np.isnan(x[0,:,0]))
        if seq_len <= 64:
            return x, y
        else:
            rand_len = np.random.randint(64, seq_len)
            rand_st = np.random.randint(0, seq_len-rand_len)

            x_cropped = x[:,rand_st:rand_st+rand_len,:]
            y_cropped = y[rand_st:rand_st+rand_len,:]
            x = np.empty(x.shape)
            y = np.empty(y.shape)
            x[:] = np.nan
            y[:] = np.nan
            x[:,:rand_len,:] = x_cropped
            y[:rand_len,:] = y_cropped
            return x, y

    def __getitem__(self, index):

        if self.mtype == 'content':
            clip = self.clips[index]
            foot_contact = self.foot_contact[index]
            cnt_label = self.cnt_label[index]
            sty_label = self.sty_label[index]

            # Random crop for data augmentation
            clip, foot_contact = self.rand_crop(clip, foot_contact)
            norm_clip = self.normalize(clip, self.mean, self.std)

            return {'clip': norm_clip, 'contact':foot_contact, 'cnt_label': cnt_label}

        else:
            clip = self.clips[index]
            cnt_label = self.cnt_label[index]
            sty_label = self.sty_label[index]

            norm_clip = self.normalize(clip, self.mean, self.std)

            # Preparing 2nd style motion
            # Choose a style motion with same style but different content
            numbers_cnt = list(range(self.num_cnt))
            numbers_cnt.remove(cnt_label)
            rand_cnt_label = random.choice(numbers_cnt)

            numbers_motion = self.clip_split[rand_cnt_label][sty_label].shape[0]
            rand_motion_label = random.randrange(numbers_motion)

            clip2 = self.clip_split[rand_cnt_label][sty_label][rand_motion_label]
            norm_clip2 = self.normalize(clip2, self.mean, self.std)

            return {'clip': norm_clip, 'clip2': norm_clip2, 'sty_label': sty_label}
        

class InputFetcher:
    def __init__(self, args, content_loader, style_loader):
        self.cnt_data_loader = content_loader
        self.sty_data_loader = style_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fetch_cnt(self):
        try:
            data = next(self.iter_cnt)
        except (AttributeError, StopIteration):
            self.iter_cnt = iter(self.cnt_data_loader)
            data = next(self.iter_cnt)
        return data

    def fetch_sty(self):
        try:
            data = next(self.iter_sty)
        except (AttributeError, StopIteration):
            self.iter_sty = iter(self.sty_data_loader)
            data = next(self.iter_sty)
        return data

    def __next__(self):
        
        data = {}
        cnt = self.fetch_cnt()
        data_cnt = {'cnt_clip': cnt['clip'].to(self.device), 'cnt_contact': cnt['contact'].to(self.device), 
                    'cnt_label': cnt['cnt_label'].to(self.device)}
        data.update(data_cnt)

        sty = self.fetch_sty()
        data_sty = {'sty_clip': sty['clip'].to(self.device), 'sty_clip2': sty['clip2'].to(self.device), 
                    'sty_label': sty['sty_label'].to(self.device)}
        data.update(data_sty)

        return data