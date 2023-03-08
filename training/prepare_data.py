import torch.utils.data as data
from torch.utils.data import Sampler, WeightedRandomSampler
import torch
import random
import numpy as np


class Dataset(data.Dataset):

    def __init__(self, data, target, static=None):

        # self.ti_data = ti_data
        self.data = data
        self.static = static
        self.target = target

    def __getitem__(self, index):
        """
         Args:
             index (int): Index
         Returns:
             tuple: time-series data, static data, target if static not None, otherwise return time-series data and target
         """
        data, target = self.data[index], self.target[index]
        data = np.float32(data)
        target = np.uint8(target)

        if not self.static:
            return data, target
        else:
            static = self.static[index]
            static = np.float32(static)
            return data, static, target

    def __len__(self):
        return len(self.target)


def col_fn(batchdata):
    # dat = [train_dataset[i] for i in range(32)]
    len_data = len(batchdata)
    # in batchdata, shape [(182, 48)]
    seq_len = [batchdata[i][0].shape[-1] for i in range(len_data)]
    # [(48, ), (28, ), (100, )....]
    len_tem = [np.zeros((batchdata[i][0].shape[-1])) for i in range(len_data)]
    max_len = max(seq_len)

    # [(182, 48) ---> (182, 100)]
    padded_td = [np.pad(batchdata[i][0], pad_width=((0, 0), (0, max_len - batchdata[i][0].shape[-1])), \
                        mode='constant', constant_values=-3) for i in range(len_data)]
    # [0, 1, 0, 0, 0, ...]
    padded_label = [batchdata[i][1] for i in range(len_data)]

    # [(48, ) ---> (100, )]
    mask = [np.pad(len_tem[i], pad_width=((0, max_len - batchdata[i][0].shape[-1])), \
                   mode='constant', constant_values=1) for i in range(len_data)]

    return torch.from_numpy(np.stack(padded_td)), torch.from_numpy(np.asarray(padded_label)).unsqueeze(-1), \
           torch.from_numpy(np.stack(mask))


def get_data_loader(args, train_head, dev_head, test_head, \
                    train_target, dev_target, test_target):
    '''
    :param args: arugments
    :param train_head: train data
    :param dev_head: dev data
    :param test_head: test data
    :param train_target: train target
    :param dev_target: dev target
    :param test_target: test target
    :return:
    '''
    train_dataset = Dataset(train_head, train_target)
    val_dataset = Dataset(dev_head, dev_target)
    test_dataset = Dataset(test_head, test_target)

    train_len = [train_head[i].shape[1] for i in range(len(train_head))]
    len_range = [i for i in range(0, 219)]
    train_hist, _ = np.histogram(train_len, bins=len_range)

    batch_sizes = args.bs
    val_batch_sizes = args.bs
    test_batch_sizes = args.bs
    # batch_sizes could be class 'numpy.int64'
    if not isinstance(batch_sizes, int):
        batch_sizes = batch_sizes.item()
        val_batch_sizes = val_batch_sizes.item()
        test_batch_sizes = test_batch_sizes.item()

    ctype, count = np.unique(train_target, return_counts=True)
    total_samples = len(train_target)
    weights_per_class = [total_samples / k / len(ctype) for k in count]
    weights = [weights_per_class[int(train_target[i])] for i in range(int(total_samples))]
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(total_samples))
    # sampler = BalancedSampler(train_sofa_tail, batch_sizes)
    # dev_sampler = EvalSampler(dev_head, val_bucket_boundaries, val_batch_sizes)
    # test_sampler = EvalSampler(test_head, bucket_boundaries, test_batch_sizes)
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_sizes,
                                       sampler=sampler, collate_fn=col_fn,
                                       drop_last=False, pin_memory=False)

    dev_dataloader = data.DataLoader(val_dataset, batch_size=val_batch_sizes,
                                     collate_fn=col_fn,
                                     drop_last=False, pin_memory=False)
    test_dataloader = data.DataLoader(test_dataset, batch_size=test_batch_sizes,
                                      collate_fn=col_fn,
                                      drop_last=False, pin_memory=False)

    return train_dataloader, dev_dataloader, test_dataloader


def get_huge_dataloader(args, train_head, dev_head, test_head, \
                        train_target, dev_taregt, test_target):
    '''
    For cross validation
    :param args: arguments
    :param train_head: train data
    :param dev_head: dev data
    :param test_head: test data
    :param train_target: train target
    :param dev_taregt: dev target
    :param test_target: test target
    :return: dataloader
    '''
    total_head = train_head + dev_head + test_head
    total_target = np.concatenate((train_target, dev_taregt, test_target), axis=0)
    train_dataset = Dataset(total_head, total_target)
    if not isinstance(args.bs, int):
        bs = args.bs.item()

    dataloader = data.DataLoader(train_dataset, batch_size=bs,
                                 collate_fn=col_fn,
                                 drop_last=False, pin_memory=False)

    return dataloader


def get_test_loader(args, test_head, test_target):
    '''
    :param args: arguments
    :param test_head: test data
    :param test_target: test target
    :return: test loader
    '''
    if not isinstance(args.bs, int):
        bs = args.bs.item()
    else:
        bs = args.bs
    test_dataset = Dataset(test_head, test_target)
    testloader = data.DataLoader(test_dataset, batch_size=bs,
                                 collate_fn=col_fn,
                                 drop_last=False, pin_memory=False)
    return testloader





