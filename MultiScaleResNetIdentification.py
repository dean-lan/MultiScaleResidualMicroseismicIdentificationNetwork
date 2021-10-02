import sys
import time

import os

import torch

import numpy as np

import torch.utils.data as Data

from sklearn.model_selection import StratifiedKFold, train_test_split

from torch import nn

from ImprovedMicroseismicEventIdentification.DataSetLoader import DataSetLoader
from ImprovedMicroseismicEventIdentification.MicroseismicEventIdentifier import MicroseismicEventNNIdentifier, \
    cross_validation_evaluation, EarlyStop
from ImprovedMicroseismicEventIdentification.WilkinsCNN import WilkinsCNN


class BasicResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(BasicResBlock, self).__init__()

        self.covNet = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),

            nn.Conv1d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_channel),
        )

        if stride != 1 or out_channel != in_channel:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channel),
            )
        else:
            self.shortcut = nn.Sequential()

        self.activation = nn.Sequential(
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.covNet(x)
        shortcut_out = self.shortcut(x)
        out = self.activation(out + shortcut_out)
        return out


class ProposedResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ProposedResBlock, self).__init__()

        """
        Input   1
        --------
        Cov1    2
        ________
        Cov2    3
        ________
        Cov3    4
        ________
        Output  5
        """
        self.Cov1 = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channel),
        )
        self.Cov2 = nn.Sequential(
            nn.Conv1d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_channel),
        )
        self.Cov3 = nn.Sequential(
            nn.Conv1d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_channel)
        )

        if stride != 1 or out_channel != in_channel:
            self.shortcut13 = nn.Sequential(
                nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channel),
            )
            self.shortcut14 = nn.Sequential(
                nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channel),
            )
        else:
            self.shortcut13 = nn.Sequential()
            self.shortcut14 = nn.Sequential()

        self.shortcut24 = nn.Sequential()

        self.activation = nn.Sequential(
            nn.ReLU(),
        )

    def forward(self, x):
        # out1 = self.shortcut14(x)
        out_cov1 = self.Cov1(x)
        out2 = self.shortcut24(out_cov1)
        out_cov2 = self.Cov2(out_cov1) + self.shortcut13(x)
        out_cov3 = self.Cov3(out_cov2)

        out = out2 + out_cov3

        out = self.activation(out)
        return out


class ProposedResBlockV1(ProposedResBlock):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ProposedResBlockV1, self).__init__(in_channel, out_channel, stride)

    def forward(self, x):
        out1 = self.shortcut14(x)
        out_cov1 = self.Cov1(x)
        out2 = self.shortcut24(out_cov1)
        out_cov2 = self.Cov2(out_cov1) + self.shortcut13(x)
        out_cov3 = self.Cov3(out_cov2)

        out = out1 + out2 + out_cov3

        out = self.activation(out)
        return out


class ProposedResBlockSE(ProposedResBlock):
    def __init__(self, in_channel, out_channel, stride=1, r=16):
        super(ProposedResBlockSE, self).__init__(in_channel, out_channel, stride)

        k = int(out_channel / r)

        k = max(4, k)

        self.SE = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),

            nn.Flatten(),

            nn.Linear(out_channel, k),
            nn.BatchNorm1d(k),
            nn.ReLU(),

            nn.Linear(k, out_channel),
            nn.BatchNorm1d(out_channel),
            nn.Sigmoid(),

            nn.Unflatten(dim=1, unflattened_size=(-1, 1)),
        )

    def forward(self, x):
        out_cov1 = self.Cov1(x)
        out2 = self.shortcut24(out_cov1)
        out_cov2 = self.Cov2(out_cov1) + self.shortcut13(x)
        out_cov3 = self.Cov3(out_cov2)

        scale = self.SE(out_cov3)

        out = out2 + out_cov3 * scale

        out = self.activation(out)
        return out


class ProposedResBlockSEV1(ProposedResBlockV1):
    def __init__(self, in_channel, out_channel, stride=1, r=16):
        super(ProposedResBlockSEV1, self).__init__(in_channel, out_channel, stride)
        k = int(out_channel / r)

        k = max(4, k)

        self.SE = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),

            nn.Flatten(),

            nn.Linear(out_channel, k),
            nn.BatchNorm1d(k),
            nn.ReLU(),

            nn.Linear(k, out_channel),
            nn.BatchNorm1d(out_channel),
            nn.Sigmoid(),

            nn.Unflatten(dim=1, unflattened_size=(-1, 1)),
        )

    def forward(self, x):
        x = super(ProposedResBlockSEV1, self).forward(x)
        scale = self.SE(x)
        x = x * scale
        return x


class ResNetMicroseismicEventIdentifierBlockCreatorBasic(object):
    def __init__(self, n_input, n_hidden, drop_ratio=0.2):
        super(ResNetMicroseismicEventIdentifierBlockCreatorBasic, self).__init__()

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.drop_ratio = drop_ratio

    def get_name(self):
        return 'Basic'

    def set_dense(self, n_input, n_hidden):
        self.n_input = n_input
        self.n_hidden = n_hidden

    def create_block(self, in_channel, out_channel, stride):
        return BasicResBlock(in_channel, out_channel, stride)

    def create_dense(self):
        dense = nn.Sequential(
            nn.Linear(self.n_input, self.n_hidden),
            nn.BatchNorm1d(self.n_hidden),
            nn.ReLU(),

            # nn.Dropout(self.drop_ratio),

            nn.Linear(self.n_hidden, 1),
            nn.Sigmoid(),
        )
        return dense


class ResNetMicroseismicEventIdentifierBlockCreatorProposed(ResNetMicroseismicEventIdentifierBlockCreatorBasic):
    def __init__(self, n_input, n_hidden):
        super(ResNetMicroseismicEventIdentifierBlockCreatorProposed, self).__init__(n_input, n_hidden)

    def get_name(self):
        return 'Proposed'

    def create_block(self, in_channel, out_channel, stride):
        return ProposedResBlock(in_channel, out_channel, stride)


class ResNetMicroseismicEventIdentifierBlockCreatorProposedSE(ResNetMicroseismicEventIdentifierBlockCreatorBasic):
    def __init__(self, n_input, n_hidden):
        super(ResNetMicroseismicEventIdentifierBlockCreatorProposedSE, self).__init__(n_input, n_hidden)

    def get_name(self):
        return 'ProposedSE'

    def create_block(self, in_channel, out_channel, stride):
        return ProposedResBlockSE(in_channel, out_channel, stride)


class ResNetMicroseismicEventIdentifierBlockCreatorProposedV1(ResNetMicroseismicEventIdentifierBlockCreatorBasic):
    def __init__(self, n_input, n_hidden):
        super(ResNetMicroseismicEventIdentifierBlockCreatorProposedV1, self).__init__(n_input, n_hidden)

    def get_name(self):
        return 'ProposedV1'

    def create_block(self, in_channel, out_channel, stride):
        return ProposedResBlockV1(in_channel, out_channel, stride)


class ResNetMicroseismicEventIdentifierBlockCreatorProposedSEV1(ResNetMicroseismicEventIdentifierBlockCreatorBasic):
    def __init__(self, n_input, n_hidden):
        super(ResNetMicroseismicEventIdentifierBlockCreatorProposedSEV1, self).__init__(n_input, n_hidden)

    def get_name(self):
        return 'ProposedSEV1'

    def create_block(self, in_channel, out_channel, stride):
        return ProposedResBlockSEV1(in_channel, out_channel, stride)


class ResNetMicroseismicEventIdentifier(MicroseismicEventNNIdentifier):
    def __init__(self, block_creator):
        super(ResNetMicroseismicEventIdentifier, self).__init__()

        self.block_creator = block_creator

        self.Cov1 = nn.Sequential(
            nn.Conv1d(1, 16, 3, 2),  # [, 8, 1000 + 1 - n_scale]
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
        )

        self.Res = nn.Sequential(
            self.block_creator.create_block(16, 32, 1),
            self.block_creator.create_block(32, 32, 2),
            self.block_creator.create_block(32, 64, 1),
            self.block_creator.create_block(64, 64, 2),
            self.block_creator.create_block(64, 128, 1),
            nn.AvgPool1d(2, 2),
            nn.Flatten(),
        )

        self.Dense = self.block_creator.create_dense()

    def dense_match(self, n_input, n_hidden):
        self.block_creator.set_dense(n_input, n_hidden)
        self.Dense = self.block_creator.create_dense()

    def forward(self, x):
        cov = self.Cov1(x)
        res = self.Res(cov)
        dense = self.Dense(res)
        return dense

    def get_name(self):
        return self.block_creator.get_name()

    def data_preparation(self, x, y):
        x = x[:, 100:-100]
        x = np.reshape(x, newshape=(x.shape[0], 1, x.shape[1]))
        if y is not None:
            y = np.reshape(y, newshape=(y.shape[0], 1))
        return x, y


def MultiScaleMicroseismicEventIdentifierTraining(dataset='./qiu.csv'):
    dataloader = DataSetLoader()
    data_x, data_y = dataloader.load_from_file(file_name=dataset)
    data_x = dataloader.normalization(data_x)

    if os.path.exists('MicroseismicEventIdentifier.pkl'):
        network = torch.load('MicroseismicEventIdentifier.pkl')
        val_x, val_y = network.data_preparation(data_x, data_y)
        val_x = torch.as_tensor(val_x, dtype=torch.float32)
        val_y = torch.as_tensor(val_y, dtype=torch.float32)
    else:
        cv = StratifiedKFold(n_splits=10, shuffle=True)
        train_index, val_index = next(cv.split(data_x, data_y))


        train_x, val_x = data_x[train_index], data_x[val_index]
        train_y, val_y = data_y[train_index], data_y[val_index]


        train_x, train_y = dataloader.augment(data_to_augment_x=train_x, data_to_augment_y=train_y,
                                              shuffle=True)

        network = ResNetMicroseismicEventIdentifier(
            block_creator=ResNetMicroseismicEventIdentifierBlockCreatorProposedSE(n_input=3200, n_hidden=256))

        train_x, train_y = network.data_preparation(train_x, train_y)
        val_x, val_y = network.data_preparation(val_x, val_y)

        train_x = torch.as_tensor(train_x, dtype=torch.float32)
        train_y = torch.as_tensor(train_y, dtype=torch.float32)

        val_x = torch.as_tensor(val_x, dtype=torch.float32)
        val_y = torch.as_tensor(val_y, dtype=torch.float32)

        optimizer = network.create_optimizer(network)
        loss_function = network.create_loss_function()
        early_stop = EarlyStop(patience=10, validation_data_x=val_x, validation_data_y=val_y)
        network, train_losses = network.do_train(network, train_x, train_y, 50,
                                                 loss_function=loss_function, batch_size=32,
                                                 device='cuda', optimizer=optimizer,
                                                 early_stop=early_stop)
        network.eval()
        network.cpu()
        torch.save(network, 'MicroseismicEventIdentifier.pkl')

    predict = network(val_x).detach().cpu().view(-1).numpy()

    scores = cross_validation_evaluation(val_y.detach().cpu().view(-1).numpy(), predict)
    print(scores)
    sys.exit()


if __name__ == '__main__':
    load_history = False

    # MultiScaleMicroseismicEventIdentifierTraining()

    dataloader = DataSetLoader()

    comparison_pairs = {
        './wilkins.csv': (
            WilkinsCNN(n_height=8, n_dense=12620),
            ResNetMicroseismicEventIdentifier(
                block_creator=ResNetMicroseismicEventIdentifierBlockCreatorBasic(n_input=5120, n_hidden=256)),
            ResNetMicroseismicEventIdentifier(
                block_creator=ResNetMicroseismicEventIdentifierBlockCreatorProposed(n_input=5120, n_hidden=256)),
            ResNetMicroseismicEventIdentifier(
                block_creator=ResNetMicroseismicEventIdentifierBlockCreatorProposedSE(n_input=5120, n_hidden=256)),
        ),
        './qiu.csv': (
            WilkinsCNN(n_height=1, n_dense=7620),
            ResNetMicroseismicEventIdentifier(
                block_creator=ResNetMicroseismicEventIdentifierBlockCreatorBasic(n_input=3200, n_hidden=256)),
            ResNetMicroseismicEventIdentifier(
                block_creator=ResNetMicroseismicEventIdentifierBlockCreatorProposed(n_input=3200, n_hidden=256)),
            ResNetMicroseismicEventIdentifier(
                block_creator=ResNetMicroseismicEventIdentifierBlockCreatorProposedSE(n_input=3200, n_hidden=256)),
        ),
    }

    for dataset in comparison_pairs.keys():
        dataset_name = os.path.basename(dataset).split('.')[0]

        networks = comparison_pairs[dataset]

        raw_data_x, raw_data_y = dataloader.load_from_file(file_name=dataset)
        raw_data_x = dataloader.normalization(raw_data_x)

        cv = StratifiedKFold(n_splits=10, shuffle=True)

        for network in networks:
            net_scores = []

            net_scores_file = 'result_%s_%s.txt' % (dataset_name, network.get_name())

            if os.path.exists(net_scores_file) and load_history:
                net_scores = np.loadtxt(net_scores_file)
            else:
                if 'wilkins' in dataset and not isinstance(network, WilkinsCNN):
                    data_x, data_y = dataloader.split(data_to_split_x=raw_data_x,
                                                      data_to_split_y=raw_data_y, n_channel=8)
                else:
                    data_x, data_y = dataloader.split(data_to_split_x=raw_data_x,
                                                      data_to_split_y=raw_data_y, n_channel=1)

                for train_index, test_index in cv.split(data_x, data_y):
                    train_x, test_x = data_x[train_index], data_x[test_index]
                    train_y, test_y = data_y[train_index], data_y[test_index]

                    train_x, train_y = dataloader.augment(data_to_augment_x=train_x, data_to_augment_y=train_y,
                                                          shuffle=True)

                    train_x, train_y = network.data_preparation(train_x, train_y)
                    test_x, test_y = network.data_preparation(test_x, test_y)

                    sub_cv = StratifiedKFold(n_splits=10, shuffle=True)
                    train_index, val_index = next(sub_cv.split(train_x, train_y))
                    train_x, val_x = train_x[train_index], train_x[val_index]
                    train_y, val_y = train_y[train_index], train_y[val_index]

                    train_x = torch.as_tensor(train_x, dtype=torch.float32)
                    test_x = torch.as_tensor(test_x, dtype=torch.float32)
                    train_y = torch.as_tensor(train_y, dtype=torch.float32)
                    test_y = torch.as_tensor(test_y, dtype=torch.float32)

                    val_x = torch.as_tensor(val_x, dtype=torch.float32)
                    val_y = torch.as_tensor(val_y, dtype=torch.float32)

                    net_inst = network
                    optimizer = network.create_optimizer(net_inst)
                    loss_function = net_inst.create_loss_function()
                    early_stop = EarlyStop(patience=10, validation_data_x=val_x, validation_data_y=val_y)

                    net_inst, train_losses = network.do_train(net_inst, train_x, train_y, 50,
                                                              loss_function=loss_function, batch_size=32,
                                                              device='cuda', optimizer=optimizer,
                                                              early_stop=early_stop)
                    net_inst.eval()
                    net_inst.cpu()
                    test_y_predict = net_inst(test_x)
                    score = cross_validation_evaluation(test_y, test_y_predict)
                    net_scores.append(score)

                    network_name = network.get_name() + '_' + dataset_name + '.pkl'
                    torch.save(net_inst, network_name)
                    break

            net_scores = np.asarray(net_scores)

            np.savetxt(net_scores_file, net_scores)

            options = np.get_printoptions()
            np.set_printoptions(linewidth=np.inf)
            print('Network: ', network.get_name(), ', Set: ', dataset, np.mean(net_scores, axis=0))
            np.set_printoptions(**options)
