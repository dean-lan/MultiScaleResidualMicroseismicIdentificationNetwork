import numpy as np
import torch
from torch import nn

from sklearn.model_selection import StratifiedKFold

from ImprovedMicroseismicEventIdentification.DataSetLoader import DataSetLoader
from ImprovedMicroseismicEventIdentification.MicroseismicEventIdentifier import MicroseismicEventNNIdentifier, \
    cross_validation_evaluation


class WilkinsCNN(MicroseismicEventNNIdentifier):
    def __init__(self, n_height=1, n_dense=12620, n_hidden=200):
        super(WilkinsCNN, self).__init__()

        self.n_height = n_height

        self.Cov = nn.Sequential(
            nn.Conv2d(1, 20, (n_height, 10)),  # [, 20, 1, 1291]
            # nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv2d(20, 10, (1, 30)),  # [10, 1, 1262]
            # nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Flatten(),
        )
        self.Dense = nn.Sequential(
            nn.Linear(n_dense, n_hidden),
            # nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(n_hidden, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def create_optimizer(net, **kwargs):
        lr = kwargs.get('lr', 0.01)
        lr_decay = kwargs.get('lr_decay', 0.0002)
        momentum = kwargs.get('momentum', 0.9)

        optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=lr_decay, momentum=momentum, nesterov=True)
        return optimizer

    def data_preparation(self, x, y):
        n_samples = x.shape[0]
        n_channel = 1
        n_height = self.n_height
        n_width = int(x.shape[1] / self.n_height)

        new_x = np.reshape(x, newshape=(n_samples, n_channel, n_height, n_width))
        new_x = new_x[:, :, :, 100:-100]
        y = np.reshape(y, newshape=(y.shape[0], 1))
        return new_x, y

    def get_name(self):
        return 'Wilkins'

    def forward(self, x):
        cov = self.Cov(x)
        dense = self.Dense(cov)
        return dense

