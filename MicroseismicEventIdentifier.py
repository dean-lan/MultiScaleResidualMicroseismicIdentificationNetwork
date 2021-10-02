import os
from copy import deepcopy

import numpy as np
import torch

from sklearn import metrics

from sklearn.model_selection import train_test_split

from torch import nn

import torch.utils.data as Data


def cross_validation_evaluation(y_true, y_predict):
    if not isinstance(y_true, np.ndarray):
        y_true = y_true.detach().cpu().numpy()
    if not isinstance(y_predict, np.ndarray):
        y_predict = y_predict.detach().cpu().numpy()

    y_true = y_true.reshape(-1)
    y_predict = y_predict.reshape(-1)

    auc = metrics.roc_auc_score(y_true, y_predict)

    y_predict = np.round(y_predict)

    precision = metrics.precision_score(y_true, y_predict)
    acc = metrics.accuracy_score(y_true, y_predict)
    recall = metrics.recall_score(y_true, y_predict)
    f1 = metrics.f1_score(y_true, y_predict)

    t = np.sum(y_predict)
    f = len(y_predict) - t

    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_predict).ravel()
    return [acc, precision, recall, f1, auc, tn, fp, fn, tp, t, f]


class EarlyStop(object):
    def __init__(self, **kwargs):
        self.delta = kwargs.get('delta', 0.0)
        self.patience = kwargs.get('patience', 10)

        self.validation_x = kwargs.get('validation_data_x')
        self.validation_y = kwargs.get('validation_data_y')

        self.dataloader = None

        self.val_losses = []

        self.stop_counts = 0

        self.best_score = None

        self.best_model = None

    def create_validation_set(self, batch_size, shuffle):
        if self.dataloader is None:
            self.dataloader = Data.DataLoader(
                dataset=Data.TensorDataset(self.validation_x, self.validation_y),
                shuffle=shuffle,
                batch_size=batch_size
            )
        return self.dataloader

    def is_stop(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.best_model = deepcopy(model.state_dict())
            self.stop_counts = 0
            return False

        if val_loss < self.best_score + self.delta:
            if val_loss < self.best_score:
                self.best_score = val_loss
                self.best_model = deepcopy(model.state_dict())
            self.stop_counts = 0
            return False

        self.stop_counts += 1
        return self.stop_counts >= self.patience

    def get_validation_data(self):
        return self.validation_x, self.validation_y

    def load_best_model(self, model):
        model.load_state_dict(self.best_model)
        return model


class MicroseismicEventNNIdentifier(nn.Module):
    def __init__(self):
        super(MicroseismicEventNNIdentifier, self).__init__()

        self.net = None

    @staticmethod
    def create_optimizer(net, **kwargs):
        lr = kwargs.get('lr', 0.001)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        return optimizer

    @staticmethod
    def create_loss_function(**kwargs):
        return nn.BCELoss()

    def data_preparation(self, x, y):
        return x, y

    @staticmethod
    def do_train(net, train_x, train_y, epochs, loss_function, batch_size, device, optimizer, early_stop=None):
        train_loader = Data.DataLoader(
            dataset=Data.TensorDataset(train_x, train_y),
            batch_size=batch_size,
            shuffle=True,
        )

        net.to(device=device)

        train_losses = []

        for epoch in range(epochs):
            net.train()

            train_loss = 0
            train_num = 0
            for step, (b_x, b_y) in enumerate(train_loader):
                b_x = b_x.to(device)
                b_y = b_y.to(device)
                out = net(b_x)
                loss = loss_function(out, b_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * b_x.size(0)
                train_num += b_x.size(0)

            if early_stop is None:
                continue
            val_dataloader = early_stop.create_validation_set(batch_size=batch_size, shuffle=False)

            net.eval()
            val_loss = 0
            val_num = 0
            for step, (b_x, b_y) in enumerate(val_dataloader):
                b_x = b_x.to(device)
                b_y = b_y.to(device)
                out = net(b_x)
                loss = loss_function(out, b_y)

                val_loss += loss.item() * b_x.size(0)
                val_num += b_x.size(0)

            train_losses.append(train_loss / train_num)

            val_loss = val_loss / val_num

            if early_stop.is_stop(val_loss, net):
                break

        if early_stop is not None:
            net = early_stop.load_best_model(net)

        return net, train_losses

    def get_name(self):
        raise NotImplementedError

    def forward(self, x):
        return self.net(x)

    # def fit(self, **kwargs):
    #     samples = kwargs.get('samples')
    #     targets = kwargs.get('targets')
    #     epochs = kwargs.get('epochs', 500)
    #     batch = kwargs.get('batch', 32)
    #     device = kwargs.get('device', 'cuda')
    #     net_name = kwargs.get('net_name', 'MultiScaleResNet.pkl')
    #
    #     if net_name is not None and os.path.exists(net_name):
    #         net = torch.load(net_name)
    #         self.net = net
    #         return net
    #
    #     loss_function = kwargs.get('loss_function', nn.CrossEntropyLoss())
    #
    #     train_x, test_x, train_y, test_y = train_test_split(samples, targets, test_size=0.2)
    #     train_x = torch.as_tensor(train_x, dtype=torch.float32)
    #     test_x = torch.as_tensor(test_x, dtype=torch.float32)
    #     train_y = torch.as_tensor(train_y, dtype=torch.float32)
    #     test_y = torch.as_tensor(test_y, dtype=torch.float32)
    #
    #     net = self._build_net()
    #     optimizer = self._create_optimizer(net, **kwargs)
    #
    #     net, train_losses, val_losses = self._do_train(net, train_x, test_x, train_y, test_y, epochs, loss_function,
    #                                                    batch_size=batch, device=device, optimizer=optimizer)
    #
    #     if net_name is not None:
    #         torch.save(net, net_name)
    #
    #     self.net = net
    #
    # def cross_validation(self, **kwargs):
    #     samples = kwargs.get('samples')
    #     targets = kwargs.get('targets')
    #     epochs = kwargs.get('epochs', 500)
    #     batch = kwargs.get('batch', 32)
    #     device = kwargs.get('device', 'cuda')
    #
    #     evaluation_function = kwargs.get('evaluation_function')
    #
    #     cv = kwargs.get('cv')
    #
    #     loss_function = kwargs.get('loss_function', nn.CrossEntropyLoss())
    #
    #     scores = []
    #     for train_index, test_index in cv.split(samples, targets):
    #         train_x, test_x = samples[train_index], samples[test_index]
    #         train_y, test_y = targets[train_index], targets[test_index]
    #
    #         train_x = torch.as_tensor(train_x, dtype=torch.float32)
    #         test_x = torch.as_tensor(test_x, dtype=torch.float32)
    #         train_y = torch.as_tensor(train_y, dtype=torch.float32)
    #         test_y = torch.as_tensor(test_y, dtype=torch.float32)
    #
    #         net = self._build_net()
    #         optimizer = self._create_optimizer(net, **kwargs)
    #
    # net, train_losses, val_losses = self._do_train(net, train_x, test_x, train_y, test_y, epochs, loss_function,
    # batch_size=batch, device=device, optimizer=optimizer) net.eval() net.cpu() test_y_predict = net(test_x) score =
    # evaluation_function(test_y, test_y_predict) scores.append(score) scores = np.asarray(scores) print(np.mean(
    # scores, axis=0))
    #
    # def predict(self, **kwargs):
    #     samples = kwargs.get('samples')
    #     device = kwargs.get('device', 'cpu')
    #     samples = torch.as_tensor(samples, dtype=torch.float32, device=device)
    #     out = self.net(samples).detach().cpu()
    #     return out
    #
    # def predict_classes(self, **kwargs):
    #     return self.predict(**kwargs)
