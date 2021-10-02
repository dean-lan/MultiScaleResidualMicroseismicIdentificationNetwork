import os

import time

import torch

import numpy as np

from torchstat import stat

from ImprovedMicroseismicEventIdentification.MultiScaleResNetIdentification import \
    ResNetMicroseismicEventIdentifierBlockCreatorProposedSE, ResNetMicroseismicEventIdentifier
from ImprovedMicroseismicEventIdentification.WilkinsCNN import WilkinsCNN
from MSPicker.SemiAutomaticPicker import SLTAPicker

if __name__ == '__main__':
    data_length = {
        500: (4620, 1920),
        1000: (9620, 3968),
        5000: (49620, 19968),
        10000: (99620, 39936),
        20000: (199620, 80000),
        50000: (499620, 199936),
        100000: (999620, 400000)
    }
    data_num = 1000

    devices = ['cpu', 'cuda']
    networks = [
        WilkinsCNN(n_height=1, n_dense=4620),
        ResNetMicroseismicEventIdentifier(
            block_creator=ResNetMicroseismicEventIdentifierBlockCreatorProposedSE(n_input=1920, n_hidden=256))
    ]


    def net_evaluation(_net, _data, _device, shape):
        _net.eval()
        _net.to(_device)
        _data = torch.as_tensor(_data, dtype=torch.float32, device=_device)
        _data = torch.reshape(_data, shape=shape)
        _s_time = time.time()
        _net(_data)
        _cost = time.time() - _s_time
        return _cost


    for device in devices:
        for dlength in data_length.keys():
            times_file = '%s_%d.txt' % (device, dlength)

            if os.path.exists(times_file):
                times = np.loadtxt(times_file)
            else:
                parameters = data_length[dlength]

                data = np.random.randn(data_num, dlength)

                s_time = time.time()
                [SLTAPicker(_d).pick(st=60, lt=300, threshold=0.1) for _d in data]
                times = [(time.time() - s_time) / data_num]

                data = torch.as_tensor(data, dtype=torch.float32)

                net = WilkinsCNN(n_height=1, n_dense=parameters[0])
                # print(stat(net, (1, 1, dlength)))
                costs = np.asarray([net_evaluation(net, _d, device, shape=(1, 1, 1, dlength)) for _d in data])
                times.append(np.mean(costs))

                input_data = torch.reshape(data, shape=(data_num, 1, dlength))
                net = ResNetMicroseismicEventIdentifier(
                    block_creator=ResNetMicroseismicEventIdentifierBlockCreatorProposedSE(n_input=parameters[1],
                                                                                          n_hidden=256))
                costs = np.asarray([net_evaluation(net, _d, device, shape=(1, 1, dlength)) for _d in data])
                times.append(np.mean(costs))

                np.savetxt(times_file, times)

            print(device, dlength, times)
