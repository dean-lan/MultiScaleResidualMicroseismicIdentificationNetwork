import os

import numpy as np
import torch

from ImprovedMicroseismicEventIdentification.DataSetLoader import DataSetLoader
from ImprovedMicroseismicEventIdentification.MultiScaleResNetIdentification import BasicResBlock, \
    ProposedResBlock, \
    ProposedResBlockSE, \
    ResNetMicroseismicEventIdentifier, \
    ResNetMicroseismicEventIdentifierBlockCreatorBasic, ResNetMicroseismicEventIdentifierBlockCreatorProposedSE, \
    ResNetMicroseismicEventIdentifierBlockCreatorProposed
from ImprovedMicroseismicEventIdentification.WilkinsCNN import WilkinsCNN


def obtain_set(labels):
    true_set = set(np.nonzero(labels > 0.0)[0])
    false_set = set(np.nonzero(labels < 1.0)[0])
    return true_set, false_set


if __name__ == '__main__':
    load_history = False

    # MultiScaleMicroseismicEventIdentifierTraining()

    dataloader = DataSetLoader()

    comparison_pairs = {
        './wilkins.csv': (
            WilkinsCNN(n_height=8, n_dense=12620),
            ResNetMicroseismicEventIdentifier(
                block_creator=ResNetMicroseismicEventIdentifierBlockCreatorBasic(n_input=5120, n_hidden=256)),
            # ResNetMicroseismicEventIdentifier(
            #     block_creator=ResNetMicroseismicEventIdentifierBlockCreatorProposed(n_input=5120, n_hidden=256)),
            ResNetMicroseismicEventIdentifier(
                block_creator=ResNetMicroseismicEventIdentifierBlockCreatorProposedSE(n_input=5120, n_hidden=256)),
        ),
        './qiu.csv': (
            WilkinsCNN(n_height=1, n_dense=7620),
            ResNetMicroseismicEventIdentifier(
                block_creator=ResNetMicroseismicEventIdentifierBlockCreatorBasic(n_input=3200, n_hidden=256)),
            # ResNetMicroseismicEventIdentifier(
            #     block_creator=ResNetMicroseismicEventIdentifierBlockCreatorProposed(n_input=3200, n_hidden=256)),
            ResNetMicroseismicEventIdentifier(
                block_creator=ResNetMicroseismicEventIdentifierBlockCreatorProposedSE(n_input=3200, n_hidden=256)),
        ),
    }

    for dataset in comparison_pairs.keys():
        dataset_name = os.path.basename(dataset).split('.')[0]

        networks = comparison_pairs[dataset]

        raw_data_x, raw_data_y = dataloader.load_from_file(file_name=dataset)
        raw_data_x = dataloader.normalization(raw_data_x)

        if os.path.exists(dataset_name + '_result.txt'):
            network_labels = np.loadtxt(dataset_name + '_result.txt')
        else:
            network_labels = []
            for network in networks:
                if 'wilkins' in dataset and not isinstance(network, WilkinsCNN):
                    data_x, data_y = dataloader.split(data_to_split_x=raw_data_x,
                                                      data_to_split_y=raw_data_y, n_channel=8)
                else:
                    data_x, data_y = dataloader.split(data_to_split_x=raw_data_x,
                                                      data_to_split_y=raw_data_y, n_channel=1)

                data_x, data_y = network.data_preparation(data_x, data_y)
                data_x = torch.as_tensor(data_x, dtype=torch.float32)

                network_name = network.get_name() + '_' + dataset_name + '.pkl'
                net_inst = torch.load(network_name)
                net_inst.eval()
                net_inst.cpu()
                labels = net_inst(data_x)
                labels = labels.detach().cpu().numpy()
                labels = np.reshape(labels, -1)

                labels = np.where(labels < 0.5, 0.0, 1.0)

                if 'wilkins' in dataset and isinstance(network, WilkinsCNN):
                    labels_list = []
                    for l in labels:
                        if l > 0.0:
                            labels_list += [1.0] * 8
                        else:
                            labels_list += [0.0] * 8
                    labels = np.asarray(labels_list)

                network_labels.append(labels)

            if 'wilkins' in dataset:
                y_list = []
                for y in raw_data_y:
                    if y > 0.0:
                        y_list += [1.0] * 8
                    else:
                        y_list += [0.0] * 8
                raw_data_y = np.asarray(y_list)

            network_labels = np.asarray(network_labels)

            np.savetxt(dataset_name + '_result.txt', network_labels)

        if 'wilkins' in dataset_name:
            raw_data_x, raw_data_y = dataloader.split(data_to_split_x=raw_data_x,
                                                      data_to_split_y=raw_data_y, n_channel=8)

        real_labels = raw_data_y
        wilkins_labels = network_labels[0, :]
        basic_labels = network_labels[1, :]
        proposed_labels = network_labels[2, :]

        true_set, false_set = obtain_set(real_labels)
        wilkins_true_set, wilkins_false_set = obtain_set(wilkins_labels)
        basic_true_set, basic_false_set = obtain_set(basic_labels)
        proposed_true_set, proposed_false_set = obtain_set(proposed_labels)

        proposed_predicted_correct = (true_set & proposed_true_set).union(false_set & proposed_false_set)
        wilkins_predicted_correct = (true_set & wilkins_true_set).union(false_set & wilkins_false_set)
        basic_predicted_correct = (true_set & basic_true_set).union(false_set & basic_false_set)

        proposed_predicted_incorrect = (true_set & proposed_false_set).union(false_set & proposed_true_set)
        wilkins_predicted_incorrect = (true_set & wilkins_false_set).union(false_set & wilkins_true_set)
        basic_predicted_incorrect = (true_set & basic_false_set).union(false_set & basic_true_set)

        both_incorrect = wilkins_predicted_incorrect & basic_predicted_incorrect
        but_correct = proposed_predicted_correct & both_incorrect

        but_correct = np.asarray(list(but_correct))
        data = raw_data_x[but_correct, :]
        data = (data - np.min(data, axis=1, keepdims=True)) / (
                np.max(data, axis=1, keepdims=True) - np.min(data, axis=1, keepdims=True))

        print(raw_data_y[but_correct])

        np.savetxt(dataset_name + '_show.txt', data)

        print(dataset_name, but_correct)

        print('hello')
