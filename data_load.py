import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os


def accuracy(output, labels):
    """Compute micro F1 score for classification accuracy"""
    preds = output.max(dim=1)[1].view(-1, 1).cpu()
    correct = labels.cpu()
    from sklearn.metrics import f1_score
    micro_f1 = f1_score(correct, preds, average='micro', zero_division=1)
    return micro_f1


def get_dataset_dim(dataset):
    """Get the feature dimension for each dataset"""
    dim = 0
    if dataset == 'EMG':
        dim = 416
    elif dataset == 'Oppo':
        dim = 160
    elif dataset == 'UCIHAR':
        dim = 16
    elif dataset == 'Uni':
        dim = 464
    return dim


def get_dataset_classes(dataset):
    """Get the number of classes for each dataset"""
    num_classes = 0
    if dataset == 'EMG':
        num_classes = 6
    elif dataset == 'Oppo':
        num_classes = 18
    elif dataset == 'UCIHAR':
        num_classes = 6
    elif dataset == 'Uni':
        num_classes = 17
    return num_classes


def process_domain(domains):
    """Convert domain labels to consecutive integers starting from 0"""
    u_domains = torch.unique(domains).numpy()
    to_new_domain = {}
    for i in range(u_domains.shape[0]):
        to_new_domain[u_domains[i]] = i
    domains = domains.numpy()
    for i in range(domains.shape[0]):
        domains[i] = to_new_domain[domains[i]]
    return torch.LongTensor(domains)


def oversample_minority(features, labels, domains, target_ratio=0.2):
    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_feats = features[pos_mask]
    pos_labels = labels[pos_mask]
    pos_domains = domains[pos_mask]

    neg_feats = features[neg_mask]
    neg_labels = labels[neg_mask]
    neg_domains = domains[neg_mask]

    current_ratio = pos_labels.size(0) / labels.size(0)
    desired_num_pos = int(target_ratio / (1 - target_ratio) * neg_labels.size(0))
    repeat_factor = max(1, int(desired_num_pos / pos_labels.size(0)))
    repeat_factor = min(repeat_factor, 20)

    pos_feats_os = pos_feats.repeat(repeat_factor, 1)
    pos_labels_os = pos_labels.repeat(repeat_factor)
    pos_domains_os = pos_domains.repeat(repeat_factor)

    features = torch.cat([neg_feats, pos_feats_os], dim=0)
    labels = torch.cat([neg_labels, pos_labels_os], dim=0)
    domains = torch.cat([neg_domains, pos_domains_os], dim=0)

    # Shuffle
    perm = torch.randperm(features.size(0))
    return features[perm], labels[perm], domains[perm]


def load_boiler_data(path, target_domain, device):
    """Load Boiler dataset for domain adaptation"""
    domains = ['1', '2', '3']
    assert target_domain in domains

    train_features, train_labels, train_domains = None, None, None
    test_features, test_labels = None, None

    for domain in domains:
        file_path = os.path.join(path, 'Boiler', f'boiler_domain_{domain}.npz')
        data = np.load(file_path)
        features = torch.FloatTensor(data['features'])
        labels = torch.LongTensor(data['labels'])
        domains_tensor = torch.LongTensor([int(domain) - 1] * len(labels))

        if domain != target_domain:
            if train_features is None:
                train_features, train_labels, train_domains = features, labels, domains_tensor
            else:
                train_features = torch.cat((train_features, features), dim=0)
                train_labels = torch.cat((train_labels, labels), dim=0)
                train_domains = torch.cat((train_domains, domains_tensor), dim=0)
        else:
            test_features = features
            test_labels = labels

    num_class = len(torch.unique(train_labels))
    num_domain = 3  # Domain 1, 2, 3

    train_features, train_labels, train_domains = oversample_minority(train_features, train_labels, train_domains,
                                                                      target_ratio=0.2)

    return train_features.to(device), train_labels.to(device), train_domains.to(device), \
           test_features.to(device), test_labels.to(device), num_class, num_domain


def load_data(dataset, path, target_domain, device):
    train_features = None
    train_labels = None
    train_domains = None
    test_features = None
    test_labels = None

    assert dataset in ['EMG', 'UCIHAR', 'Uni', 'Oppo']

    if dataset == 'EMG':
        domains = ['0', '1', '2', '3']
        assert target_domain in domains
        for domain in domains:
            file_name = 'emg_domain_{}.npz'.format(domain)
            domain_data = np.load(path + dataset + '/' + file_name)
            temp_features = torch.FloatTensor(domain_data['features'])
            temp_labels = torch.LongTensor(domain_data['labels'])
            temp_domains = torch.LongTensor(domain_data['domains'])
            if domain != target_domain:
                if train_features is None:
                    train_features = temp_features
                    train_labels = temp_labels
                    train_domains = temp_domains
                else:
                    train_features = torch.cat((train_features, temp_features), dim=0)
                    train_labels = torch.cat((train_labels, temp_labels))
                    train_domains = torch.cat((train_domains, temp_domains))
            else:
                test_features = temp_features
                test_labels = temp_labels
    elif dataset == 'UCIHAR':
        domains = ['0', '1', '2', '3', '4']
        assert target_domain in domains
        for domain in domains:
            file_name = 'ucihar_domain_{}_wd.data'.format(domain)
            domain_data = np.load(path + dataset + '/' + file_name, allow_pickle=True)
            temp_features = torch.FloatTensor(domain_data[0][0])
            temp_labels = torch.LongTensor(domain_data[0][1])
            temp_domains = torch.LongTensor(domain_data[0][2])
            if domain != target_domain:
                if train_features is None:
                    train_features = temp_features
                    train_labels = temp_labels
                    train_domains = temp_domains
                else:
                    train_features = torch.cat((train_features, temp_features), dim=0)
                    train_labels = torch.cat((train_labels, temp_labels))
                    train_domains = torch.cat((train_domains, temp_domains))
            else:
                test_features = temp_features
                test_labels = temp_labels
    elif dataset == 'Uni':
        dataset_path = 'UniMiB-SHAR'
        domains = ['1', '2', '3', '5']
        assert target_domain in domains
        for domain in domains:
            file_name = 'shar_domain_{}_wd.data'.format(domain)
            domain_data = np.load(path + dataset_path + '/' + file_name, allow_pickle=True)
            temp_features = torch.FloatTensor(domain_data[0][0])
            temp_features = torch.unsqueeze(temp_features, dim=1)  # Add channel dimension
            temp_labels = torch.LongTensor(domain_data[0][1])
            temp_domains = torch.LongTensor(domain_data[0][2])
            if domain != target_domain:
                if train_features is None:
                    train_features = temp_features
                    train_labels = temp_labels
                    train_domains = temp_domains
                else:
                    train_features = torch.cat((train_features, temp_features), dim=0)
                    train_labels = torch.cat((train_labels, temp_labels))
                    train_domains = torch.cat((train_domains, temp_domains))
            else:
                test_features = temp_features
                test_labels = temp_labels
    elif dataset == 'Oppo':
        domains = ['S1', 'S2', 'S3', 'S4']
        assert target_domain in domains
        dataset_path = 'Opportunity'
        for domain in domains:
            file_name = 'oppor_domain_{}_wd.data'.format(domain)
            domain_data = np.load(path + dataset_path + '/' + file_name, allow_pickle=True)
            temp_features = torch.FloatTensor(domain_data[0][0])
            temp_features = torch.unsqueeze(temp_features, dim=1)  # Add channel dimension
            temp_labels = torch.LongTensor(domain_data[0][1])
            temp_domains = torch.LongTensor(domain_data[0][2])
            if domain != target_domain:
                if train_features is None:
                    train_features = temp_features
                    train_labels = temp_labels
                    train_domains = temp_domains
                else:
                    train_features = torch.cat((train_features, temp_features), dim=0)
                    train_labels = torch.cat((train_labels, temp_labels))
                    train_domains = torch.cat((train_domains, temp_domains))
            else:
                test_features = temp_features
                test_labels = temp_labels

    num_class = torch.unique(train_labels).shape[0]
    num_domain = torch.unique(train_domains).shape[0]

    return train_features.to(device), train_labels.to(device), process_domain(train_domains).to(device), \
        test_features.to(device), test_labels.to(device), int(num_class), int(num_domain)


class DomainDataset(Dataset):

    def __init__(self, features, labels, domains=None):

        self.features = features
        self.labels = labels
        self.domains = domains

    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, idx):
        if self.domains is not None:
            return self.features[idx], self.labels[idx], self.domains[idx]
        else:
            return self.features[idx], self.labels[idx], torch.zeros(1)


def create_data_loaders(dataset_name, data_path, target_domain, batch_size, device):

    if dataset_name == 'Boiler':
        train_features, train_labels, train_domains, test_features, test_labels, num_classes, num_domains = \
            load_boiler_data(data_path, target_domain, device)

        train_dataset = DomainDataset(train_features, train_labels, train_domains)
        test_dataset = DomainDataset(test_features, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        input_dim = train_features.shape[1]

        return train_loader, test_loader, num_classes, num_domains, input_dim
    
    else:
        train_features, train_labels, train_domains, test_features, test_labels, num_classes, num_domains = \
            load_data(dataset_name, data_path, target_domain, device)

        train_dataset = DomainDataset(train_features, train_labels, train_domains)
        test_dataset = DomainDataset(test_features, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        if dataset_name in ['Uni', 'Oppo']:
            input_dim = 1
        else:
            if len(train_features.shape) > 2:
                input_dim = train_features.shape[1]
            else:
                input_dim = 1

        return train_loader, test_loader, num_classes, num_domains, input_dim