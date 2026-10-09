import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=512, dataset_type='image'):
        super(FeatureExtractor, self).__init__()
        self.dataset_type = dataset_type
        self.hidden_dim = hidden_dim

        if dataset_type == 'image':
            self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=5, stride=1, padding=2)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
            self.bn2 = nn.BatchNorm2d(128)
            self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2)
            self.bn3 = nn.BatchNorm2d(256)
            self.conv4 = nn.Conv2d(256, hidden_dim, kernel_size=5, stride=1, padding=2)
            self.bn4 = nn.BatchNorm2d(hidden_dim)
            self.output_dim = hidden_dim * 2 * 2

        elif dataset_type == 'time_series':
            self.conv1d_1 = nn.Conv1d(input_dim, 64, kernel_size=3, stride=1, padding=1)
            self.bn1d_1 = nn.BatchNorm1d(64)
            self.conv1d_2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
            self.bn1d_2 = nn.BatchNorm1d(128)
            self.conv1d_3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
            self.bn1d_3 = nn.BatchNorm1d(256)
            self.conv1d_4 = nn.Conv1d(256, hidden_dim, kernel_size=3, stride=1, padding=1)
            self.bn1d_4 = nn.BatchNorm1d(hidden_dim)

            self.adaptive_pool = nn.AdaptiveMaxPool1d(4)
            self.output_dim = hidden_dim * 4

        elif dataset_type == 'uni_shar':
            self.conv1d_1 = nn.Conv1d(input_dim, 64, kernel_size=3, stride=1, padding=1)
            self.bn1d_1 = nn.BatchNorm1d(64)
            self.conv1d_2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
            self.bn1d_2 = nn.BatchNorm1d(128)
            self.conv1d_3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
            self.bn1d_3 = nn.BatchNorm1d(256)
            self.conv1d_4 = nn.Conv1d(256, hidden_dim, kernel_size=3, stride=1, padding=1)
            self.bn1d_4 = nn.BatchNorm1d(hidden_dim)

            self.adaptive_pool = nn.AdaptiveMaxPool1d(4)
            self.fc = nn.Linear(hidden_dim * 4, hidden_dim)
            self.output_dim = hidden_dim
            self.hidden_dim = hidden_dim

        elif dataset_type == 'opportunity':
            self.conv1d_1 = nn.Conv1d(input_dim, 64, kernel_size=3, stride=1, padding=1)
            self.bn1d_1 = nn.BatchNorm1d(64)
            self.conv1d_2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
            self.bn1d_2 = nn.BatchNorm1d(128)
            self.conv1d_3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
            self.bn1d_3 = nn.BatchNorm1d(256)
            self.conv1d_4 = nn.Conv1d(256, hidden_dim, kernel_size=3, stride=1, padding=1)
            self.bn1d_4 = nn.BatchNorm1d(hidden_dim)

            self.adaptive_pool = nn.AdaptiveMaxPool1d(4)
            self.fc = nn.Linear(hidden_dim * 4, hidden_dim)
            self.output_dim = hidden_dim

        elif dataset_type == 'tabular':
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.n_steps = 3
            self.n_glu = 2

            self.bn_input = nn.BatchNorm1d(input_dim, eps=1e-5, track_running_stats=True)

            self.embed = nn.Linear(input_dim, hidden_dim)

            self.shared_blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GLU(),  # Gated Linear Unit
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GLU()
                ) for _ in range(self.n_glu)
            ])

            self.decision_steps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                ) for _ in range(self.n_steps)
            ])

            self.attentive_blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, input_dim),
                    nn.Sigmoid()
                ) for _ in range(self.n_steps)
            ])

            self.output_dim = hidden_dim
        else:
            self.fc1 = nn.Linear(input_dim, 512)
            self.bn_fc1 = nn.BatchNorm1d(512, track_running_stats=True)
            self.fc2 = nn.Linear(512, 512)
            self.bn_fc2 = nn.BatchNorm1d(512, track_running_stats=True)
            self.fc3 = nn.Linear(512, hidden_dim)
            self.bn_fc3 = nn.BatchNorm1d(hidden_dim, track_running_stats=True)
            self.output_dim = hidden_dim

    def forward(self, x):
        if self.dataset_type == 'image':
            # 2D CNN path
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.bn4(self.conv4(x)))
            x = F.max_pool2d(x, 2)
            return x
        
        elif self.dataset_type == 'time_series':
            if len(x.shape) == 2:
                x = x.unsqueeze(1)

            x = F.relu(self.bn1d_1(self.conv1d_1(x)))
            x = F.relu(self.bn1d_2(self.conv1d_2(x)))
            x = F.relu(self.bn1d_3(self.conv1d_3(x)))
            x = F.relu(self.bn1d_4(self.conv1d_4(x)))

            x = self.adaptive_pool(x)
            return x
        
        elif self.dataset_type == 'uni_shar' or self.dataset_type == 'opportunity':

            if len(x.shape) == 2:
                x = x.unsqueeze(1)

            x = F.relu(self.bn1d_1(self.conv1d_1(x)))
            x = F.relu(self.bn1d_2(self.conv1d_2(x)))
            x = F.relu(self.bn1d_3(self.conv1d_3(x)))
            x = F.relu(self.bn1d_4(self.conv1d_4(x)))

            x = self.adaptive_pool(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc(x))
            return x.view(x.size(0), -1, 1, 1)  # Reshape to match expected dimensions
        
        elif self.dataset_type == 'tabular':
            x = x.view(x.size(0), -1)

            # Handle batch size = 1 case for BatchNorm
            if x.size(0) == 1:
                # Skip BatchNorm when batch size is 1
                pass
            else:
                x = self.bn_input(x)

            masked_features = x
            step_outputs = []

            for step_idx in range(self.n_steps):
                x_embed = self.embed(masked_features)

                for block in self.shared_blocks:
                    x_embed = block(x_embed)

                out = self.decision_steps[step_idx](x_embed)
                step_outputs.append(out)

                if step_idx < self.n_steps - 1:
                    mask = self.attentive_blocks[step_idx](x_embed)
                    masked_features = mask * x

            agg = sum(step_outputs) / self.n_steps
            return agg.view(agg.size(0), -1, 1, 1)
        
        else:
            # MLP path for flat features
            x = x.view(x.size(0), -1)
            if x.size(0) > 1:
                x = F.relu(self.bn_fc1(self.fc1(x)))
                x = F.relu(self.bn_fc2(self.fc2(x)))
                x = F.relu(self.bn_fc3(self.fc3(x)))
            else:
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))

            return x.view(x.size(0), -1, 1, 1)  # Reshape to match expected dimensions


class TaskSpecificHead(nn.Module):

    def __init__(self, input_dim, output_dim, task_type='classification', dataset_type='time_series'):
        super(TaskSpecificHead, self).__init__()
        self.task_type = task_type
        self.dataset_type = dataset_type

        if dataset_type == 'tabular':
            # Even more enhanced architecture for tabular data
            self.fc1 = nn.Linear(input_dim, 256)
            self.bn1 = nn.BatchNorm1d(256, track_running_stats=True)
            self.dropout1 = nn.Dropout(0.3)

            self.fc2 = nn.Linear(256, 128)
            self.bn2 = nn.BatchNorm1d(128, track_running_stats=True)
            self.dropout2 = nn.Dropout(0.2)

            self.fc3 = nn.Linear(128, 64)
            self.bn3 = nn.BatchNorm1d(64, track_running_stats=True)
            self.dropout3 = nn.Dropout(0.1)

            self.fc4 = nn.Linear(64, output_dim)

            # Add residual connection if dimensions allow
            self.residual = nn.Linear(input_dim, 64) if input_dim != 64 else None
        else:
            # Standard architecture for other data types
            self.fc1 = nn.Linear(input_dim, 256)
            self.bn1 = nn.BatchNorm1d(256, track_running_stats=True)
            self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten

        if self.dataset_type == 'tabular':
            # Enhanced forward pass for tabular data
            original_x = x

            # Handle batch size = 1 case for BatchNorm
            if x.size(0) > 1:
                x = F.relu(self.bn1(self.fc1(x)))
            else:
                x = F.relu(self.fc1(x))
            x = self.dropout1(x)

            if x.size(0) > 1:
                x = F.relu(self.bn2(self.fc2(x)))
            else:
                x = F.relu(self.fc2(x))
            x = self.dropout2(x)

            if x.size(0) > 1:
                x = F.relu(self.bn3(self.fc3(x)))
            else:
                x = F.relu(self.fc3(x))
            x = self.dropout3(x)

            if self.residual is not None:
                residual = self.residual(original_x)
                x = x + residual

            x = self.fc4(x)
        else:
            if x.size(0) > 1:
                x = F.relu(self.bn1(self.fc1(x)))
            else:
                x = F.relu(self.fc1(x))
            x = self.fc2(x)

        if self.task_type == 'classification':
            return x  # Return logits directly
        elif self.task_type == 'regression':
            return x  # Regression returns values directly
        elif self.task_type == 'anomaly_detection':
            return torch.sigmoid(x)  # Anomaly detection returns probabilities
        else:
            return x


class MultiTaskModel(nn.Module):
    """Multi-task learning model with support for different dataset types"""

    def __init__(self, input_dim=1, feature_dim=512, tasks=None, dataset_type='time_series'):
        super(MultiTaskModel, self).__init__()
        self.dataset_type = dataset_type
        self.feature_extractor = FeatureExtractor(input_dim, feature_dim, dataset_type)

        # Default tasks if none provided
        if tasks is None:
            tasks = {
                'classification': {'output_dim': 10, 'type': 'classification'},
                'regression': {'output_dim': 1, 'type': 'regression'},
                'anomaly': {'output_dim': 1, 'type': 'anomaly_detection'}
            }

        self.tasks = tasks
        self.task_heads = nn.ModuleDict({
            task_name: TaskSpecificHead(
                self.feature_extractor.output_dim,
                task_config['output_dim'],
                task_config['type'],
                dataset_type
            ) for task_name, task_config in tasks.items()
        })

    def forward(self, x, task=None):
        features = self.feature_extractor(x)

        if task is not None:
            # If task specified, return only that task's results
            return self.task_heads[task](features)
        else:
            return {task_name: head(features) for task_name, head in self.task_heads.items()}

    def get_features(self, x):
        """Extract features for domain-specific and label-specific energy models"""
        return self.feature_extractor(x)


class LabelSpecificEnergyModel(nn.Module):

    def __init__(self, feature_dim, hidden_dim=256, num_classes=10):
        super(LabelSpecificEnergyModel, self).__init__()
        self.num_classes = num_classes

        self.energy_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Softplus()
            ) for _ in range(num_classes)
        ])

        self.class_prototypes = nn.Parameter(torch.randn(num_classes, feature_dim))

    def forward(self, x, label_idx=None):
        x = x.view(x.size(0), -1) 

        if label_idx is not None:
            return self.energy_networks[label_idx](x)
        else:
            # [batch_size, num_classes]
            return torch.cat([network(x) for network in self.energy_networks], dim=1)

    def get_label_energy_loss(self, x, labels):
        batch_size = x.size(0)
        all_energies = self.forward(x)  # [batch_size, num_classes]

        labels_one_hot = F.one_hot(labels, self.num_classes).float()

        margin_L1 = 1.0
        energy_loss = torch.mean(labels_one_hot * all_energies +
                                 (1 - labels_one_hot) * torch.clamp(margin_L1 - all_energies, min=0))

        prototype_loss = 0
        x_flat = x.view(batch_size, -1)

        margin_L2 = 0.8
        for i in range(batch_size):
            label = labels[i]
            distances = torch.sum((x_flat[i].unsqueeze(0) - self.class_prototypes) ** 2, dim=1)
            same_class_dist = distances[label]
            diff_class_dist = torch.mean(torch.cat([distances[:label], distances[label + 1:]]))
            prototype_loss += torch.clamp(same_class_dist - diff_class_dist + margin_L2, min=0)

        prototype_loss = prototype_loss / batch_size

        return energy_loss + 0.1 * prototype_loss


class DomainSpecificEnergyModel(nn.Module):

    def __init__(self, feature_dim, hidden_dim=256, num_domains=3):
        super(DomainSpecificEnergyModel, self).__init__()
        self.num_domains = num_domains

        self.energy_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Softplus()
            ) for _ in range(num_domains)
        ])

    def forward(self, x, domain_idx=None):
        x = x.view(x.size(0), -1)

        if domain_idx is not None:
            return self.energy_networks[domain_idx](x)
        else:
            return [network(x) for network in self.energy_networks]

    def get_domain_energy_loss(self, x, domain_labels):
        batch_size = x.size(0)
        all_energies = torch.zeros(batch_size, self.num_domains).to(x.device)

        for d in range(self.num_domains):
            all_energies[:, d] = self.energy_networks[d](x.view(batch_size, -1)).squeeze()

        domain_labels_one_hot = F.one_hot(domain_labels, self.num_domains).float()

        margin = 0.3
        energy_loss = torch.mean(domain_labels_one_hot * all_energies +
                                 (1 - domain_labels_one_hot) * torch.clamp(margin - all_energies, min=0))

        return energy_loss


class MultiDomainAdapter(nn.Module):
    def __init__(self, feature_dim, hidden_dim=256, num_domains=3):
        super(MultiDomainAdapter, self).__init__()
        self.num_domains = num_domains

        # 每个域的适配器
        self.domain_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, feature_dim)
            ) for _ in range(num_domains)
        ])

    def forward(self, x, domain_idx=None, domain_weights=None):
        x_flat = x.view(x.size(0), -1)

        if domain_idx is not None:
            adapted_features = self.domain_adapters[domain_idx](x_flat)
        elif domain_weights is not None:
            adapted_features = torch.zeros_like(x_flat)
            for d in range(self.num_domains):
                adapted_features += domain_weights[:, d].unsqueeze(1) * self.domain_adapters[d](x_flat)
        else:
            adapted_features = torch.stack([adapter(x_flat) for adapter in self.domain_adapters])
            adapted_features = torch.mean(adapted_features, dim=0)

        return adapted_features.view_as(x)


class GaussianNoiseLayer(nn.Module):

    def __init__(self, sigma=0.1):
        super(GaussianNoiseLayer, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.sigma
            return x + noise
        else:
            return x


class VIB(nn.Module):

    def __init__(self, input_dim, bottleneck_dim=256):
        super(VIB, self).__init__()
        self.bottleneck_dim = bottleneck_dim

        self.encoder_mean = nn.Linear(input_dim, bottleneck_dim)
        self.encoder_logvar = nn.Linear(input_dim, bottleneck_dim)

        self.decoder = nn.Linear(bottleneck_dim, input_dim)

    def encode(self, x):
        mean = self.encoder_mean(x)
        logvar = self.encoder_logvar(x)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        mean, logvar = self.encode(x_flat)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decode(z)

        return z, mean, logvar, x_recon.view_as(x)

    def get_bottleneck_loss(self, mean, logvar, beta=1e-4):
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return beta * kl_loss


class InformationBottleneckModule(nn.Module):

    def __init__(self, feature_dim, bottleneck_dim=256, num_domains=3):
        super(InformationBottleneckModule, self).__init__()
        self.feature_dim = feature_dim
        self.num_domains = num_domains

        self.vib = VIB(feature_dim, bottleneck_dim)

        self.domain_classifier = nn.Sequential(
            nn.Linear(bottleneck_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_domains)
        )

    def forward(self, x):
        bottleneck_features, mean, logvar, recon_x = self.vib(x)
        return bottleneck_features, mean, logvar, recon_x

    def get_losses(self, x, target_features=None, domain_labels=None):
        bottleneck_features, mean, logvar, recon_x = self.forward(x)

        # 计算信息瓶颈损失
        bottleneck_loss = self.vib.get_bottleneck_loss(mean, logvar)

        # 计算重构损失
        if target_features is not None:
            recon_loss = F.mse_loss(recon_x, target_features)
        else:
            recon_loss = F.mse_loss(recon_x, x)

        # 计算域对抗损失
        adv_loss = None
        if domain_labels is not None:
            domain_logits = self.domain_classifier(bottleneck_features.view(bottleneck_features.size(0), -1))
            adv_loss = F.cross_entropy(domain_logits, domain_labels)

        return bottleneck_features, bottleneck_loss, recon_loss, adv_loss

    def domain_confusion_loss(self, features):
        domain_logits = self.domain_classifier(features.view(features.size(0), -1))
        # 均匀分布的域标签
        uniform_domain_prob = torch.ones_like(domain_logits) / self.num_domains
        # 使预测趋向于均匀分布
        confusion_loss = F.kl_div(
            F.log_softmax(domain_logits, dim=1),
            uniform_domain_prob,
            reduction='batchmean'
        )
        return confusion_loss


class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_domains=3):
        super(DomainDiscriminator, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.layer3 = nn.Linear(hidden_dim // 2, num_domains)

    def forward(self, x, alpha=1.0):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class VAT(nn.Module):
    def __init__(self, xi=10.0, eps=1.0, ip=1):
        super(VAT, self).__init__()
        self.xi = xi  
        self.eps = eps 
        self.ip = ip 

    def forward(self, model, x, task='classification'):
        """Return model prediction"""
        return model(x, task=task)

    def generate_virtual_adversarial_perturbation(self, model, x, logits, task='classification'):
        d = torch.randn_like(x)
        d = F.normalize(d.view(d.size(0), -1), p=2, dim=1).view_as(x)

        for _ in range(self.ip):
            d.requires_grad_(True)
            pred_hat = model(x + self.xi * d, task=task)
            kl_div = F.kl_div(
                F.log_softmax(pred_hat, dim=1),
                F.softmax(logits.detach(), dim=1),
                reduction='batchmean'
            )
            kl_div.backward()

            d = d.grad.detach()
            d = F.normalize(d.view(d.size(0), -1), p=2, dim=1).view_as(x)

        return self.eps * d

    def virtual_adversarial_loss(self, model, x, task='classification'):
        with torch.no_grad():
            logits = model(x, task=task)

        r_vadv = self.generate_virtual_adversarial_perturbation(model, x, logits, task)
        logits_vadv = model(x + r_vadv, task=task)

        loss = F.kl_div(
            F.log_softmax(logits_vadv, dim=1),
            F.softmax(logits.detach(), dim=1),
            reduction='batchmean'
        )

        return loss


class ERIS(nn.Module):
    def __init__(self, input_dim=1, feature_dim=512, bottleneck_dim=256,
                 num_classes=10, num_domains=3, tasks=None, dataset_type='time_series'):
        super(ERIS, self).__init__()
        self.multi_task_model = MultiTaskModel(input_dim, feature_dim, tasks, dataset_type)

        feature_flat_dim = self.multi_task_model.feature_extractor.output_dim

        # Domain specific energy model
        self.domain_energy_model = DomainSpecificEnergyModel(feature_flat_dim, hidden_dim=256, num_domains=num_domains)
        self.domain_adapter = MultiDomainAdapter(feature_flat_dim, hidden_dim=256, num_domains=num_domains)

        # Label specific energy model
        self.label_energy_model = LabelSpecificEnergyModel(feature_flat_dim, hidden_dim=256, num_classes=num_classes)

        # Label classifier that works with the bottleneck dimensions
        self.label_classifier = nn.Linear(bottleneck_dim, num_classes)

        self.info_bottleneck = InformationBottleneckModule(feature_flat_dim, bottleneck_dim=bottleneck_dim,
                                                           num_domains=num_domains)

        self.domain_discriminator = DomainDiscriminator(bottleneck_dim, hidden_dim=256, num_domains=num_domains)

        # Virtual adversarial training module
        self.vat = VAT(xi=10.0, eps=1.0, ip=1)

        self.dataset_type = dataset_type  # 保存在 ERIS 中用于后续判断

    def forward(self, x, task=None, domain_idx=None):
        features = self.multi_task_model.get_features(x)
        features_flat = features.view(features.size(0), -1)

        if domain_idx is not None:
            adapted_features = self.domain_adapter(features, domain_idx=domain_idx)
        else:
            adapted_features = features

        bottleneck_features, mean, logvar, _ = self.info_bottleneck(adapted_features)
        bottleneck_features_flat = bottleneck_features.view(bottleneck_features.size(0), -1)

        if task == 'classification':
            return self.label_classifier(bottleneck_features_flat)
        elif task is not None:
            return self.multi_task_model(bottleneck_features, task=task)
        else:
            results = self.multi_task_model(bottleneck_features)
            results['label_energy_classification'] = self.label_classifier(bottleneck_features_flat)
            return results

    def get_features(self, x):
        """Extract features"""
        return self.multi_task_model.get_features(x)

    def focal_loss(self, logits, targets, gamma=2.0):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        pt = torch.clamp(pt, min=1e-4, max=1.0)

        # dynamic alpha: 0.5 for abnormal (1), 0.25 for normal (0)
        alpha = torch.where(targets == 1, 0.5, 0.25).to(logits.device)

        focal = alpha * (1 - pt) ** gamma * ce_loss
        return focal.mean()

    def get_bottleneck_features(self, x, domain_idx=None):
        features = self.get_features(x)

        if domain_idx is not None:
            adapted_features = self.domain_adapter(features, domain_idx=domain_idx)
        else:
            adapted_features = features

        bottleneck_features, _, _, _ = self.info_bottleneck(adapted_features)

        bottleneck_features = torch.nan_to_num(bottleneck_features)
        bottleneck_features = torch.clamp(bottleneck_features, min=-1e4, max=1e4)

        return bottleneck_features

    def compute_all_losses(self, x, y, domain_labels, alpha=1.0):
        features = self.get_features(x)
        features_flat = features.view(features.size(0), -1)

        orth_loss = torch.zeros(1, device=x.device, requires_grad=True)
        for d_net in self.domain_energy_model.energy_networks:
            Wd = d_net[0].weight
            for l_net in self.label_energy_model.energy_networks:
                Wl = l_net[0].weight
                orth_loss = orth_loss + torch.norm(Wd.T @ Wl, p='fro') ** 2
        orth_loss = orth_loss.squeeze()

        # Domain Energy Loss
        domain_energy_loss = self.domain_energy_model.get_domain_energy_loss(features_flat, domain_labels)

        # Label Energy Loss
        label_energy_loss = self.label_energy_model.get_label_energy_loss(features_flat, y)

        bottleneck_features, bottleneck_loss, recon_loss, _ = self.info_bottleneck.get_losses(features,
                                                                                              domain_labels=domain_labels)
        bottleneck_features_flat = bottleneck_features.view(bottleneck_features.size(0), -1)

        domain_outputs = self.domain_discriminator(bottleneck_features_flat, alpha)
        domain_adv_loss = F.cross_entropy(domain_outputs, domain_labels)

        domain_confusion_loss = self.info_bottleneck.domain_confusion_loss(bottleneck_features_flat)

        cls_outputs = self.label_classifier(bottleneck_features_flat)
        classification_loss = F.cross_entropy(cls_outputs, y)

        vat_loss = self.vat.virtual_adversarial_loss(self, x, task='classification')

        # Combine all regular losses
        if self.dataset_type == 'tabular':
            regular_loss = classification_loss + 0.1 * vat_loss
        else:
            regular_loss = classification_loss + 0.1 * vat_loss + 0.1 * domain_adv_loss + 0.1 * domain_confusion_loss

        losses = {
            'orthogonality_loss': orth_loss,
            'domain_energy_loss': domain_energy_loss,
            'label_energy_loss': label_energy_loss,
            'regular_loss': regular_loss
        }

        return losses

    def compute_total_loss(self, losses, weights=None):
        if weights is None:
            weights = {
                'orthogonality_loss': 1,
                'domain_energy_loss': 0.9,
                'label_energy_loss': 2,
                'regular_loss': 1.0
            }

        total_loss = None
        for k in weights.keys():
            if k in losses:
                weighted_loss = weights[k] * losses[k]
                if total_loss is None:
                    total_loss = weighted_loss
                else:
                    total_loss = total_loss + weighted_loss

        return total_loss if total_loss is not None else torch.tensor(0.0, device=next(iter(losses.values())).device)