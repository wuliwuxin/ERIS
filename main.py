import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import logging
import json
import glob
import re

from models import ERIS
from data_load import create_data_loaders


def eval_ece(pred_scores_np, pred_np, label_np, num_bins=15):
    acc_tab = np.zeros(num_bins) 
    mean_conf = np.zeros(num_bins)
    nb_items_bin = np.zeros(num_bins)
    tau_tab = np.linspace(0, 1, num_bins + 1)

    for i in np.arange(num_bins):
        sec = (tau_tab[i + 1] > pred_scores_np) & (pred_scores_np >= tau_tab[i])
        nb_items_bin[i] = np.sum(sec)
        class_pred_sec, y_sec = pred_np[sec], label_np[sec]
        mean_conf[i] = np.mean(
            pred_scores_np[sec]) if nb_items_bin[i] > 0 else np.nan
        acc_tab[i] = np.mean(
            class_pred_sec == y_sec) if nb_items_bin[i] > 0 else np.nan

    # Cleaning
    mean_conf = mean_conf[nb_items_bin > 0]
    acc_tab = acc_tab[nb_items_bin > 0]
    nb_items_bin = nb_items_bin[nb_items_bin > 0]

    if sum(nb_items_bin) != 0:
        ece = np.average(
            np.absolute(mean_conf - acc_tab),
            weights=nb_items_bin.astype(float) / np.sum(nb_items_bin))
    else:
        ece = 0.0
    return ece


# Training function
def train_epoch(model, train_loader, optimizer, device, epoch, weights=None):

    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # Dynamically adjust adversarial strength
    p = float(epoch) / 100
    alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1

    progress_bar = tqdm(train_loader)
    for batch_idx, (data, target, domain_labels) in enumerate(progress_bar):
        data, target, domain_labels = data.to(device), target.to(device), domain_labels.to(device)

        optimizer.zero_grad()

        losses = model.compute_all_losses(data, target, domain_labels, alpha)

        # Compute total loss using weights
        if weights is None:
            weights = {
                'orthogonality_loss': 1.0,
                'domain_energy_loss': 0.9,
                'label_energy_loss': 2.0,
                'regular_loss': 1.0
            }

        loss = model.compute_total_loss(losses, weights)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_loss += loss.item()

        outputs = model(data, task='classification')
        _, predicted = outputs.max(1)
        correct += predicted.eq(target).sum().item()
        total += target.size(0)

        progress_bar.set_description(
            f'Epoch: {epoch} | Loss: {loss.item():.4f} | '
            f'Orth: {losses["orthogonality_loss"].item():.3f} | '
            f'Dom: {losses["domain_energy_loss"].item():.3f} | '
            f'Lab: {losses["label_energy_loss"].item():.3f} | '
            f'Reg: {losses["regular_loss"].item():.3f} | '
            f'Acc: {100. * correct / total:.2f}%'
        )

    avg_loss = total_loss / len(train_loader)
    avg_acc = 100. * correct / total
    print(f'Epoch {epoch}: Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_acc:.2f}%')

    return avg_loss, avg_acc


# Test function
def test(model, test_loader, device, return_detailed=False):
    model.eval()
    correct = 0
    total = 0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing")
        for data, target, _ in progress_bar:
            data, target = data.to(device), target.to(device)

            # Forward pass
            outputs = model(data, task='classification')
            _, predicted = outputs.max(1)

            correct += predicted.eq(target).sum().item()
            total += target.size(0)

            if return_detailed:
                all_outputs.append(outputs.cpu())
                all_targets.append(target.cpu())

            # Update progress bar
            progress_bar.set_postfix(acc=f'{100. * correct / total:.2f}%')

    accuracy = 100. * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

    if return_detailed:
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        return accuracy, all_outputs, all_targets
    else:
        return accuracy


# Test function for each domain
def test_on_domain(model, test_loader, domain_idx, device, return_detailed=False):
    model.eval()
    correct = 0
    total = 0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for data, target, domains in test_loader:
            # Select samples from specified domain
            domain_mask = (domains == domain_idx)
            if not domain_mask.any():
                continue

            data, target = data[domain_mask].to(device), target[domain_mask].to(device)

            # Forward pass using domain-specific adaptation
            outputs = model(data, task='classification', domain_idx=domain_idx)
            _, predicted = outputs.max(1)

            correct += predicted.eq(target).sum().item()
            total += target.size(0)

            if return_detailed:
                all_outputs.append(outputs.cpu())
                all_targets.append(target.cpu())

    if total == 0:
        if return_detailed:
            return 0.0, None, None
        return 0.0

    accuracy = 100. * correct / total
    print(f'Domain {domain_idx} Test Accuracy: {accuracy:.2f}%')

    if return_detailed:
        if all_outputs:
            all_outputs = torch.cat(all_outputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            return accuracy, all_outputs, all_targets
        else:
            return accuracy, None, None
    else:
        return accuracy


# Setup logging
def setup_logger(save_dir, dataset_name, target_domain):
    """Set up logger for training process"""
    log_file = os.path.join(save_dir, f'Ours_{dataset_name}_target_domain_{target_domain}.log')

    logger = logging.getLogger('training_logger')
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_dataset_type(dataset_name):
    if dataset_name == 'EMG':
        return 'time_series'
    elif dataset_name == 'UCIHAR':
        return 'time_series'
    elif dataset_name == 'Uni':
        return 'uni_shar'
    elif dataset_name == 'Oppo':
        return 'opportunity'
    else:
        return 'time_series'  # Default


def visualize_features(model, test_loader, num_samples=100, save_dir='./visualizations',
                       dataset_name='dataset', target_domain='0'):
    os.makedirs(save_dir, exist_ok=True)
    device = next(model.parameters()).device

    features = []
    labels = []
    domains = []

    model.eval()
    count = 0
    with torch.no_grad():
        for data, target, domain_label in test_loader:
            batch_size = data.size(0)
            if count + batch_size > num_samples:
                needed = num_samples - count
                data = data[:needed]
                target = target[:needed]
                domain_label = domain_label[:needed]
                if needed <= 0:
                    break

            count += data.size(0)
            data = data.to(device)

            bottleneck_features = model.get_bottleneck_features(data)

            bottleneck_features = bottleneck_features.cpu().numpy()

            features.append(bottleneck_features)
            labels.append(target.cpu().numpy())
            domains.append(domain_label.cpu().numpy())

            if count >= num_samples:
                break

    features = np.vstack([f.reshape(f.shape[0], -1) for f in features])
    labels = np.concatenate(labels)
    domains = np.concatenate(domains)

    if len(domains.shape) > 1 and domains.shape[1] == 1:
        domains = domains.flatten()

    features_save_path = os.path.join(save_dir, f'Ours_{dataset_name}_features_target_domain_{target_domain}.npy')
    labels_save_path = os.path.join(save_dir, f'Ours_{dataset_name}_labels_target_domain_{target_domain}.npy')
    np.save(features_save_path, features)
    np.save(labels_save_path, labels)

    tsne = TSNE(n_components=2, random_state=42)
    try:
        features_2d = tsne.fit_transform(features)

        plt.figure(figsize=(12, 10))

        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)

        if num_classes <= 10:
            cmap = plt.cm.tab10
        elif num_classes <= 20:
            cmap = plt.cm.tab20
        else:
            cmap = plt.cm.nipy_spectral

        for i, class_idx in enumerate(unique_labels):
            mask = labels == class_idx
            if np.any(mask):
                color = cmap(i / num_classes)
                plt.scatter(
                    features_2d[mask, 0], features_2d[mask, 1],
                    color=color, marker='o', alpha=0.7,
                    label=f'Class {int(class_idx)}'
                )

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f'Feature Visualization by Class - {dataset_name} (Target: {target_domain})')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'Ours_{dataset_name}_class_visualization_target_{target_domain}.png'))
        plt.close()

    except Exception as e:
        print(f"Error during t-SNE visualization: {e}")
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features)

            plt.figure(figsize=(12, 10))
            unique_labels = np.unique(labels)
            num_classes = len(unique_labels)

            if num_classes <= 10:
                cmap = plt.cm.tab10
            elif num_classes <= 20:
                cmap = plt.cm.tab20
            else:
                cmap = plt.cm.nipy_spectral

            for i, class_idx in enumerate(unique_labels):
                mask = labels == class_idx
                if np.any(mask):
                    color = cmap(i / num_classes)
                    plt.scatter(
                        features_2d[mask, 0], features_2d[mask, 1],
                        color=color, marker='o', alpha=0.7,
                        label=f'Class {int(class_idx)}'
                    )

            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.title(f'PCA Visualization by Class - {dataset_name} (Target: {target_domain})')
            plt.xlabel('PC 1')
            plt.ylabel('PC 2')
            plt.tight_layout()
            plt.savefig(
                os.path.join(save_dir, f'Ours_{dataset_name}_class_visualization_pca_target_{target_domain}.png'))
            plt.close()

        except Exception as e:
            print(f"PCA visualization also failed: {e}")


def evaluate_model(model, test_loader, device, num_classes):

    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for data, target, _ in test_loader:
            data, target = data.to(device), target.to(device)

            # Forward pass
            outputs = model(data, task='classification')
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    max_probs = np.max(all_probs, axis=1)

    accuracy = np.mean(all_preds == all_targets)

    f1_micro = f1_score(all_targets, all_preds, average='micro', zero_division=1)
    f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=1)
    f1_weighted = f1_score(all_targets, all_preds, average='weighted', zero_division=1)

    precision_macro = precision_score(all_targets, all_preds, average='macro', zero_division=1)
    precision_weighted = precision_score(all_targets, all_preds, average='weighted', zero_division=1)

    recall_macro = recall_score(all_targets, all_preds, average='macro', zero_division=1)
    recall_weighted = recall_score(all_targets, all_preds, average='weighted', zero_division=1)

    ece = eval_ece(max_probs, all_preds, all_targets)

    conf_matrix = confusion_matrix(all_targets, all_preds)

    metrics = {
        'accuracy': accuracy,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'ece': ece,
        'confusion_matrix': conf_matrix.tolist()
    }

    return metrics


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def clean_old_model_weights(save_dir, dataset_name, target_domain):

    model_pattern = os.path.join(save_dir, f'Ours_{dataset_name}_target_domain_{target_domain}_*.pth')
    model_files = glob.glob(model_pattern)

    if not model_files:
        return

    accuracies = {}

    pattern = re.compile(r'.*_acc([\d\.]+)(?:_ece[\d\.]+)?\.pth$|.*_([\d\.]+)\.pth$')

    for file_path in model_files:
        match = pattern.match(file_path)
        if match:
            try:
                acc_str = match.group(1) if match.group(1) else match.group(2)
                if acc_str:
                    acc = float(acc_str)
                    accuracies[file_path] = acc
            except (ValueError, TypeError):
                # If filename doesn't contain valid accuracy, skip it
                continue

    if not accuracies:
        return  # No valid files found

    best_model = max(accuracies.items(), key=lambda x: x[1])[0]

    for file_path in model_files:
        if file_path != best_model:
            try:
                os.remove(file_path)
                print(f"Removed old model: {os.path.basename(file_path)}")
            except OSError as e:
                print(f"Error removing file {file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description='ERIS Model for Time Series Datasets')
    parser.add_argument('--dataset', type=str, default='UCIHAR', choices=['EMG', 'UCIHAR', 'Uni', 'Oppo', 'Boiler'])
    parser.add_argument('--target-domain', type=str, default='0',
                        help='target domain for testing (format depends on dataset)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--data-dir', type=str, default='./data/',
                        help='data directory')
    parser.add_argument('--save-dir', type=str, default='./results/',
                        help='model save directory')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'visualize'],
                        help='operation mode: train, test or visualize')
    parser.add_argument('--model-path', type=str, default=None,
                        help='path to saved model')
    parser.add_argument('--domain', type=int, default=None,
                        help='domain index for testing')
    parser.add_argument('--keep-all-weights', action='store_true',
                        help='keep all model weights instead of just the best one')

    # Loss weight arguments for the 4 main loss categories
    parser.add_argument('--orth-weight', type=float, default=1.0,
                        help='weight for orthogonality loss')
    parser.add_argument('--domain-energy-weight', type=float, default=0.9,
                        help='weight for domain energy loss')
    parser.add_argument('--label-energy-weight', type=float, default=2.0,
                        help='weight for label energy loss')
    parser.add_argument('--regular-weight', type=float, default=1.0,
                        help='weight for regular loss')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Create models subdirectory for organized structure
    models_dir = os.path.join(args.save_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Setup logger
    logger = setup_logger(args.save_dir, args.dataset, args.target_domain)
    logger.info(f"Starting run with arguments: {args}")
    logger.info(f"Using device: {device}")

    # Get dataset information
    logger.info(f"Loading {args.dataset} dataset with target domain {args.target_domain}")
    train_loader, test_loader, num_classes, num_domains, input_dim = create_data_loaders(
        args.dataset, args.data_dir, args.target_domain, args.batch_size, device
    )

    logger.info(f"Dataset info: {num_classes} classes, {num_domains} domains, {input_dim} input dimensions")

    tasks = {
        'classification': {'output_dim': num_classes, 'type': 'classification'},
    }

    dataset_type = get_dataset_type(args.dataset)

    if dataset_type == 'tabular':
        logger.info("Boiler tabular data detected: simplified loss structure will be used.")

    loss_weights = {
        'orthogonality_loss': args.orth_weight,
        'domain_energy_loss': args.domain_energy_weight,
        'label_energy_loss': args.label_energy_weight,
        'regular_loss': args.regular_weight
    }

    logger.info(f"Loss weights: {loss_weights}")

    if args.mode == 'train':
        # Create model
        model = ERIS(
            input_dim=input_dim,
            feature_dim=512,
            bottleneck_dim=256,
            num_classes=num_classes,
            num_domains=num_domains,
            tasks=tasks,
            dataset_type=dataset_type
        )
        model = model.to(device)

        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

        # Create learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        # Training loop
        history = {
            'train_loss': [],
            'train_acc': [],
            'test_acc': [],
            'ece': [],
            'metrics': [],
            'loss_components': []
        }

        best_accuracy = 0
        best_ece = float('inf')
        best_model_path = None

        for epoch in range(1, args.epochs + 1):
            p = float(epoch) / args.epochs

            current_weights = loss_weights.copy()
            if dataset_type != 'tabular':
                # Decrease domain energy loss weight over time
                current_weights['domain_energy_loss'] = args.domain_energy_weight * (1 - p * 0.5)
                # Increase orthogonality loss weight over time
                current_weights['orthogonality_loss'] = args.orth_weight * (1 + p)

            # Train
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, epoch, current_weights)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            metrics = evaluate_model(model, test_loader, device, num_classes)
            history['test_acc'].append(metrics['accuracy'])
            history['ece'].append(metrics['ece'])
            history['metrics'].append(metrics)

            logger.info(f"Epoch {epoch:3d} - "
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:6.2f}%, "
                        f"Test Acc: {metrics['accuracy']:6.4f}, ECE: {metrics['ece']:.4f}, "
                        f"F1 Micro: {metrics['f1_micro']:.4f}, F1 Macro: {metrics['f1_macro']:.4f}")

            # Update learning rate
            scheduler.step()

            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                best_ece = metrics['ece']

                accuracy_str = f"{best_accuracy:.4f}"
                ece_str = f"{best_ece:.4f}"

                new_model_path = os.path.join(
                    models_dir,
                    f'Ours_{args.dataset}_target_domain_{args.target_domain}_acc{accuracy_str}_ece{ece_str}.pth'
                )

                if best_model_path and os.path.exists(best_model_path) and not args.keep_all_weights:
                    try:
                        os.remove(best_model_path)
                        logger.info(f"Removed previous best model: {os.path.basename(best_model_path)}")
                    except OSError as e:
                        logger.error(f"Error removing previous best model {best_model_path}: {e}")

                # Save the new best model
                torch.save(model.state_dict(), new_model_path)
                best_model_path = new_model_path
                logger.info(f"New best model saved with accuracy: {best_accuracy:.4f}, ECE: {best_ece:.4f}")

            # Save checkpoint
            if epoch % 10 == 0 or epoch == args.epochs:
                visualize_features(model, test_loader, num_samples=500,
                                   save_dir=args.save_dir,
                                   dataset_name=args.dataset,
                                   target_domain=args.target_domain)

                history_path = os.path.join(args.save_dir,
                                            f'Ours_{args.dataset}_history_target_domain_{args.target_domain}.json')
                with open(history_path, 'w') as f:
                    json.dump(history, f, cls=NumpyEncoder)

                logger.info(f"Checkpoint saved at epoch {epoch}")

        logger.info(f"Training completed. Best accuracy: {best_accuracy:.4f}, Best ECE: {best_ece:.4f}")

        if args.keep_all_weights:
            clean_old_model_weights(models_dir, args.dataset, args.target_domain)
            logger.info("Cleaned up old model weights, keeping only the best model")

        plt.figure(figsize=(25, 5))

        # Plot training loss
        plt.subplot(1, 5, 1)
        plt.plot(history['train_loss'])
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        # Plot training and test accuracy
        plt.subplot(1, 5, 2)
        plt.plot(history['train_acc'], label='Train')
        plt.plot(history['test_acc'], label='Test')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot ECE
        plt.subplot(1, 5, 3)
        plt.plot(history['ece'])
        plt.title('Expected Calibration Error (ECE)')
        plt.xlabel('Epoch')
        plt.ylabel('ECE')

        # Plot F1 scores
        plt.subplot(1, 5, 4)
        f1_micros = [m['f1_micro'] for m in history['metrics']]
        f1_macros = [m['f1_macro'] for m in history['metrics']]
        f1_weighteds = [m['f1_weighted'] for m in history['metrics']]
        plt.plot(f1_micros, label='F1 Micro')
        plt.plot(f1_macros, label='F1 Macro')
        plt.plot(f1_weighteds, label='F1 Weighted')
        plt.title('F1 Scores')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()

        # Plot Precision and Recall
        plt.subplot(1, 5, 5)
        precisions = [m['precision_weighted'] for m in history['metrics']]
        recalls = [m['recall_weighted'] for m in history['metrics']]
        plt.plot(precisions, label='Precision')
        plt.plot(recalls, label='Recall')
        plt.title('Precision and Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()

        plt.tight_layout()
        plt.savefig(
            os.path.join(args.save_dir, f'Ours_{args.dataset}_training_history_target_{args.target_domain}.png'))
        plt.close()

        # Visualize the final confusion matrix
        final_metrics = history['metrics'][-1]
        conf_matrix = np.array(final_metrics['confusion_matrix'])

        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {args.dataset} (Target: {args.target_domain})')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(
            os.path.join(args.save_dir, f'Ours_{args.dataset}_confusion_matrix_target_{args.target_domain}.png'))
        plt.close()

        # Visualize features for the final model
        visualize_features(model, test_loader, num_samples=500,
                           save_dir=args.save_dir,
                           dataset_name=args.dataset,
                           target_domain=args.target_domain)

    elif args.mode == 'test':
        if args.model_path is None:
            # Make sure we only have the best model
            clean_old_model_weights(models_dir, args.dataset, args.target_domain)

            # Find the best model if none specified
            model_files = [f for f in os.listdir(models_dir)
                           if f.startswith(f'Ours_{args.dataset}_target_domain_{args.target_domain}')
                           and f.endswith('.pth')]

            if not model_files:
                logger.error("No model found for testing")
                return

            # Sort by accuracy in the filename
            args.model_path = os.path.join(models_dir, sorted(model_files)[-1])
            logger.info(f"Using best model found: {args.model_path}")

        # Create model
        model = ERIS(
            input_dim=input_dim,
            feature_dim=512,
            bottleneck_dim=256,
            num_classes=num_classes,
            num_domains=num_domains,
            tasks=tasks,
            dataset_type=dataset_type
        )

        # Load model weights
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
        model = model.to(device)

        # Test on specific domain if specified
        if args.domain is not None:
            domain_idx = int(args.domain)

            # Get detailed predictions for ECE calculation
            accuracy, outputs, targets = test_on_domain(model, test_loader, domain_idx, device, return_detailed=True)

            if outputs is not None and targets is not None:
                # Calculate ECE for domain-specific test
                probs = F.softmax(outputs, dim=1)
                max_probs = torch.max(probs, dim=1)[0].numpy()
                preds = torch.max(outputs, dim=1)[1].numpy()
                targets_np = targets.numpy()

                domain_ece = eval_ece(max_probs, preds, targets_np)
                logger.info(f"Domain {domain_idx} Test Accuracy: {accuracy:.4f}, ECE: {domain_ece:.4f}")
            else:
                logger.info(f"Domain {domain_idx} Test Accuracy: {accuracy:.4f}, ECE: N/A (no samples)")
            return

        # Evaluate model
        metrics = evaluate_model(model, test_loader, device, num_classes)

        # Log and print the results with enhanced ECE info
        logger.info(f"Test Evaluation Results:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"ECE: {metrics['ece']:.4f}")
        logger.info(f"F1 Micro: {metrics['f1_micro']:.4f}")
        logger.info(f"F1 Macro: {metrics['f1_macro']:.4f}")
        logger.info(f"F1 Weighted: {metrics['f1_weighted']:.4f}")
        logger.info(f"Precision Macro: {metrics['precision_macro']:.4f}")
        logger.info(f"Precision Weighted: {metrics['precision_weighted']:.4f}")
        logger.info(f"Recall Macro: {metrics['recall_macro']:.4f}")
        logger.info(f"Recall Weighted: {metrics['recall_weighted']:.4f}")

        # Save test results to the test_results subdirectory
        test_results_dir = os.path.join(args.save_dir, 'test_results')
        os.makedirs(test_results_dir, exist_ok=True)

        test_results_path = os.path.join(test_results_dir,
                                         f'Ours_{args.dataset}_test_results_target_domain_{args.target_domain}.json')
        with open(test_results_path, 'w') as f:
            json.dump(metrics, f, cls=NumpyEncoder)

        # Visualize confusion matrix
        conf_matrix = np.array(metrics['confusion_matrix'])
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {args.dataset} (Target: {args.target_domain})')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(
            os.path.join(args.save_dir, f'Ours_{args.dataset}_confusion_matrix_target_{args.target_domain}.png'))
        plt.close()

    elif args.mode == 'visualize':
        # Make sure we only have the best model
        clean_old_model_weights(models_dir, args.dataset, args.target_domain)

        if args.model_path is None:
            # Find the best model if none specified
            model_files = [f for f in os.listdir(models_dir)
                           if f.startswith(f'Ours_{args.dataset}_target_domain_{args.target_domain}')
                           and f.endswith('.pth')]

            if not model_files:
                logger.error("No model found for visualization")
                return

            # Sort by accuracy in the filename
            args.model_path = os.path.join(models_dir, sorted(model_files)[-1])
            logger.info(f"Using best model found: {args.model_path}")

        # Create model
        model = ERIS(
            input_dim=input_dim,
            feature_dim=512,
            bottleneck_dim=256,
            num_classes=num_classes,
            num_domains=num_domains,
            tasks=tasks,
            dataset_type=dataset_type
        )

        # Load model weights
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
        model = model.to(device)

        # Visualize features
        visualize_features(model, test_loader, num_samples=500,
                           save_dir=args.save_dir,
                           dataset_name=args.dataset,
                           target_domain=args.target_domain)

        logger.info("Feature visualization completed")


if __name__ == '__main__':
    main()