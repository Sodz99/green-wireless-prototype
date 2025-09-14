import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy


def compute_l1_importance(model):
    """
    Compute L1 importance scores for all parameters in the model.
    Importance = |weight| (L1 norm of each parameter)
    """
    importance_scores = {}

    for name, param in model.named_parameters():
        if param.requires_grad and len(param.shape) > 1:  # Only prune conv/linear layers
            # L1 importance = absolute value of weights
            importance_scores[name] = torch.abs(param.data).cpu().numpy()

    return importance_scores


def get_global_pruning_threshold(importance_scores, sparsity_ratio):
    """
    Compute global threshold for pruning based on desired sparsity ratio.

    Args:
        importance_scores: dict of parameter importance scores
        sparsity_ratio: fraction of weights to prune (0.0 - 1.0)

    Returns:
        threshold: value below which weights will be pruned
    """
    # Flatten all importance scores
    all_scores = []
    for scores in importance_scores.values():
        all_scores.extend(scores.flatten())

    all_scores = np.array(all_scores)
    all_scores = np.sort(all_scores)

    # Find threshold at the sparsity percentile
    threshold_idx = int(sparsity_ratio * len(all_scores))
    threshold = all_scores[threshold_idx]

    print(f"Global pruning threshold: {threshold:.6f}")
    print(f"Total parameters: {len(all_scores):,}")
    print(f"Parameters to prune: {threshold_idx:,} ({sparsity_ratio*100:.1f}%)")

    return threshold


def apply_l1_pruning(model, sparsity_ratio=0.5):
    """
    Apply L1 magnitude-based pruning to the model.

    Args:
        model: PyTorch model to prune
        sparsity_ratio: fraction of weights to prune (0.0 - 1.0)

    Returns:
        pruned_model: model with weights set to zero (structured pruning mask applied)
        pruning_info: dictionary with pruning statistics
    """
    # Create a copy to avoid modifying original
    pruned_model = deepcopy(model)

    # Compute importance scores
    importance_scores = compute_l1_importance(pruned_model)

    # Get global pruning threshold
    threshold = get_global_pruning_threshold(importance_scores, sparsity_ratio)

    # Apply pruning (set weights below threshold to zero)
    pruning_info = {}
    total_params = 0
    pruned_params = 0

    for name, param in pruned_model.named_parameters():
        if name in importance_scores:
            original_shape = param.shape
            original_count = param.numel()

            # Create mask: keep weights above threshold
            mask = torch.abs(param.data) > threshold

            # Apply mask (set pruned weights to zero)
            param.data = param.data * mask.float()

            # Count pruned parameters
            remaining_count = mask.sum().item()
            layer_pruned = original_count - remaining_count
            layer_sparsity = layer_pruned / original_count

            pruning_info[name] = {
                'original_params': original_count,
                'remaining_params': remaining_count,
                'pruned_params': layer_pruned,
                'sparsity_ratio': layer_sparsity,
                'shape': original_shape
            }

            total_params += original_count
            pruned_params += layer_pruned

            print(f"{name}: {remaining_count:,}/{original_count:,} remaining "
                  f"({layer_sparsity*100:.1f}% pruned)")

    overall_sparsity = pruned_params / total_params
    pruning_info['overall'] = {
        'total_params': total_params,
        'pruned_params': pruned_params,
        'remaining_params': total_params - pruned_params,
        'sparsity_ratio': overall_sparsity
    }

    print(f"\nOverall pruning: {pruned_params:,}/{total_params:,} parameters pruned "
          f"({overall_sparsity*100:.1f}%)")

    return pruned_model, pruning_info


def count_nonzero_parameters(model):
    """Count non-zero parameters in a model (for sparsity measurement)"""
    total_params = 0
    nonzero_params = 0

    for param in model.parameters():
        if param.requires_grad:
            total = param.numel()
            nonzero = torch.count_nonzero(param).item()
            total_params += total
            nonzero_params += nonzero

    sparsity = (total_params - nonzero_params) / total_params
    return total_params, nonzero_params, sparsity


def fine_tune_pruned_model(model, train_loader, val_loader, device, config, epochs=3):
    """
    Fine-tune pruned model to recover accuracy.

    Args:
        model: pruned model to fine-tune
        train_loader, val_loader: data loaders
        device: training device
        config: configuration dictionary
        epochs: number of fine-tuning epochs

    Returns:
        fine_tuned_model: model after fine-tuning
        best_acc: best validation accuracy achieved
    """
    import torch.optim as optim
    from tqdm import tqdm

    model.train()

    # Use lower learning rate for fine-tuning
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['student']['lr'] * 0.1,  # 10x lower LR
        weight_decay=config['student']['weight_decay']
    )

    criterion = nn.CrossEntropyLoss()
    best_acc = 0

    print(f"\nFine-tuning pruned model for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Fine-tune Epoch {epoch}')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Zero out gradients for pruned weights (maintain sparsity)
            for param in model.parameters():
                if param.grad is not None:
                    # Keep gradients only for non-zero weights
                    param.grad = param.grad * (param.data != 0).float()

            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

            acc = 100. * correct / total
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({'Loss': f'{avg_loss:.4f}', 'Acc': f'{acc:.2f}%'})

        train_acc = 100. * correct / total
        print(f'Fine-tune Epoch {epoch} - Train Acc: {train_acc:.2f}%')

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()
                val_total += target.size(0)

        val_acc = 100. * val_correct / val_total
        print(f'Fine-tune Epoch {epoch} - Val Acc: {val_acc:.2f}%')

        if val_acc > best_acc:
            best_acc = val_acc

    return model, best_acc


if __name__ == "__main__":
    # Test pruning functionality
    import yaml
    from models.student import StudentCNN

    # Load config
    with open('src/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create model
    model = StudentCNN(config)
    print(f"Original model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test pruning
    sparsity_ratios = [0.3, 0.5, 0.7, 0.9]

    for sparsity in sparsity_ratios:
        print(f"\n{'='*50}")
        print(f"Testing {sparsity*100:.0f}% sparsity")
        print(f"{'='*50}")

        pruned_model, info = apply_l1_pruning(model, sparsity)

        # Verify sparsity
        total, nonzero, actual_sparsity = count_nonzero_parameters(pruned_model)
        print(f"Actual sparsity achieved: {actual_sparsity*100:.1f}%")
        print(f"Non-zero parameters: {nonzero:,}/{total:,}")