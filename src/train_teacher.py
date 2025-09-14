import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from utils import assert_cuda_or_exit, set_all_seeds
from dataset import get_dataloaders
from models.teacher import TeacherCNN, count_parameters


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        # Update progress bar
        acc = 100. * correct / total
        avg_loss = total_loss / (batch_idx + 1)
        pbar.set_postfix({'Loss': f'{avg_loss:.4f}', 'Acc': f'{acc:.2f}%'})

    return total_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    accuracy = 100. * correct / total
    avg_loss = total_loss / len(val_loader)
    print(f'Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='src/config.yaml')
    args = parser.parse_args()

    # Check CUDA
    assert_cuda_or_exit()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set seeds
    set_all_seeds(config['seed'])

    # Setup device
    device = torch.device(config['device'])
    print(f"Using device: {device}")

    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # Create model
    print("\nCreating Teacher model...")
    model = TeacherCNN(config).to(device)
    print(f"Teacher parameters: {count_parameters(model):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['teacher']['lr'],
        weight_decay=config['teacher']['weight_decay']
    )

    # Training setup
    epochs = config['teacher']['epochs']
    best_acc = 0
    patience = 2
    patience_counter = 0

    print(f"\nTraining for {epochs} epochs...")

    # Training loop
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f'Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            # Save best model
            os.makedirs('artifacts', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'epoch': epoch,
                'best_acc': best_acc
            }, 'artifacts/teacher_fp32.pt')
            print(f"New best model saved! Accuracy: {best_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break

    # Final test evaluation
    print(f"\nFinal evaluation on test set...")
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f'Test - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%')

    print(f"\nTeacher training complete!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Final test accuracy: {test_acc:.2f}%")
    print(f"Model saved to: artifacts/teacher_fp32.pt")


if __name__ == "__main__":
    main()