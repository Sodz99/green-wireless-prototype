import argparse
import yaml
import torch
import torch.optim as optim
from tqdm import tqdm
import os

from utils import assert_cuda_or_exit, set_all_seeds
from dataset import get_dataloaders
from models.student import StudentCNN, count_parameters
from models.teacher import TeacherCNN
from kd_loss import KnowledgeDistillationLoss, compute_accuracy


def train_epoch_kd(student, teacher, train_loader, criterion, optimizer, device):
    """Train student for one epoch using Knowledge Distillation"""
    student.train()
    teacher.eval()  # Teacher stays in eval mode

    total_loss = 0
    total_kd_loss = 0
    total_ce_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training KD')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Forward pass through both models
        with torch.no_grad():
            teacher_logits = teacher(data)  # No gradients for teacher

        student_logits = student(data)

        # Compute KD loss
        loss, loss_dict = criterion(student_logits, teacher_logits, target)
        loss.backward()
        optimizer.step()

        # Update statistics
        total_loss += loss_dict['total_loss']
        total_kd_loss += loss_dict['kd_loss']
        total_ce_loss += loss_dict['ce_loss']

        pred = student_logits.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        # Update progress bar
        acc = 100. * correct / total
        avg_loss = total_loss / (batch_idx + 1)
        avg_kd = total_kd_loss / (batch_idx + 1)
        avg_ce = total_ce_loss / (batch_idx + 1)

        pbar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'KD': f'{avg_kd:.4f}',
            'CE': f'{avg_ce:.4f}',
            'Acc': f'{acc:.2f}%'
        })

    return {
        'total_loss': total_loss / len(train_loader),
        'kd_loss': total_kd_loss / len(train_loader),
        'ce_loss': total_ce_loss / len(train_loader),
        'accuracy': 100. * correct / total
    }


def validate(model, val_loader, device):
    """Validate the model"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    accuracy = 100. * correct / total
    print(f'Validation Accuracy: {accuracy:.2f}%')
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='src/config.yaml')
    parser.add_argument('--teacher', required=True, help='Path to teacher checkpoint')
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

    # Load teacher model
    print(f"\nLoading teacher from: {args.teacher}")
    teacher = TeacherCNN(config).to(device)
    teacher_checkpoint = torch.load(args.teacher, map_location=device)
    teacher.load_state_dict(teacher_checkpoint['model_state_dict'])
    teacher.eval()  # Keep teacher in eval mode
    print(f"Teacher parameters: {count_parameters(teacher):,}")

    # Create student model
    print("\nCreating Student model...")
    student = StudentCNN(config).to(device)
    print(f"Student parameters: {count_parameters(student):,}")

    # Knowledge Distillation loss
    kd_config = config['kd']
    criterion = KnowledgeDistillationLoss(
        temperature=kd_config['temperature'],
        alpha=kd_config['alpha']
    )
    print(f"KD Loss - Temperature: {kd_config['temperature']}, Alpha: {kd_config['alpha']}")

    # Optimizer
    optimizer = optim.Adam(
        student.parameters(),  # Only optimize student
        lr=config['student']['lr'],
        weight_decay=config['student']['weight_decay']
    )

    # Training setup
    epochs = config['student']['epochs']
    best_acc = 0
    patience = 2
    patience_counter = 0

    print(f"\nTraining Student with Knowledge Distillation for {epochs} epochs...")

    # Training loop
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # Train with KD
        train_stats = train_epoch_kd(student, teacher, train_loader, criterion, optimizer, device)
        print(f'Train - Loss: {train_stats["total_loss"]:.4f} '
              f'(KD: {train_stats["kd_loss"]:.4f}, CE: {train_stats["ce_loss"]:.4f}), '
              f'Accuracy: {train_stats["accuracy"]:.2f}%')

        # Validate
        val_acc = validate(student, val_loader, device)

        # Early stopping and save best model
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            # Save best model
            os.makedirs('artifacts', exist_ok=True)
            torch.save({
                'model_state_dict': student.state_dict(),
                'config': config,
                'epoch': epoch,
                'best_acc': best_acc,
                'teacher_path': args.teacher,
                'kd_config': kd_config
            }, 'artifacts/student_kd_fp32.pt')
            print(f"New best Student KD model saved! Accuracy: {best_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break

    # Final test evaluation
    print(f"\nFinal evaluation on test set...")
    test_acc = validate(student, test_loader, device)

    print(f"\nStudent KD training complete!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Final test accuracy: {test_acc:.2f}%")
    print(f"Model saved to: artifacts/student_kd_fp32.pt")

    # Compare with baseline (if available)
    baseline_path = 'artifacts/student_fp32.pt'
    if os.path.exists(baseline_path):
        baseline_checkpoint = torch.load(baseline_path, map_location=device)
        baseline_acc = baseline_checkpoint.get('best_acc', 0)
        improvement = best_acc - baseline_acc
        print(f"\nComparison with baseline:")
        print(f"Baseline Student: {baseline_acc:.2f}%")
        print(f"KD Student: {best_acc:.2f}%")
        print(f"Improvement: {improvement:+.2f}% {'✓' if improvement > 0 else ''}")


if __name__ == "__main__":
    main()