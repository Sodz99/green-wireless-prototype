import argparse
import yaml
import torch
import os

from utils import assert_cuda_or_exit, set_all_seeds
from dataset import get_dataloaders
from models.student import StudentCNN, count_parameters
from pruning import apply_l1_pruning, fine_tune_pruned_model, count_nonzero_parameters


def validate_model(model, test_loader, device):
    """Quick validation function"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    accuracy = 100. * correct / total
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='src/config.yaml')
    parser.add_argument('--student', default='artifacts/student_fp32.pt',
                        help='Path to trained student model')
    parser.add_argument('--sparsity', type=float, default=0.7,
                        help='Pruning sparsity ratio (0.0-1.0)')
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

    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # Load trained student model
    print(f"\nLoading student model from: {args.student}")
    if not os.path.exists(args.student):
        print(f"ERROR: Student checkpoint not found: {args.student}")
        print("Train the student model first: python src/train_student.py")
        return

    student = StudentCNN(config).to(device)
    checkpoint = torch.load(args.student, map_location=device)
    student.load_state_dict(checkpoint['model_state_dict'])
    baseline_acc = checkpoint.get('best_acc', 0)

    print(f"Student parameters: {count_parameters(student):,}")
    print(f"Baseline accuracy: {baseline_acc:.2f}%")

    # Test baseline accuracy
    print(f"\nValidating baseline model...")
    baseline_test_acc = validate_model(student, test_loader, device)
    print(f"Baseline test accuracy: {baseline_test_acc:.2f}%")

    # Apply L1 pruning
    print(f"\n{'='*60}")
    print(f"Applying L1 Magnitude Pruning ({args.sparsity*100:.0f}% sparsity)")
    print(f"{'='*60}")

    pruned_model, pruning_info = apply_l1_pruning(student, args.sparsity)

    # Verify sparsity
    total_params, nonzero_params, actual_sparsity = count_nonzero_parameters(pruned_model)
    print(f"\nPruning verification:")
    print(f"Original parameters: {total_params:,}")
    print(f"Remaining parameters: {nonzero_params:,}")
    print(f"Actual sparsity: {actual_sparsity*100:.1f}%")

    # Test pruned model accuracy (before fine-tuning)
    print(f"\nTesting pruned model (before fine-tuning)...")
    pruned_acc_before = validate_model(pruned_model, test_loader, device)
    print(f"Pruned accuracy (before fine-tuning): {pruned_acc_before:.2f}%")
    accuracy_drop = baseline_test_acc - pruned_acc_before
    print(f"Accuracy drop from pruning: {accuracy_drop:.2f}%")

    # Fine-tune pruned model
    print(f"\nFine-tuning pruned model...")
    fine_tuned_model, best_val_acc = fine_tune_pruned_model(
        pruned_model, train_loader, val_loader, device, config, epochs=3
    )

    # Final evaluation
    print(f"\nFinal evaluation...")
    final_test_acc = validate_model(fine_tuned_model, test_loader, device)
    recovery = final_test_acc - pruned_acc_before
    final_drop = baseline_test_acc - final_test_acc

    # Save pruned model
    pruned_filename = f'artifacts/student_pruned_{int(args.sparsity*100):02d}pct.pt'
    os.makedirs('artifacts', exist_ok=True)
    torch.save({
        'model_state_dict': fine_tuned_model.state_dict(),
        'config': config,
        'pruning_info': pruning_info,
        'sparsity_ratio': actual_sparsity,
        'baseline_acc': baseline_test_acc,
        'pruned_acc_before_ft': pruned_acc_before,
        'final_test_acc': final_test_acc,
        'best_val_acc': best_val_acc
    }, pruned_filename)

    # Results summary
    print(f"\n{'='*60}")
    print(f"L1 PRUNING RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Sparsity ratio: {actual_sparsity*100:.1f}%")
    print(f"Parameters: {nonzero_params:,}/{total_params:,} remaining")
    print(f"Model size reduction: {(1-nonzero_params/total_params)*100:.1f}%")
    print(f"")
    print(f"Accuracy Results:")
    print(f"  Baseline (unpruned): {baseline_test_acc:.2f}%")
    print(f"  After pruning: {pruned_acc_before:.2f}% (drop: {accuracy_drop:.2f}%)")
    print(f"  After fine-tuning: {final_test_acc:.2f}% (recovery: {recovery:.2f}%)")
    print(f"  Final accuracy drop: {final_drop:.2f}%")
    print(f"")
    print(f"Model saved: {pruned_filename}")

    # Quality assessment
    if final_drop < 2.0:
        quality = "Excellent"
    elif final_drop < 5.0:
        quality = "Good"
    elif final_drop < 10.0:
        quality = "Acceptable"
    else:
        quality = "Poor"

    print(f"Pruning quality: {quality} (<2% drop = Excellent)")


if __name__ == "__main__":
    main()