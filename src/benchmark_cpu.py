import argparse
import time
import numpy as np
import torch
import onnxruntime as ort
import yaml
import os
from pathlib import Path
from statistics import mean, stdev
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from models.teacher import TeacherCNN
from models.student import StudentCNN, count_parameters


def benchmark_pytorch_model(model, test_loader, device, num_samples=500):
    """
    Benchmark PyTorch model on CPU

    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device (should be 'cpu' for this benchmark)
        num_samples: Number of samples to benchmark

    Returns:
        dict: Benchmark results
    """
    print(f"Benchmarking PyTorch model on {device}...")

    model.eval()
    model = model.to(device)

    correct = 0
    total = 0
    inference_times = []

    sample_count = 0
    with torch.no_grad():
        for data, target in test_loader:
            if sample_count >= num_samples:
                break

            data, target = data.to(device), target.to(device)
            batch_size = data.shape[0]

            for i in range(batch_size):
                if sample_count >= num_samples:
                    break

                single_input = data[i:i+1]
                single_target = target[i:i+1]

                # Measure inference time
                start_time = time.time()
                output = model(single_input)
                end_time = time.time()

                inference_times.append((end_time - start_time) * 1000)  # ms

                # Check accuracy
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(single_target.view_as(pred)).sum().item()
                total += 1
                sample_count += 1

    accuracy = 100. * correct / total
    avg_time = mean(inference_times)
    std_time = stdev(inference_times) if len(inference_times) > 1 else 0

    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Avg inference time: {avg_time:.2f} ± {std_time:.2f} ms")

    return {
        'accuracy': accuracy,
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'total_samples': total,
        'times': inference_times
    }


def benchmark_onnx_model(model_path, test_loader, num_samples=500):
    """
    Benchmark ONNX model on CPU

    Args:
        model_path: Path to ONNX model
        test_loader: Test data loader
        num_samples: Number of samples to benchmark

    Returns:
        dict: Benchmark results
    """
    print(f"Benchmarking ONNX model: {Path(model_path).name}")

    # Create ONNX Runtime session with CPU provider
    providers = ['CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)

    input_name = session.get_inputs()[0].name

    correct = 0
    total = 0
    inference_times = []

    sample_count = 0
    for data, target in test_loader:
        if sample_count >= num_samples:
            break

        data_np = data.numpy()
        target_np = target.numpy()
        batch_size = data_np.shape[0]

        for i in range(batch_size):
            if sample_count >= num_samples:
                break

            single_input = data_np[i:i+1].astype(np.float32)
            single_target = target_np[i]

            # Measure inference time
            start_time = time.time()
            try:
                output = session.run(None, {input_name: single_input})[0]
                end_time = time.time()

                inference_times.append((end_time - start_time) * 1000)  # ms

                # Check accuracy
                pred = np.argmax(output[0])
                if pred == single_target:
                    correct += 1

                total += 1
                sample_count += 1

            except Exception as e:
                print(f"  ERROR during inference: {e}")
                # Skip this model if inference fails
                return {
                    'accuracy': 0,
                    'avg_time_ms': 0,
                    'std_time_ms': 0,
                    'total_samples': 0,
                    'times': [],
                    'error': str(e)
                }

    if total == 0:
        return {
            'accuracy': 0,
            'avg_time_ms': 0,
            'std_time_ms': 0,
            'total_samples': 0,
            'times': [],
            'error': 'No samples processed'
        }

    accuracy = 100. * correct / total
    avg_time = mean(inference_times)
    std_time = stdev(inference_times) if len(inference_times) > 1 else 0

    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Avg inference time: {avg_time:.2f} ± {std_time:.2f} ms")

    return {
        'accuracy': accuracy,
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'total_samples': total,
        'times': inference_times
    }


def get_model_size_info(model_path):
    """Get model file size and parameter info"""
    if isinstance(model_path, (str, Path)):
        # ONNX model
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        return {'size_mb': size_mb, 'params': 'N/A'}
    else:
        # PyTorch model
        size_mb = 0  # Not easily calculable for PyTorch
        params = count_parameters(model_path)
        return {'size_mb': size_mb, 'params': params}


def plot_benchmark_results(results, output_path):
    """Create visualization of benchmark results"""

    # Prepare data
    model_names = []
    accuracies = []
    inference_times = []
    sizes = []

    for result in results:
        if 'error' not in result:
            model_names.append(result['name'])
            accuracies.append(result['accuracy'])
            inference_times.append(result['avg_time_ms'])
            sizes.append(result.get('size_mb', 0))

    if not model_names:
        print("No valid results to plot")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Accuracy comparison
    bars1 = ax1.bar(model_names, accuracies, color='lightblue')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Model Accuracy Comparison')
    ax1.tick_params(axis='x', rotation=45)
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom')

    # Inference time comparison
    bars2 = ax2.bar(model_names, inference_times, color='lightcoral')
    ax2.set_ylabel('Inference Time (ms)')
    ax2.set_title('Inference Time Comparison (CPU)')
    ax2.tick_params(axis='x', rotation=45)
    for bar, time_ms in zip(bars2, inference_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{time_ms:.1f}ms', ha='center', va='bottom')

    # Model size comparison
    if any(s > 0 for s in sizes):
        bars3 = ax3.bar(model_names, sizes, color='lightgreen')
        ax3.set_ylabel('Model Size (MB)')
        ax3.set_title('Model Size Comparison')
        ax3.tick_params(axis='x', rotation=45)
        for bar, size in zip(bars3, sizes):
            if size > 0:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{size:.2f}MB', ha='center', va='bottom')
    else:
        ax3.text(0.5, 0.5, 'Size data not available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Model Size Comparison')

    # Accuracy vs Speed trade-off
    ax4.scatter(inference_times, accuracies, s=100, alpha=0.7)
    for i, name in enumerate(model_names):
        ax4.annotate(name, (inference_times[i], accuracies[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax4.set_xlabel('Inference Time (ms)')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_title('Accuracy vs Speed Trade-off')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Benchmark visualization saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='src/config.yaml')
    parser.add_argument('--num-samples', type=int, default=500,
                        help='Number of samples to benchmark')
    parser.add_argument('--pytorch-models', action='store_true',
                        help='Benchmark original PyTorch models')
    parser.add_argument('--onnx-fp32', action='store_true',
                        help='Benchmark FP32 ONNX models')
    parser.add_argument('--onnx-int8', action='store_true',
                        help='Benchmark INT8 ONNX models')
    parser.add_argument('--all', action='store_true',
                        help='Benchmark all available models')
    parser.add_argument('--output', default='artifacts/benchmark_results.png',
                        help='Output path for visualization')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load test data
    print("Loading test data...")
    _, _, test_loader = get_dataloaders(config)

    # Force CPU device for fair comparison
    device = torch.device('cpu')
    print(f"Running CPU benchmark with {args.num_samples} samples...\n")

    benchmark_results = []

    # Benchmark PyTorch models
    if args.pytorch_models or args.all:
        print("=" * 60)
        print("PYTORCH MODELS (CPU)")
        print("=" * 60)

        pytorch_models = [
            ('artifacts/teacher_fp32.pt', 'Teacher (PyTorch)', TeacherCNN),
            ('artifacts/student_fp32.pt', 'Student (PyTorch)', StudentCNN),
            ('artifacts/student_kd_fp32.pt', 'Student KD (PyTorch)', StudentCNN),
            ('artifacts/student_pruned_50pct.pt', 'Student Pruned (PyTorch)', StudentCNN)
        ]

        for model_path, model_name, model_class in pytorch_models:
            if os.path.exists(model_path):
                print(f"\n{model_name}")
                print("-" * 40)

                # Load model
                model = model_class(config)
                checkpoint = torch.load(model_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])

                # Benchmark
                result = benchmark_pytorch_model(model, test_loader, device, args.num_samples)
                result['name'] = model_name
                result['type'] = 'pytorch'
                result['params'] = count_parameters(model)
                result['size_mb'] = 0  # Not easily calculable
                benchmark_results.append(result)

    # Benchmark FP32 ONNX models
    if args.onnx_fp32 or args.all:
        print(f"\n{'=' * 60}")
        print("FP32 ONNX MODELS (CPU)")
        print("=" * 60)

        onnx_fp32_dir = Path('artifacts/onnx')
        onnx_models = list(onnx_fp32_dir.glob("*.onnx"))

        for model_path in onnx_models:
            print(f"\n{model_path.stem.title()} (FP32 ONNX)")
            print("-" * 40)

            result = benchmark_onnx_model(model_path, test_loader, args.num_samples)
            result['name'] = f"{model_path.stem.title()} (FP32)"
            result['type'] = 'onnx_fp32'

            size_info = get_model_size_info(model_path)
            result['size_mb'] = size_info['size_mb']

            benchmark_results.append(result)

    # Benchmark INT8 ONNX models
    if args.onnx_int8 or args.all:
        print(f"\n{'=' * 60}")
        print("INT8 ONNX MODELS (CPU)")
        print("=" * 60)

        onnx_int8_dir = Path('artifacts/onnx_quantized')
        if onnx_int8_dir.exists():
            onnx_models = list(onnx_int8_dir.glob("*_int8.onnx"))

            for model_path in onnx_models:
                model_name = model_path.stem.replace('_int8', '').title()
                print(f"\n{model_name} (INT8 ONNX)")
                print("-" * 40)

                result = benchmark_onnx_model(model_path, test_loader, args.num_samples)
                result['name'] = f"{model_name} (INT8)"
                result['type'] = 'onnx_int8'

                size_info = get_model_size_info(model_path)
                result['size_mb'] = size_info['size_mb']

                benchmark_results.append(result)
        else:
            print("No INT8 ONNX models found. Run quantization first.")

    # Summary
    print(f"\n{'=' * 60}")
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    valid_results = [r for r in benchmark_results if 'error' not in r]

    if valid_results:
        print(f"\nSuccessfully benchmarked {len(valid_results)} models:")
        print(f"{'Model':<30} {'Accuracy':<12} {'Time (ms)':<12} {'Size (MB)':<12}")
        print("-" * 66)

        for result in valid_results:
            name = result['name'][:28]
            acc = f"{result['accuracy']:.1f}%"
            time_ms = f"{result['avg_time_ms']:.2f}"
            size_mb = f"{result['size_mb']:.2f}" if result['size_mb'] > 0 else "N/A"
            print(f"{name:<30} {acc:<12} {time_ms:<12} {size_mb:<12}")

        # Find best models
        best_accuracy = max(valid_results, key=lambda x: x['accuracy'])
        fastest_model = min(valid_results, key=lambda x: x['avg_time_ms'])

        print(f"\nBest accuracy: {best_accuracy['name']} ({best_accuracy['accuracy']:.2f}%)")
        print(f"Fastest inference: {fastest_model['name']} ({fastest_model['avg_time_ms']:.2f}ms)")

        # Create visualization
        plot_benchmark_results(valid_results, args.output)

    else:
        print("No models were successfully benchmarked.")

    failed_results = [r for r in benchmark_results if 'error' in r]
    if failed_results:
        print(f"\nFailed benchmarks ({len(failed_results)}):")
        for result in failed_results:
            print(f"  {result['name']}: {result['error']}")

    print(f"\nCPU benchmarking complete!")
    if valid_results:
        print(f"Results visualization saved to: {args.output}")


if __name__ == "__main__":
    main()