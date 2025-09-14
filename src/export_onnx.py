import argparse
import os
import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
import yaml
from pathlib import Path

from models.teacher import TeacherCNN
from models.student import StudentCNN, count_parameters


def export_to_onnx(model, example_input, output_path, input_names=None, output_names=None):
    """
    Export PyTorch model to ONNX format

    Args:
        model: PyTorch model to export
        example_input: Example input tensor for tracing
        output_path: Path to save ONNX model
        input_names: List of input names (optional)
        output_names: List of output names (optional)
    """
    print(f"Exporting model to ONNX: {output_path}")

    # Set model to evaluation mode
    model.eval()

    # Default names
    if input_names is None:
        input_names = ['input']
    if output_names is None:
        output_names = ['output']

    # Export the model
    torch.onnx.export(
        model,
        example_input,
        output_path,
        export_params=True,
        opset_version=11,  # Use opset 11 for compatibility
        do_constant_folding=True,  # Optimize constant operations
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            input_names[0]: {0: 'batch_size'},  # Variable batch size
            output_names[0]: {0: 'batch_size'}
        }
    )

    print(f"ONNX model saved to: {output_path}")


def verify_onnx_model(onnx_path, pytorch_model, example_input, tolerance=1e-5):
    """
    Verify ONNX model produces same outputs as PyTorch model

    Args:
        onnx_path: Path to ONNX model
        pytorch_model: Original PyTorch model
        example_input: Test input tensor
        tolerance: Numerical tolerance for comparison

    Returns:
        bool: True if outputs match within tolerance
    """
    print(f"Verifying ONNX model: {onnx_path}")

    # Load ONNX model
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("OK ONNX model is valid")
    except Exception as e:
        print(f"ERROR ONNX model validation failed: {e}")
        return False

    # Create ONNX Runtime session
    try:
        ort_session = ort.InferenceSession(onnx_path)
        print("OK ONNX Runtime session created successfully")
    except Exception as e:
        print(f"ERROR Failed to create ONNX Runtime session: {e}")
        return False

    # Get PyTorch output
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(example_input)

    # Get ONNX output
    input_name = ort_session.get_inputs()[0].name
    onnx_input = {input_name: example_input.numpy()}
    onnx_output = ort_session.run(None, onnx_input)[0]

    # Compare outputs
    pytorch_output_np = pytorch_output.numpy()
    max_diff = np.max(np.abs(pytorch_output_np - onnx_output))
    mean_diff = np.mean(np.abs(pytorch_output_np - onnx_output))

    print(f"Output comparison:")
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")
    print(f"  Tolerance: {tolerance:.2e}")

    if max_diff < tolerance:
        print("OK ONNX and PyTorch outputs match")
        return True
    else:
        print("ERROR ONNX and PyTorch outputs differ significantly")
        return False


def get_onnx_model_info(onnx_path):
    """Get information about ONNX model"""
    try:
        onnx_model = onnx.load(onnx_path)

        # Get model size
        model_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB

        # Get input/output info
        inputs = [(inp.name, inp.type.tensor_type.shape) for inp in onnx_model.graph.input]
        outputs = [(out.name, out.type.tensor_type.shape) for out in onnx_model.graph.output]

        # Count operations
        num_nodes = len(onnx_model.graph.node)

        print(f"ONNX Model Info:")
        print(f"  File size: {model_size:.2f} MB")
        print(f"  Number of nodes: {num_nodes}")
        print(f"  Inputs: {inputs}")
        print(f"  Outputs: {outputs}")

        return {
            'size_mb': model_size,
            'num_nodes': num_nodes,
            'inputs': inputs,
            'outputs': outputs
        }
    except Exception as e:
        print(f"Error getting ONNX model info: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='src/config.yaml')
    parser.add_argument('--export-all', action='store_true',
                        help='Export all available models')
    parser.add_argument('--teacher', default='artifacts/teacher_fp32.pt')
    parser.add_argument('--student', default='artifacts/student_fp32.pt')
    parser.add_argument('--student-kd', default='artifacts/student_kd_fp32.pt')
    parser.add_argument('--student-pruned', default='artifacts/student_pruned_50pct.pt')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create output directory
    onnx_dir = Path('artifacts/onnx')
    onnx_dir.mkdir(exist_ok=True)

    # Example input (batch_size=1, channels=2, sequence_length=2048)
    example_input = torch.randn(1, 2, 2048)

    print("Starting ONNX export process...")
    print(f"Example input shape: {example_input.shape}")

    models_to_export = []

    if args.export_all:
        # Export all available models
        model_files = [
            (args.teacher, 'teacher', TeacherCNN),
            (args.student, 'student', StudentCNN),
            (args.student_kd, 'student_kd', StudentCNN),
            (args.student_pruned, 'student_pruned', StudentCNN)
        ]
    else:
        # Export only teacher and student by default
        model_files = [
            (args.teacher, 'teacher', TeacherCNN),
            (args.student, 'student', StudentCNN)
        ]

    export_results = []

    for model_path, model_name, model_class in model_files:
        if not os.path.exists(model_path):
            print(f"Skipping {model_name}: checkpoint not found at {model_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Exporting {model_name.upper()} model")
        print(f"{'='*60}")

        try:
            # Load model
            print(f"Loading model from: {model_path}")
            model = model_class(config)
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            print(f"Model parameters: {count_parameters(model):,}")

            # Export to ONNX
            onnx_path = onnx_dir / f"{model_name}.onnx"
            export_to_onnx(
                model,
                example_input,
                str(onnx_path),
                input_names=['rf_signal'],
                output_names=['modulation_logits']
            )

            # Verify export
            is_valid = verify_onnx_model(onnx_path, model, example_input)

            # Get model info
            model_info = get_onnx_model_info(onnx_path)

            export_results.append({
                'name': model_name,
                'path': str(onnx_path),
                'pytorch_params': count_parameters(model),
                'onnx_size_mb': model_info['size_mb'] if model_info else 0,
                'valid': is_valid
            })

            if is_valid:
                print(f"OK Successfully exported {model_name} to ONNX")
            else:
                print(f"ERROR Export verification failed for {model_name}")

        except Exception as e:
            print(f"ERROR Failed to export {model_name}: {e}")
            export_results.append({
                'name': model_name,
                'path': None,
                'pytorch_params': 0,
                'onnx_size_mb': 0,
                'valid': False
            })

    # Summary
    print(f"\n{'='*60}")
    print("ONNX EXPORT SUMMARY")
    print(f"{'='*60}")

    successful_exports = [r for r in export_results if r['valid']]
    failed_exports = [r for r in export_results if not r['valid']]

    print(f"Successfully exported: {len(successful_exports)}")
    print(f"Failed exports: {len(failed_exports)}")

    if successful_exports:
        print(f"\nSuccessful exports:")
        for result in successful_exports:
            print(f"  {result['name']}: {result['onnx_size_mb']:.2f} MB "
                  f"({result['pytorch_params']:,} params) -> {result['path']}")

    if failed_exports:
        print(f"\nFailed exports:")
        for result in failed_exports:
            print(f"  {result['name']}")

    # Create inference test script
    create_onnx_inference_test(onnx_dir, successful_exports)

    print(f"\nONNX models saved to: {onnx_dir}")
    print("Ready for deployment and quantization!")


def create_onnx_inference_test(onnx_dir, export_results):
    """Create a test script for ONNX inference"""
    test_script = f"""#!/usr/bin/env python3
\"\"\"
ONNX Inference Test Script
Generated automatically by export_onnx.py
\"\"\"

import numpy as np
import onnxruntime as ort
import time

def test_onnx_inference(onnx_path, num_samples=100):
    \"\"\"Test ONNX model inference speed and correctness\"\"\"
    print(f"Testing: {{onnx_path}}")

    # Load ONNX model
    session = ort.InferenceSession(onnx_path)

    # Get input info
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"Input: {{input_name}} {{input_shape}}")

    # Create test data
    batch_size = 1
    test_input = np.random.randn(batch_size, 2, 2048).astype(np.float32)

    # Warmup
    for _ in range(10):
        session.run(None, {{input_name: test_input}})

    # Benchmark
    start_time = time.time()
    for _ in range(num_samples):
        outputs = session.run(None, {{input_name: test_input}})
    end_time = time.time()

    avg_time = (end_time - start_time) / num_samples * 1000  # ms
    print(f"Average inference time: {{avg_time:.2f}} ms")
    print(f"Output shape: {{outputs[0].shape}}")

    return avg_time

if __name__ == "__main__":
    import os

    onnx_models = ["""

    for result in export_results:
        if result['valid']:
            test_script += f'        "{result["path"]}",\n'

    test_script += """    ]

    print("ONNX Inference Performance Test")
    print("=" * 50)

    for model_path in onnx_models:
        if os.path.exists(model_path):
            test_onnx_inference(model_path)
            print()
        else:
            print(f"Model not found: {model_path}")

    print("Test completed!")
"""

    test_file = onnx_dir / "test_inference.py"
    with open(test_file, 'w') as f:
        f.write(test_script)

    print(f"Created ONNX inference test: {test_file}")


if __name__ == "__main__":
    main()