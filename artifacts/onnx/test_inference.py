#!/usr/bin/env python3
"""
ONNX Inference Test Script
Generated automatically by export_onnx.py
"""

import numpy as np
import onnxruntime as ort
import time

def test_onnx_inference(onnx_path, num_samples=100):
    """Test ONNX model inference speed and correctness"""
    print(f"Testing: {onnx_path}")

    # Load ONNX model
    session = ort.InferenceSession(onnx_path)

    # Get input info
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"Input: {input_name} {input_shape}")

    # Create test data
    batch_size = 1
    test_input = np.random.randn(batch_size, 2, 2048).astype(np.float32)

    # Warmup
    for _ in range(10):
        session.run(None, {input_name: test_input})

    # Benchmark
    start_time = time.time()
    for _ in range(num_samples):
        outputs = session.run(None, {input_name: test_input})
    end_time = time.time()

    avg_time = (end_time - start_time) / num_samples * 1000  # ms
    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"Output shape: {outputs[0].shape}")

    return avg_time

if __name__ == "__main__":
    import os

    onnx_models = [        "artifacts\onnx\teacher.onnx",
        "artifacts\onnx\student.onnx",
        "artifacts\onnx\student_kd.onnx",
        "artifacts\onnx\student_pruned.onnx",
    ]

    print("ONNX Inference Performance Test")
    print("=" * 50)

    for model_path in onnx_models:
        if os.path.exists(model_path):
            test_onnx_inference(model_path)
            print()
        else:
            print(f"Model not found: {model_path}")

    print("Test completed!")
