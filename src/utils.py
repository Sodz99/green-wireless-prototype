import sys
import time
import random
import os
import numpy as np
import torch
import onnxruntime as ort


def assert_cuda_or_exit():
    """Exit with error if CUDA is not available"""
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This project requires GPU training.")
        sys.exit(1)
    print(f"CUDA available: {torch.cuda.get_device_name(0)}")


def set_all_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def file_size_mb(path):
    """Get file size in MB"""
    if not os.path.exists(path):
        return 0.0
    return os.path.getsize(path) / (1024 * 1024)


def perf_timer(fn, runs=1000, warmup=50):
    """Performance timer for functions. Returns {median_ms, p95_ms}"""
    # Warmup runs
    for _ in range(warmup):
        fn()

    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    times = np.array(times)
    return {
        'median_ms': float(np.median(times)),
        'p95_ms': float(np.percentile(times, 95))
    }


def setup_ort_threads():
    """Set ONNX Runtime CPU threads to 1 for stable latency benchmarks"""
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    # Create session options for consistent threading
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    return session_options


def print_env_summary():
    """Print environment summary for reproducibility"""
    print("\n=== Environment Summary ===")
    print(f"PyTorch: {torch.__version__}")
    print(f"ONNX Runtime: {ort.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"Python: {sys.version}")
    print("=" * 30)