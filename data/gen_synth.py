import numpy as np
import yaml
import argparse
import os
from scipy.signal import resample


def generate_bpsk(n_symbols, snr_db):
    """Generate BPSK modulated signal"""
    # Random binary data
    data = np.random.randint(0, 2, n_symbols)
    # BPSK mapping: 0 -> -1, 1 -> +1
    symbols = 2 * data - 1
    return symbols.astype(np.complex64)


def generate_qpsk(n_symbols, snr_db):
    """Generate QPSK modulated signal"""
    # Random 2-bit symbols
    data = np.random.randint(0, 4, n_symbols)
    # QPSK constellation
    constellation = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
    symbols = constellation[data]
    return symbols.astype(np.complex64)


def generate_8psk(n_symbols, snr_db):
    """Generate 8-PSK modulated signal"""
    # Random 3-bit symbols
    data = np.random.randint(0, 8, n_symbols)
    # 8-PSK constellation
    angles = 2 * np.pi * data / 8
    symbols = np.exp(1j * angles)
    return symbols.astype(np.complex64)


def generate_16qam(n_symbols, snr_db):
    """Generate 16-QAM modulated signal"""
    # Random 4-bit symbols
    data = np.random.randint(0, 16, n_symbols)
    # 16-QAM constellation
    I = np.array([-3, -1, 1, 3])
    Q = np.array([-3, -1, 1, 3])
    constellation = []
    for q in Q:
        for i in I:
            constellation.append(i + 1j*q)
    constellation = np.array(constellation) / np.sqrt(10)  # Normalize
    symbols = constellation[data]
    return symbols.astype(np.complex64)


def add_channel_effects(signal, snr_db, freq_offset_hz, phase_jitter_std, sample_rate=1e6):
    """Add realistic wireless channel effects"""
    # Add AWGN noise
    signal_power = np.mean(np.abs(signal)**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal)))
    signal_noisy = signal + noise

    # Add frequency offset
    t = np.arange(len(signal)) / sample_rate
    freq_offset = np.exp(1j * 2 * np.pi * freq_offset_hz * t)
    signal_offset = signal_noisy * freq_offset

    # Add phase jitter
    phase_noise = np.cumsum(np.random.randn(len(signal)) * phase_jitter_std)
    phase_jitter = np.exp(1j * phase_noise)
    signal_final = signal_offset * phase_jitter

    return signal_final.astype(np.complex64)


def complex_to_iq(signal):
    """Convert complex signal to I/Q format (2, N)"""
    return np.stack([signal.real, signal.imag], axis=0)


def generate_dataset(config):
    """Generate complete RF modulation dataset"""
    print("Generating synthetic RF modulation dataset...")

    modulations = {
        'BPSK': generate_bpsk,
        'QPSK': generate_qpsk,
        'PSK8': generate_8psk,
        'QAM16': generate_16qam
    }

    classes = config['classes']
    snr_values = config['snr_db']
    sample_len = config['sample_len']

    # Calculate samples per class per split
    train_per_class = config['train_n'] // len(classes)
    val_per_class = config['val_n'] // len(classes)
    test_per_class = config['test_n'] // len(classes)

    datasets = {}
    for split, samples_per_class in [('train', train_per_class), ('val', val_per_class), ('test', test_per_class)]:
        X_data = []
        y_data = []

        for class_idx, mod_name in enumerate(classes):
            mod_func = modulations[mod_name]

            for i in range(samples_per_class):
                # Random SNR for variety
                snr = np.random.choice(snr_values)

                # Generate base modulated signal (fewer symbols, will be upsampled)
                n_symbols = sample_len // 8  # Oversample by 8x
                symbols = mod_func(n_symbols, snr)

                # Upsample to desired length
                signal = resample(symbols, sample_len)

                # Add channel effects
                signal = add_channel_effects(
                    signal, snr,
                    config['freq_offset_hz'],
                    config['phase_jitter_std']
                )

                # Convert to I/Q format (2, sample_len)
                iq_signal = complex_to_iq(signal)

                X_data.append(iq_signal)
                y_data.append(class_idx)

        # Convert to numpy arrays
        X_data = np.stack(X_data, axis=0).astype(np.float32)  # Shape: (N, 2, sample_len)
        y_data = np.array(y_data, dtype=np.int64)

        # Shuffle
        indices = np.random.permutation(len(X_data))
        X_data = X_data[indices]
        y_data = y_data[indices]

        datasets[split] = (X_data, y_data)
        print(f"{split}: {X_data.shape}, labels: {y_data.shape}")

    return datasets


def save_datasets(datasets, data_dir):
    """Save datasets to .npy files"""
    os.makedirs(data_dir, exist_ok=True)

    for split, (X, y) in datasets.items():
        np.save(os.path.join(data_dir, f"{split}_x.npy"), X)
        np.save(os.path.join(data_dir, f"{split}_y.npy"), y)
        print(f"Saved {split}: {X.shape}")

    # Save a single sample for demo
    sample_x = datasets['test'][0][0:1]  # First test sample
    np.save(os.path.join(data_dir, "sample.npy"), sample_x)
    print(f"Saved sample.npy: {sample_x.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='src/config.yaml')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set seed
    np.random.seed(config['seed'])

    # Generate datasets
    datasets = generate_dataset(config)

    # Save datasets
    save_datasets(datasets, 'data')

    print(f"\nDataset generation complete!")
    print(f"Classes: {config['classes']}")
    print(f"Sample length: {config['sample_len']}")
    print(f"Train/Val/Test: {config['train_n']}/{config['val_n']}/{config['test_n']}")


if __name__ == "__main__":
    main()