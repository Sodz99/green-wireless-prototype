import torch
import torch.nn as nn


class TeacherCNN(nn.Module):
    """Larger 1D-CNN Teacher model for RF modulation classification"""

    def __init__(self, config):
        super(TeacherCNN, self).__init__()

        # Extract teacher config
        teacher_cfg = config['teacher']
        conv_channels = teacher_cfg['conv_channels']  # [64, 128, 256]
        kernel_sizes = teacher_cfg['kernel_sizes']    # [7, 5, 3]
        strides = teacher_cfg['strides']             # [2, 2, 2]
        dropout_rate = teacher_cfg['dropout']        # 0.1
        num_classes = len(config['classes'])         # 4

        # Input: (batch, 2, 2048) - I/Q channels
        input_channels = config['channels']  # 2

        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, conv_channels[0], kernel_sizes[0], stride=strides[0], padding=kernel_sizes[0]//2),
            nn.BatchNorm1d(conv_channels[0]),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_sizes[1], stride=strides[1], padding=kernel_sizes[1]//2),
            nn.BatchNorm1d(conv_channels[1]),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(conv_channels[1], conv_channels[2], kernel_sizes[2], stride=strides[2], padding=kernel_sizes[2]//2),
            nn.BatchNorm1d(conv_channels[2]),
            nn.ReLU(inplace=True)
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classifier
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(conv_channels[2], num_classes)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize model weights"""
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass
        Input: (batch, 2, 2048)
        Output: (batch, num_classes)
        """
        # Convolutional feature extraction
        x = self.conv1(x)      # (batch, 64, 1024)
        x = self.conv2(x)      # (batch, 128, 512)
        x = self.conv3(x)      # (batch, 256, 256)

        # Global pooling and classification
        x = self.global_pool(x)  # (batch, 256, 1)
        x = x.squeeze(-1)        # (batch, 256)
        x = self.dropout(x)
        logits = self.classifier(x)  # (batch, num_classes)

        return logits

    def get_feature_maps(self, x):
        """Extract intermediate feature maps for visualization/analysis"""
        features = []
        x = self.conv1(x)
        features.append(x)
        x = self.conv2(x)
        features.append(x)
        x = self.conv3(x)
        features.append(x)
        return features


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    import yaml
    with open('src/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model = TeacherCNN(config)
    print(f"Teacher model parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(8, 2, 2048)
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")

    # Test feature extraction
    features = model.get_feature_maps(x)
    for i, feat in enumerate(features):
        print(f"Feature map {i+1}: {feat.shape}")