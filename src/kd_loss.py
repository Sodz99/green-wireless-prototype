import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss combining soft targets (teacher) and hard targets (ground truth)

    L_total = alpha * L_KD + (1-alpha) * L_CE

    where:
    - L_KD = T^2 * KLDiv(log_softmax(student/T), softmax(teacher/T))
    - L_CE = CrossEntropy(student_logits, true_labels)
    - T = temperature parameter
    - alpha = weight for KD loss vs CE loss
    """

    def __init__(self, temperature=4.0, alpha=0.6):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, true_labels):
        """
        Args:
            student_logits: (batch_size, num_classes) - raw logits from student
            teacher_logits: (batch_size, num_classes) - raw logits from teacher
            true_labels: (batch_size,) - ground truth class labels

        Returns:
            total_loss: weighted combination of KD and CE losses
            loss_dict: dictionary with individual loss components for logging
        """

        # Temperature-scaled probabilities
        T = self.temperature

        # Teacher soft targets (detached to prevent teacher gradient updates)
        teacher_soft = F.softmax(teacher_logits.detach() / T, dim=1)

        # Student soft predictions (log probabilities for KL divergence)
        student_soft = F.log_softmax(student_logits / T, dim=1)

        # KL Divergence loss between soft distributions
        kd_loss = self.kl_div(student_soft, teacher_soft) * (T * T)

        # Standard cross-entropy loss with hard labels
        ce_loss = self.ce_loss(student_logits, true_labels)

        # Weighted combination
        total_loss = self.alpha * kd_loss + (1.0 - self.alpha) * ce_loss

        # Return loss components for logging
        loss_dict = {
            'total_loss': total_loss.item(),
            'kd_loss': kd_loss.item(),
            'ce_loss': ce_loss.item(),
            'kd_weight': self.alpha,
            'ce_weight': (1.0 - self.alpha)
        }

        return total_loss, loss_dict


def compute_accuracy(logits, labels):
    """Compute classification accuracy"""
    with torch.no_grad():
        predictions = torch.argmax(logits, dim=1)
        correct = (predictions == labels).float()
        accuracy = correct.mean().item() * 100.0
    return accuracy


if __name__ == "__main__":
    # Test the KD loss function
    batch_size, num_classes = 32, 4
    temperature, alpha = 4.0, 0.6

    # Create test data
    student_logits = torch.randn(batch_size, num_classes, requires_grad=True)
    teacher_logits = torch.randn(batch_size, num_classes)
    true_labels = torch.randint(0, num_classes, (batch_size,))

    # Initialize KD loss
    kd_criterion = KnowledgeDistillationLoss(temperature=temperature, alpha=alpha)

    # Compute loss
    total_loss, loss_dict = kd_criterion(student_logits, teacher_logits, true_labels)

    print("Knowledge Distillation Loss Test:")
    print(f"Temperature: {temperature}, Alpha: {alpha}")
    print(f"Student logits shape: {student_logits.shape}")
    print(f"Teacher logits shape: {teacher_logits.shape}")
    print(f"True labels shape: {true_labels.shape}")
    print()
    print("Loss Components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")

    print(f"\nTotal loss: {total_loss.item():.4f}")
    print(f"Loss requires grad: {total_loss.requires_grad}")

    # Test accuracy computation
    acc = compute_accuracy(student_logits, true_labels)
    print(f"Student accuracy: {acc:.2f}%")

    # Test backward pass
    total_loss.backward()
    print(f"Student grad norm: {student_logits.grad.norm().item():.4f}")
    print("KD loss function working correctly!")