"""
ArcFace (Additive Angular Margin Loss) for manuscript classification.

ArcFace adds an angular margin to the softmax loss to enhance discriminative power.
It requires normalized features and normalized classifier weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss
    
    Paper: "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    
    Args:
        num_classes: Number of classes
        embedding_dim: Dimension of feature embeddings
        margin: Angular margin (default: 0.5 radians ≈ 28.6 degrees)
        scale: Scale factor for logits (default: 64)
    """
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        margin: float = 0.5,
        scale: float = 64.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.target_margin = margin  # Final margin after warmup
        self.register_buffer("current_margin", torch.tensor(float(margin), dtype=torch.float32))
        self.scale = scale
        
        # Classifier weight matrix (will be normalized)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
    
    def set_margin(self, margin: float):
        """Update the active margin (used for warmup scheduling)."""
        self.current_margin.fill_(float(margin))
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute ArcFace loss.
        
        Args:
            embeddings: Normalized feature embeddings [B, embedding_dim]
            labels: Class labels [B] (must be in range [0, num_classes-1])
            
        Returns:
            Tuple of (loss, logits):
            - loss: ArcFace loss (scalar)
            - logits: Scaled cosine logits [B, num_classes] (for accuracy computation)
        """
        # Validate inputs
        if embeddings.size(0) != labels.size(0):
            raise ValueError(f"Batch size mismatch: embeddings={embeddings.size(0)}, labels={labels.size(0)}")
        
        # Check for invalid labels (out of range or negative)
        if (labels < 0).any() or (labels >= self.num_classes).any():
            invalid_labels = labels[(labels < 0) | (labels >= self.num_classes)]
            raise ValueError(
                f"Invalid labels detected: {invalid_labels.tolist()}. "
                f"Labels must be in range [0, {self.num_classes-1}]. "
                f"This usually indicates a data loading bug (e.g., label2idx returning -1)."
            )
        
        # Check for zero vectors before normalization (would produce NaN)
        embedding_norms = torch.norm(embeddings, p=2, dim=1)
        if (embedding_norms < 1e-8).any():
            zero_count = (embedding_norms < 1e-8).sum().item()
            raise ValueError(
                f"Found {zero_count} zero/near-zero embedding vectors. "
                f"Normalization would produce NaN. Check model outputs."
            )
        
        # Normalize embeddings (should already be normalized, but ensure it)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Check for NaN after normalization
        if torch.isnan(embeddings).any():
            raise ValueError("NaN detected in embeddings after normalization. Check model outputs.")
        
        # Normalize weight matrix and ensure it matches embeddings dtype
        # This handles mixed precision training where embeddings may be bfloat16/float16
        weight = F.normalize(self.weight, p=2, dim=1)
        weight = weight.to(dtype=embeddings.dtype)  # Match embeddings dtype
        
        # Compute cosine similarity: embeddings @ weight.T
        cosine = F.linear(embeddings, weight)  # [B, num_classes]
        
        # Clamp cosine to valid range [-1, 1] for numerical stability
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
        
        # Compute angles
        theta = torch.acos(cosine)  # [B, num_classes]
        
        # Add margin to the angle of the true class
        # Use advanced indexing safely (labels are already validated)
        batch_indices = torch.arange(0, embeddings.size(0), device=labels.device)
        target_theta = theta[batch_indices, labels].view(-1, 1)  # [B, 1]
        target_theta_margin = target_theta + self.current_margin  # [B, 1]
        
        # Create one-hot encoding for true class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Apply margin only to true class
        output = (one_hot * (torch.cos(target_theta_margin) - torch.cos(target_theta))) + cosine
        output *= self.scale
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(output, labels)
        
        # Return both loss and *unmodified* scaled-cosine logits for accuracy
        # (margin is a training-time trick; at inference we predict via cosine)
        cosine_logits = cosine * self.scale  # [B, num_classes]
        
        return loss, cosine_logits
