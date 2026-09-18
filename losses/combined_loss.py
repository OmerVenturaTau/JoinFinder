import torch
import torch.nn as nn
from losses.arcface_loss import ArcFaceLoss

class CombinedLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        arcface_weight: float = 1.0,
        ce_weight: float = 1.0,
        sparsity_weight: float = 0.0,
        arcface_margin: float = 0.5,
        arcface_scale: float = 64.0,
        tile_aux_weight: float = 0.0,
        glyph_aux_weight: float = 0.0,
        fusion_aux_weight: float = 0.0,
        word_aux_weight: float = 0.0,
        word_aux_full_weight_epochs: int = 0,
        word_aux_decay_end_epoch: int = 0,
        aux_embedding_dim: int = 768,
        shared_branch_arcface: bool = False,
        gate_entropy_weight: float = 0.0,
        gate_entropy_decay_epochs: int = 0,
    ):
        super().__init__()
        self.arcface_weight = arcface_weight
        self.ce_weight = ce_weight
        self.sparsity_weight = sparsity_weight
        self.tile_aux_weight = tile_aux_weight
        self.glyph_aux_weight = glyph_aux_weight
        self.fusion_aux_weight = fusion_aux_weight
        self.word_aux_weight = word_aux_weight
        self.word_aux_full_weight_epochs = max(0, int(word_aux_full_weight_epochs))
        self.word_aux_decay_end_epoch = max(0, int(word_aux_decay_end_epoch))
        self.current_word_aux_weight = float(word_aux_weight)
        self.shared_branch_arcface = bool(shared_branch_arcface)
        self.gate_entropy_weight = float(gate_entropy_weight)
        self.gate_entropy_decay_epochs = int(gate_entropy_decay_epochs)
        self.current_gate_entropy_weight = float(gate_entropy_weight)
        
        # Always initialize CE loss (used when ArcFace is disabled)
        self.ce = nn.CrossEntropyLoss()
        
        # Main ArcFace loss
        if arcface_weight > 0:
            self.arcface = ArcFaceLoss(
                num_classes=num_classes,
                embedding_dim=embedding_dim,
                margin=arcface_margin,
                scale=arcface_scale,
            )
        else:
            self.arcface = None
        
        # Auxiliary losses (per modality)
        # Use separate ArcFace heads so each branch receives its own gradient signal
        if tile_aux_weight > 0:
            self.tile_arcface = ArcFaceLoss(
                num_classes=num_classes,
                embedding_dim=aux_embedding_dim,
                margin=arcface_margin,
                scale=arcface_scale,
            )
        else:
            self.tile_arcface = None

        if glyph_aux_weight > 0:
            self.glyph_arcface = ArcFaceLoss(
                num_classes=num_classes,
                embedding_dim=aux_embedding_dim,
                margin=arcface_margin,
                scale=arcface_scale,
            )
        else:
            self.glyph_arcface = None

        if fusion_aux_weight > 0:
            self.fusion_arcface = ArcFaceLoss(
                num_classes=num_classes,
                embedding_dim=aux_embedding_dim,
                margin=arcface_margin,
                scale=arcface_scale,
            )
        else:
            self.fusion_arcface = None

        if word_aux_weight > 0:
            self.word_arcface = ArcFaceLoss(
                num_classes=num_classes,
                embedding_dim=aux_embedding_dim,
                margin=arcface_margin,
                scale=arcface_scale,
            )
        else:
            self.word_arcface = None

        if self.shared_branch_arcface:
            active = [head for head in (self.tile_arcface, self.glyph_arcface, self.word_arcface) if head is not None]
            if active:
                shared = active[0]
                if self.tile_arcface is not None:
                    self.tile_arcface = shared
                if self.glyph_arcface is not None:
                    self.glyph_arcface = shared
                if self.word_arcface is not None:
                    self.word_arcface = shared

    def set_training_epoch(self, epoch: int) -> None:
        epoch_number = int(epoch) + 1
        if (
            self.word_aux_weight <= 0
            or self.word_aux_decay_end_epoch <= self.word_aux_full_weight_epochs
        ):
            self.current_word_aux_weight = float(self.word_aux_weight)
        elif epoch_number <= self.word_aux_full_weight_epochs:
            self.current_word_aux_weight = float(self.word_aux_weight)
        elif epoch_number >= self.word_aux_decay_end_epoch:
            self.current_word_aux_weight = 0.0
        else:
            decay_span = self.word_aux_decay_end_epoch - self.word_aux_full_weight_epochs
            remaining = self.word_aux_decay_end_epoch - epoch_number
            self.current_word_aux_weight = float(self.word_aux_weight) * remaining / decay_span

        if self.gate_entropy_decay_epochs <= 0:
            self.current_gate_entropy_weight = 0.0
            return
        fraction = max(0.0, 1.0 - float(epoch) / self.gate_entropy_decay_epochs)
        self.current_gate_entropy_weight = self.gate_entropy_weight * fraction

    def set_arcface_margin(self, margin: float) -> None:
        """Update every active ArcFace head to the same warmup margin."""
        for arcface_module in (
            self.arcface,
            self.tile_arcface,
            self.glyph_arcface,
            self.fusion_arcface,
            self.word_arcface,
        ):
            if arcface_module is not None:
                arcface_module.set_margin(margin)
        
    def _compute_aux_loss(self, aux_latent: torch.Tensor, labels: torch.Tensor, arcface_module: nn.Module) -> torch.Tensor:
        """
        Compute auxiliary loss safely by filtering out zero vectors (dropped modalities).
        """
        # Identify samples with non-zero embeddings
        # We use a small epsilon for robustness, though model returns exact zeros for dropped mods.
        mask = torch.norm(aux_latent, p=2, dim=1) > 1e-8
        
        if not mask.any():
            return torch.tensor(0.0, device=aux_latent.device, requires_grad=True)
        
        # Filter samples
        valid_latents = aux_latent[mask]
        valid_labels = labels[mask]
        
        # Normalize and compute loss for valid samples only
        valid_norm = torch.nn.functional.normalize(valid_latents, dim=1)
        # ArcFaceLoss handles its own normalization of weights/embeddings internally, 
        # but we pass normalized embeddings for clarity/consistency.
        loss, _ = arcface_module(valid_norm, valid_labels)
        
        # Scale back by the fraction of valid samples in the batch to keep gradient scale consistent
        return loss * (mask.sum().float() / mask.size(0))

    def forward(self, logits, features, labels, aux_latents=None):
        """
        Forward pass.
        
        Args:
            logits: [B, num_classes] - logits from classifier
            features: [B, embedding_dim] - normalized main features
            labels: [B] - class labels
            aux_latents: Dict[str, Tensor] - auxiliary latents from model
            
        Returns:
            Tuple of (total_loss, classification_loss, sparsity_loss, effective_logits, aux_loss_dict)
        """
        # 1. Main Classification loss: ArcFace or CrossEntropy
        if self.arcface_weight > 0 and self.arcface is not None:
            classification_loss, arcface_logits = self.arcface(features, labels)
            ce_loss = torch.tensor(0.0, device=features.device, requires_grad=False)
            effective_logits = arcface_logits
        else:
            ce_loss = self.ce(logits, labels)
            classification_loss = ce_loss
            effective_logits = logits
        
        # 2. Sparsity Loss
        if self.sparsity_weight > 0:
            sparsity_loss = torch.mean(torch.abs(features))
        else:
            sparsity_loss = torch.tensor(0.0, device=features.device, requires_grad=False)

        # 3. Auxiliary losses
        aux_loss_dict = {}
        total_aux_loss = torch.tensor(0.0, device=features.device)
        
        # Tile Aux
        if self.tile_aux_weight > 0 and self.tile_arcface is not None and aux_latents and 'tile' in aux_latents:
            t_aux_l = self._compute_aux_loss(aux_latents['tile'], labels, self.tile_arcface)
            aux_loss_dict['tile'] = t_aux_l
            total_aux_loss = total_aux_loss + self.tile_aux_weight * t_aux_l
        else:
            aux_loss_dict['tile'] = torch.tensor(0.0, device=features.device, requires_grad=False)
            
        # Glyph Aux
        if self.glyph_aux_weight > 0 and self.glyph_arcface is not None and aux_latents and 'glyph' in aux_latents:
            g_aux_l = self._compute_aux_loss(aux_latents['glyph'], labels, self.glyph_arcface)
            aux_loss_dict['glyph'] = g_aux_l
            total_aux_loss = total_aux_loss + self.glyph_aux_weight * g_aux_l
        else:
            aux_loss_dict['glyph'] = torch.tensor(0.0, device=features.device, requires_grad=False)

        # Fusion-token Aux (pre-head fusion representation)
        if self.fusion_aux_weight > 0 and self.fusion_arcface is not None and aux_latents and 'fusion' in aux_latents:
            f_aux_l = self._compute_aux_loss(aux_latents['fusion'], labels, self.fusion_arcface)
            aux_loss_dict['fusion'] = f_aux_l
            total_aux_loss = total_aux_loss + self.fusion_aux_weight * f_aux_l
        else:
            aux_loss_dict['fusion'] = torch.tensor(0.0, device=features.device, requires_grad=False)
            
        # Word Aux
        if self.current_word_aux_weight > 0 and self.word_arcface is not None and aux_latents and 'word' in aux_latents:
            w_aux_l = self._compute_aux_loss(aux_latents['word'], labels, self.word_arcface)
            aux_loss_dict['word'] = w_aux_l
            total_aux_loss = total_aux_loss + self.current_word_aux_weight * w_aux_l
        else:
            aux_loss_dict['word'] = torch.tensor(0.0, device=features.device, requires_grad=False)

        gate_entropy_loss = torch.tensor(0.0, device=features.device)
        if aux_latents and self.current_gate_entropy_weight > 0 and 'reliability_weights' in aux_latents:
            gates = aux_latents['reliability_weights'].float().clamp_min(1e-8)
            available = aux_latents.get('modality_available')
            if available is None:
                available = torch.ones_like(gates, dtype=torch.bool)
            else:
                available = available.bool()
            gates = torch.where(available, gates, torch.zeros_like(gates))
            n_available = available.sum(dim=1)
            entropy = -(gates * gates.clamp_min(1e-8).log()).sum(dim=1)
            eligible = n_available > 1
            if eligible.any():
                max_entropy = n_available[eligible].float().log()
                gate_entropy_loss = (max_entropy - entropy[eligible]).mean()
                total_aux_loss = total_aux_loss + self.current_gate_entropy_weight * gate_entropy_loss
        aux_loss_dict['gate_entropy'] = gate_entropy_loss

        # 4. Compute total loss
        if self.arcface_weight > 0:
            total_loss = (self.arcface_weight * classification_loss + 
                          self.sparsity_weight * sparsity_loss +
                          total_aux_loss)
        else:
            total_loss = (self.ce_weight * ce_loss + 
                          self.sparsity_weight * sparsity_loss +
                          total_aux_loss)
        
        return total_loss, classification_loss, sparsity_loss, effective_logits, aux_loss_dict
