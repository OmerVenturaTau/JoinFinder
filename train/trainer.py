import torch
from tqdm import tqdm
import torch.nn.functional as F
import wandb
from typing import Dict, List, Mapping, Tuple
import numpy as np
import gc
import math
import psutil
import os
import faulthandler
import datetime
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from sklearn.decomposition import PCA
import logging
from typing import Optional
from itertools import groupby

import torch.distributed as dist

# Set up logger
logger = logging.getLogger(__name__)
RETRIEVAL_BRANCH_NAMES = ("fusion", "tile", "glyph", "word")
CLUSTER_TEST_SOURCE_NAMES = ("clusters_images_metadata", "cluster_members")
CLUSTER_TEST_METRIC_NAMES = ("mAP", "knn_at_1", "knn_at_5", "knn_at_10")
CLUSTER_TEST_WANDB_KEYS = frozenset(
    f"test/{source_name}/{metric_name}/{branch_name}"
    for source_name in CLUSTER_TEST_SOURCE_NAMES
    for metric_name in CLUSTER_TEST_METRIC_NAMES
    for branch_name in RETRIEVAL_BRANCH_NAMES
)

# Retrieval is the deployment objective, so checkpoint selection is retrieval-only.
VAL_RETRIEVAL_CHECKPOINT_WEIGHT = 1.0
VAL_ACCURACY_CHECKPOINT_WEIGHT = 0.0

from system import (
    HEBREW_ALPHABET,
    NUM_FIXED_VALIDATION_SAMPLES,
    NUM_ATTENTION_IMAGES, 
    NUM_TOP_ERRORS_PER_EPOCH,
    ATTENTION_IMAGES_DIR,
    RUN_NAME,
    BEST_MODEL_PATH_BASE,
    TRAINING_MODE,
    TILE_SIZE,
    PCA_SAVE_DIR,
    PCA_DIMENSION,
    PCA_MAX_SAMPLES,
    PCA_MAX_COMPONENTS,
    PCA_EXPLAINED_VARIANCE_THRESHOLD,
    ENABLE_PCA_ANALYSIS,
    PCA_EVERY_EPOCHS,
    ENABLE_TRAIN_HANG_WATCHDOG,
    TRAIN_HANG_WATCHDOG_SECONDS,
    MAX_CHARS_PER_IMAGE,
    CHAR_PATCH_SIZE,
    GLYPH_ENCODER_TYPE,
    CHAR_PATCH_APPLY_IMAGENET_NORM,
    LATENT_DIM,
    ARCFACE_WEIGHT,
    ARCFACE_MARGIN,
    ARCFACE_MARGIN_WARMUP_EPOCHS,
    GRADIENT_CLIP_NORM,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    CLUSTER_PAIRS_CSV_PATH,
    CLUSTER_MEMBERS_XLSX_PATH,
    CLUSTER_MEMBERS_XLSX_SHEET,
    CLUSTER_TEST_EXPECTED_COUNTS,
    CLUSTER_PAIRS_EVAL_BATCH_SIZE,
    CLUSTERING_TILE_SIZE,
    CLUSTERING_TILE_STRIDE,
    CLUSTERING_MAX_TILES_EVAL,
    GENIZA_IMAGE_BASE,
    GENIZA_XML_BASE,
    GENIZA_XML_FILENAME_SUFFIX,
    USE_VISUAL_MOD,
    USE_CHAR_MOD,
    USE_WORD_MOD,
)
from train.dataset import ManuscriptDataset, tile_collate_with_padding, save_xml_warnings_to_file
from train.metric_learning import compute_retrieval_metrics
from utilities.checkpoint_utils import load_training_checkpoint_for_evaluation


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate(
    model,
    data_loader,
    combined_loss,
    idx2label=None,
    fixed_sample_paths=None,
    device: Optional[torch.device] = None,
    return_stats: bool = False,
):
    """
    Evaluate the model on a dataset (validation or test).
    
    Args:
        model: MultiModal model instance
        data_loader: DataLoader for evaluation dataset
        combined_loss: CombinedLoss instance (for computing loss metrics)
        idx2label: Optional dict mapping label indices to label strings
        fixed_sample_paths: Optional set of paths to track consistently across epochs
        
    Returns:
        Tuple of (avg_loss, avg_arcface_loss, accuracy, misclassified_samples, validation_batch_samples)
        - avg_loss: Average combined loss
        - avg_arcface_loss: Average ArcFace loss (or CE loss if ArcFace disabled)
        - accuracy: Classification accuracy
        - misclassified_samples: List of misclassified samples with top-3 predictions
        - validation_batch_samples: List of fixed validation samples for tracking
    """
    if data_loader is None:
        stats = {
            "eval.total_seen": 0,
            "eval.total_counted": 0,
            "eval.correct": 0,
            "eval.skipped_nonfinite_batches": 0,
            "eval.skipped_zero_latent_batches": 0,
            "eval.skipped_nonfinite_samples": 0,
            "eval.skipped_zero_latent_samples": 0,
            "eval.fixed_paths_requested": 0,
            "eval.fixed_paths_seen": 0,
            "eval.fixed_paths_matched": 0,
        }
        if return_stats:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [], [], stats
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [], []

    model.eval()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_loss = 0
    total_arcface_loss = 0
    total_tile_aux_loss = 0
    total_glyph_aux_loss = 0
    total_fusion_aux_loss = 0
    total_word_aux_loss = 0
    correct = 0
    total = 0
    total_seen = 0
    skipped_nonfinite_batches = 0
    skipped_zero_latent_batches = 0
    skipped_nonfinite_samples = 0
    skipped_zero_latent_samples = 0
    misclassified_samples = []
    validation_batch_samples = []
    fixed_matches = 0
    fixed_seen = 0
    
    # Convert fixed_sample_paths to a set for fast lookup
    def _norm_path(p) -> str:
        # Normalize paths to avoid subtle mismatches (relative vs absolute, symlinks, path objects).
        try:
            return os.path.abspath(str(p))
        except Exception:
            return str(p)

    fixed_paths_set = {_norm_path(p) for p in fixed_sample_paths} if fixed_sample_paths else set()
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if batch is None:
                continue
            tiles, valid_mask, coords, tile_page_segments, char_patches, char_valid_mask, glyph_coords, glyph_page_segments, char_class_ids, _char_metadata, words, word_metadata, labels, paths = batch
            
            tiles = tiles.to(device, non_blocking=True)
            valid_mask = valid_mask.to(device, non_blocking=True)
            coords = coords.to(device, non_blocking=True)
            tile_page_segments = tile_page_segments.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            char_patches = char_patches.to(device, non_blocking=True)
            char_valid_mask = char_valid_mask.to(device, non_blocking=True)
            glyph_coords = glyph_coords.to(device, non_blocking=True)
            glyph_page_segments = glyph_page_segments.to(device, non_blocking=True)
            char_class_ids = char_class_ids.to(device, non_blocking=True)
            batch_size = tiles.shape[0]
            total_seen += batch_size

            if idx2label and fixed_paths_set and paths:
                try:
                    fixed_seen += sum(1 for p in paths if _norm_path(p) in fixed_paths_set)
                except Exception:
                    pass
            
            batch_element_indices = torch.arange(
                tiles.shape[0], dtype=torch.long, device=tiles.device
            )
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, latent, aux_latents = model(
                    tiles=tiles,
                    tile_coords=coords,
                    tile_valid_mask=valid_mask,
                    tile_page_segments=tile_page_segments,
                    glyph_patches=char_patches,
                    glyph_coords=glyph_coords,
                    glyph_valid_mask=char_valid_mask,
                    glyph_page_segments=glyph_page_segments,
                    char_class_ids=char_class_ids,
                    words=words,
                    word_metadata=word_metadata,
                    paths=list(paths) if paths is not None else None,
                    batch_element_indices=batch_element_indices,
                    return_aux_latents=True,
                )
            # Root-cause NaNs: non-finite or near-zero latents cannot be used for
            # normalization/loss. Filter only the bad samples instead of dropping
            # the whole batch, so one corrupt sample does not erase valid eval data.
            logits_finite = torch.isfinite(logits).all(dim=1)
            latent_finite = torch.isfinite(latent).all(dim=1)
            finite_mask = logits_finite & latent_finite
            latent_norms = torch.norm(latent.float(), p=2, dim=1)
            nonzero_mask = latent_norms >= 1e-8
            valid_output_mask = finite_mask & nonzero_mask

            if not finite_mask.all():
                bad_idx = (~finite_mask).nonzero(as_tuple=True)[0]
                skipped_nonfinite_batches += 1
                skipped_nonfinite_samples += int(bad_idx.numel())
                logger.error(f"[EVAL] Non-finite model outputs at batch {i}: idx={bad_idx.tolist()[:10]}")
                if paths:
                    logger.error(f"[EVAL] Example non-finite path: {paths[bad_idx[0].item()]}")

            zero_mask = finite_mask & ~nonzero_mask
            if zero_mask.any():
                zero_idx = zero_mask.nonzero(as_tuple=True)[0]
                skipped_zero_latent_batches += 1
                skipped_zero_latent_samples += int(zero_idx.numel())
                logger.error(f"[EVAL] Zero/near-zero latent vectors at batch {i}: idx={zero_idx.tolist()[:10]}")
                if paths:
                    logger.error(f"[EVAL] Example zero-latent path: {paths[zero_idx[0].item()]}")

            if not valid_output_mask.any():
                continue
            if not valid_output_mask.all():
                valid_idx = valid_output_mask.nonzero(as_tuple=True)[0]
                logits = logits[valid_idx]
                latent = latent[valid_idx]
                labels = labels[valid_idx]
                batch_size = int(valid_idx.numel())
                if isinstance(aux_latents, dict):
                    aux_latents = {
                        name: value[valid_idx] if torch.is_tensor(value) and value.shape[0] == valid_output_mask.shape[0] else value
                        for name, value in aux_latents.items()
                    }
                if paths:
                    paths = [paths[j] for j in valid_idx.detach().cpu().tolist()]

            # Use eps to be extra safe (even though we guard above).
            normalized_latent = F.normalize(latent, dim=1, eps=1e-8)
            # All tensors should already be on the correct device (from model outputs)
            # No need to move them - loss functions don't have parameters, they work with any device
            eval_res = combined_loss(logits, normalized_latent, labels, aux_latents=aux_latents)
            loss, arcface_loss, sparsity_loss, effective_logits, aux_loss_dict = eval_res
            total_loss += loss.item() * batch_size
            total_arcface_loss += arcface_loss.item() * batch_size
            total_tile_aux_loss += aux_loss_dict.get('tile', torch.tensor(0.0)).item() * batch_size
            total_glyph_aux_loss += aux_loss_dict.get('glyph', torch.tensor(0.0)).item() * batch_size
            total_fusion_aux_loss += aux_loss_dict.get('fusion', torch.tensor(0.0)).item() * batch_size
            total_word_aux_loss += aux_loss_dict.get('word', torch.tensor(0.0)).item() * batch_size
            # IMPORTANT: use the same logits for both accuracy and table logging.
            # `effective_logits` are what ArcFace (or CE) actually optimises, so
            # predictions + probabilities should be derived from them consistently.
            preds = effective_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

            # If idx2label is provided, collect samples
            if idx2label:
                # Collect fixed validation batch samples (same 100 images each epoch)
                if fixed_paths_set:
                    for j in range(len(paths)):
                        if _norm_path(paths[j]) in fixed_paths_set:
                            true_label_idx = labels[j].item()
                            pred_label_idx = preds[j].item()
                            # Use effective_logits for probabilities so that:
                            # - `predicted_label`
                            # - `confidence`
                            # - `top_predictions`
                            # are all derived from the same source.
                            probs = F.softmax(effective_logits[j:j+1], dim=1)[0]
                            confidence = probs[pred_label_idx].item()
                            
                            # Get top 3 predictions (from effective logits)
                            top3_probs, top3_indices = torch.topk(probs, 3)
                            top3_preds_str = [idx2label.get(idx.item(), "Unknown") for idx in top3_indices]
                            top3_probs_val = top3_probs.tolist()
                            predictions_str = ", ".join([f"{pred} ({prob:.2%})" for pred, prob in zip(top3_preds_str, top3_probs_val)])
                            
                            validation_batch_samples.append({
                                "path": paths[j],
                                "true_label": idx2label.get(true_label_idx, "Unknown"),
                                "predicted_label": idx2label.get(pred_label_idx, "Unknown"),
                                "correct": "✅" if true_label_idx == pred_label_idx else "❌",
                                "confidence": confidence,
                                "top_predictions": predictions_str
                            })
                            fixed_matches += 1
                
                # Collect ALL misclassified samples (for top 100 selection later)
                incorrect_indices = (preds != labels).nonzero(as_tuple=True)[0]
                if len(incorrect_indices) > 0:
                    # Use effective_logits for probabilities to stay consistent with `preds`.
                    probs = F.softmax(effective_logits[incorrect_indices], dim=1)
                    top3_probs, top3_indices = torch.topk(probs, 3, dim=1)
                    
                    # Calculate confidence (probability of predicted class)
                    pred_probs = probs.gather(1, preds[incorrect_indices].unsqueeze(1)).squeeze(1)

                    for j, original_idx in enumerate(incorrect_indices):
                        true_label_idx = labels[original_idx].item()
                        image_path = paths[original_idx]
                        
                        # Get string representations of labels
                        true_label_str = idx2label.get(true_label_idx, "Unknown")
                        top3_preds_str = [idx2label.get(idx, "Unknown") for idx in top3_indices[j].cpu().tolist()]
                        
                        misclassified_samples.append({
                            "path": image_path,
                            "true_label": true_label_str,
                            "predicted_label": idx2label.get(preds[original_idx].item(), "Unknown"),
                            "correct": "❌",
                            "top_predictions": ", ".join([f"{pred} ({prob:.2%})" for pred, prob in zip(top3_preds_str, top3_probs[j].cpu().tolist())]),
                            "confidence": pred_probs[j].item()  # How confident was the wrong prediction
                        })

    if total == 0:
        logger.error(
            "[EVAL] No samples were evaluated (total==0). "
            f"total_seen={total_seen}, skipped_nonfinite_batches={skipped_nonfinite_batches}, "
            f"skipped_zero_latent_batches={skipped_zero_latent_batches}, "
            f"skipped_nonfinite_samples={skipped_nonfinite_samples}, "
            f"skipped_zero_latent_samples={skipped_zero_latent_samples}"
        )
        avg_loss = 0.0
        avg_arcface_loss = 0.0
        avg_tile_aux_loss = 0.0
        avg_glyph_aux_loss = 0.0
        avg_fusion_aux_loss = 0.0
        avg_word_aux_loss = 0.0
        accuracy = 0.0
    else:
        avg_loss = total_loss / total
        avg_arcface_loss = total_arcface_loss / total
        avg_tile_aux_loss = total_tile_aux_loss / total
        avg_glyph_aux_loss = total_glyph_aux_loss / total
        avg_fusion_aux_loss = total_fusion_aux_loss / total
        avg_word_aux_loss = total_word_aux_loss / total
        accuracy = correct / total

    stats = {
        "eval.total_seen": int(total_seen),
        "eval.total_counted": int(total),
        "eval.correct": int(correct),
        "eval.skipped_nonfinite_batches": int(skipped_nonfinite_batches),
        "eval.skipped_zero_latent_batches": int(skipped_zero_latent_batches),
        "eval.skipped_nonfinite_samples": int(skipped_nonfinite_samples),
        "eval.skipped_zero_latent_samples": int(skipped_zero_latent_samples),
        "eval.fixed_paths_requested": int(len(fixed_paths_set)),
        "eval.fixed_paths_seen": int(fixed_seen),
        "eval.fixed_paths_matched": int(fixed_matches),
    }
    if return_stats:
        return avg_loss, avg_arcface_loss, avg_tile_aux_loss, avg_glyph_aux_loss, avg_fusion_aux_loss, avg_word_aux_loss, accuracy, misclassified_samples, validation_batch_samples, stats
    return avg_loss, avg_arcface_loss, avg_tile_aux_loss, avg_glyph_aux_loss, avg_fusion_aux_loss, avg_word_aux_loss, accuracy, misclassified_samples, validation_batch_samples


def _forward_latent_batch(model, batch, device: torch.device, return_aux_latents: bool = False):
    """Run the standard multimodal forward path with the repository's mask convention."""
    if batch is None:
        raise ValueError("_forward_latent_batch received an empty/skipped batch")
    tiles, valid_mask, coords, tile_page_segments, char_patches, char_valid_mask, glyph_coords, glyph_page_segments, char_class_ids, _char_metadata, words, word_metadata, labels, paths = batch
    tiles = tiles.to(device, non_blocking=True)
    valid_mask = valid_mask.to(device, non_blocking=True)
    coords = coords.to(device, non_blocking=True)
    tile_page_segments = tile_page_segments.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    char_patches = char_patches.to(device, non_blocking=True)
    char_valid_mask = char_valid_mask.to(device, non_blocking=True)
    glyph_coords = glyph_coords.to(device, non_blocking=True)
    glyph_page_segments = glyph_page_segments.to(device, non_blocking=True)
    char_class_ids = char_class_ids.to(device, non_blocking=True)

    batch_element_indices = torch.arange(tiles.shape[0], dtype=torch.long, device=device)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits, latent, aux_latents = model(
            tiles=tiles,
            tile_coords=coords,
            tile_valid_mask=valid_mask,
            tile_page_segments=tile_page_segments,
            glyph_patches=char_patches,
            glyph_coords=glyph_coords,
            glyph_valid_mask=char_valid_mask,
            glyph_page_segments=glyph_page_segments,
            char_class_ids=char_class_ids,
            words=words,
            word_metadata=word_metadata,
            paths=list(paths) if paths is not None else None,
            batch_element_indices=batch_element_indices,
            return_aux_latents=return_aux_latents,
        )
    return logits, latent, aux_latents, labels, paths


@torch.no_grad()
def evaluate_retrieval(
    model,
    data_loader,
    device: torch.device,
    contrastive_loss_fn=None,
    contrastive_weight: float = 1.0,
    metric_prefix: str = "val/retrieval",
) -> Dict[str, float]:
    if data_loader is None:
        return {}
    was_training = model.training
    model.eval()
    latents = []
    labels = []
    for batch in data_loader:
        if batch is None:
            continue
        _logits, latent, _aux, batch_labels, _paths = _forward_latent_batch(
            model, batch, device=device, return_aux_latents=False
        )
        if not torch.isfinite(latent).all():
            continue
        norms = torch.norm(latent, p=2, dim=1)
        if (norms < 1e-8).any():
            continue
        latents.append(F.normalize(latent, dim=1, eps=1e-8).detach().float().cpu())
        labels.append(batch_labels.detach().long().cpu())
    if was_training:
        model.train()
    if not latents:
        return {}
    features = torch.cat(latents, dim=0)
    all_labels = torch.cat(labels, dim=0)
    metrics = compute_retrieval_metrics(features, all_labels)
    gap_for_score = metrics.same_diff_cosine_gap if math.isfinite(metrics.same_diff_cosine_gap) else 0.0
    hard_neg_p95_for_score = metrics.hard_negative_cosine_p95 if math.isfinite(metrics.hard_negative_cosine_p95) else 0.0
    checkpoint_score = (
        metrics.mean_average_precision
        + 0.05 * gap_for_score
        - 0.02 * max(0.0, hard_neg_p95_for_score)
    )
    results = {
        f"{metric_prefix}/knn_at_1": metrics.knn_at_1,
        f"{metric_prefix}/knn_at_5": metrics.knn_at_5,
        f"{metric_prefix}/knn_at_10": metrics.knn_at_10,
        f"{metric_prefix}/mAP": metrics.mean_average_precision,
        f"{metric_prefix}/same_cosine_mean": metrics.same_cosine_mean,
        f"{metric_prefix}/different_cosine_mean": metrics.different_cosine_mean,
        f"{metric_prefix}/same_diff_cosine_gap": metrics.same_diff_cosine_gap,
        f"{metric_prefix}/hard_negative_cosine_p90": metrics.hard_negative_cosine_p90,
        f"{metric_prefix}/hard_negative_cosine_p95": metrics.hard_negative_cosine_p95,
        f"{metric_prefix}/hard_negative_cosine_max": metrics.hard_negative_cosine_max,
        f"{metric_prefix}/retrieval_score": checkpoint_score,
        f"{metric_prefix}/num_queries": metrics.num_queries,
        f"{metric_prefix}/num_eval_images": metrics.num_eval_images,
    }
    if contrastive_loss_fn is not None:
        val_contrastive_loss, val_contrastive_stats = contrastive_loss_fn(features, all_labels)
        results.update({
            f"{metric_prefix}/contrastive_loss": float(contrastive_weight) * float(val_contrastive_loss.detach().cpu().item()),
            f"{metric_prefix}/contrastive_accuracy": float(val_contrastive_stats.get("top1_acc", 0.0)),
        })
    return results


def _validation_checkpoint_score(val_retrieval_metrics: Dict[str, float], val_accuracy: float) -> Optional[Tuple[float, float]]:
    """Return a validation-only checkpoint score and validation mAP."""
    if not val_retrieval_metrics:
        return None

    map_value = val_retrieval_metrics.get("val/retrieval/mAP")
    if map_value is None or not math.isfinite(float(map_value)):
        return None

    # Checkpoint selection follows the deployed retrieval objective exactly.
    retrieval_score = float(map_value)
    accuracy = float(val_accuracy) if math.isfinite(float(val_accuracy)) else 0.0
    score = (
        VAL_RETRIEVAL_CHECKPOINT_WEIGHT * retrieval_score
        + VAL_ACCURACY_CHECKPOINT_WEIGHT * accuracy
    )
    return score, float(map_value)


def _compute_pair_and_retrieval_metrics(
    features_t: torch.Tensor,
    labels_t: torch.Tensor,
    metric_prefix: str,
    k_values: tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """
    Pair separation + image retrieval metrics for a set of L2-normalised features
    and integer cluster labels. Mirrors the metric set produced by
    results_analysis/test_set/compare_clusters_pairs.py so the in-training
    "test" eval and the offline analysis script are directly comparable.
    """
    if features_t.numel() == 0:
        return {}

    metrics = compute_retrieval_metrics(features_t, labels_t)

    features = F.normalize(features_t.float(), dim=1, eps=1e-8).numpy()
    labs = labels_t.numpy()
    n = int(features.shape[0])
    results: Dict[str, float] = {
        f"{metric_prefix}/cluster_retrieval/n_images_with_vectors": float(n),
        f"{metric_prefix}/cluster_retrieval/n_eval_images": float(metrics.num_eval_images),
        f"{metric_prefix}/cluster_retrieval/n_queries_with_relevant": float(metrics.num_queries),
        f"{metric_prefix}/cluster_retrieval/mAP": metrics.mean_average_precision,
        f"{metric_prefix}/cluster_retrieval/knn_at_1": metrics.knn_at_1,
        f"{metric_prefix}/cluster_retrieval/knn_at_5": metrics.knn_at_5,
        f"{metric_prefix}/cluster_retrieval/knn_at_10": metrics.knn_at_10,
        f"{metric_prefix}/cluster_retrieval/mean_positive_rank": metrics.mean_positive_rank,
        f"{metric_prefix}/cluster_retrieval/same_cluster_mean": metrics.same_cosine_mean,
        f"{metric_prefix}/cluster_retrieval/diff_cluster_mean": metrics.different_cosine_mean,
        f"{metric_prefix}/cluster_retrieval/same_cosine_mean": metrics.same_cosine_mean,
        f"{metric_prefix}/cluster_retrieval/different_cosine_mean": metrics.different_cosine_mean,
        f"{metric_prefix}/cluster_retrieval/same_diff_cosine_gap": metrics.same_diff_cosine_gap,
        f"{metric_prefix}/cluster_retrieval/hard_negative_cosine_p90": metrics.hard_negative_cosine_p90,
        f"{metric_prefix}/cluster_retrieval/hard_negative_cosine_p95": metrics.hard_negative_cosine_p95,
        f"{metric_prefix}/cluster_retrieval/hard_negative_cosine_max": metrics.hard_negative_cosine_max,
    }

    if n < 2:
        return results

    sim = features @ features.T
    same = labs[:, None] == labs[None, :]
    eye = np.eye(n, dtype=bool)
    pair_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    y_true = same[pair_mask].astype(np.int32)
    scores = sim[pair_mask].astype(np.float64)
    same_scores = scores[y_true == 1]
    diff_scores = scores[y_true == 0]

    results.update({
        f"{metric_prefix}/pair/n_pairs": float(len(scores)),
        f"{metric_prefix}/pair/n_same_pairs": float(len(same_scores)),
        f"{metric_prefix}/pair/n_different_pairs": float(len(diff_scores)),
    })
    if len(same_scores) > 0:
        results.update({
            f"{metric_prefix}/pair/same_mean": float(np.mean(same_scores)),
            f"{metric_prefix}/pair/same_cluster_mean": float(np.mean(same_scores)),
            f"{metric_prefix}/pair/same_std": float(np.std(same_scores)),
            f"{metric_prefix}/pair/same_median": float(np.median(same_scores)),
            f"{metric_prefix}/pair/same_p10": float(np.percentile(same_scores, 10)),
            f"{metric_prefix}/pair/same_p90": float(np.percentile(same_scores, 90)),
        })
    if len(diff_scores) > 0:
        results.update({
            f"{metric_prefix}/pair/different_mean": float(np.mean(diff_scores)),
            f"{metric_prefix}/pair/diff_cluster_mean": float(np.mean(diff_scores)),
            f"{metric_prefix}/pair/different_std": float(np.std(diff_scores)),
            f"{metric_prefix}/pair/different_median": float(np.median(diff_scores)),
            f"{metric_prefix}/pair/different_p10": float(np.percentile(diff_scores, 10)),
            f"{metric_prefix}/pair/different_p90": float(np.percentile(diff_scores, 90)),
        })
    if len(same_scores) > 0 and len(diff_scores) > 0:
        results[f"{metric_prefix}/pair/mean_gap_same_minus_different"] = (
            float(np.mean(same_scores) - np.mean(diff_scores))
        )

    if len(np.unique(y_true)) >= 2:
        from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

        results[f"{metric_prefix}/pair/roc_auc"] = float(roc_auc_score(y_true, scores))
        results[f"{metric_prefix}/pair/pr_auc_average_precision"] = float(average_precision_score(y_true, scores))
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        if len(thresholds) > 0:
            p = precision[:-1]
            r = recall[:-1]
            f1 = (2 * p * r) / np.clip(p + r, 1e-12, None)
            best_idx = int(np.nanargmax(f1))
            pred = scores >= thresholds[best_idx]
            results.update({
                f"{metric_prefix}/pair/best_f1_threshold": float(thresholds[best_idx]),
                f"{metric_prefix}/pair/best_f1": float(f1[best_idx]),
                f"{metric_prefix}/pair/best_f1_precision": float(p[best_idx]),
                f"{metric_prefix}/pair/best_f1_recall": float(r[best_idx]),
                f"{metric_prefix}/pair/best_f1_accuracy": float((pred.astype(np.int32) == y_true).mean()),
            })

    scored = sim.copy()
    scored[eye] = -np.inf
    for k in k_values:
        recalls = []
        hits = []
        kk = min(k, max(0, n - 1))
        if kk <= 0:
            continue
        for i in range(n):
            relevant = same[i] & ~eye[i]
            relevant_total = int(relevant.sum())
            if relevant_total <= 0:
                continue
            top_idx = np.argsort(scored[i])[::-1][:kk]
            rel_in_top_k = int(relevant[top_idx].sum())
            recalls.append(rel_in_top_k / relevant_total)
            hits.append(float(rel_in_top_k > 0))
        if recalls:
            results[f"{metric_prefix}/cluster_retrieval/recall@{k}"] = float(np.mean(recalls))
            results[f"{metric_prefix}/cluster_retrieval/hit@{k}"] = float(np.mean(hits))

    return results


@torch.no_grad()
def evaluate_cluster_diagnostics(
    model,
    data_loader,
    device: torch.device,
    prefix: str = "clusters/test",
    k_values: tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """
    Compute compare_clusters_pairs-style metrics on eval latents.

    During training we use the dataset labels as cluster IDs. This mirrors the
    pair separation and image retrieval metrics from
    results_analysis/test_set/compare_clusters_pairs.py without requiring the
    external clusters metadata file.
    """
    if data_loader is None:
        return {}

    was_training = model.training
    model.eval()
    latents = []
    labels = []
    branch_latents: Dict[str, List[torch.Tensor]] = {name: [] for name in RETRIEVAL_BRANCH_NAMES}
    branch_labels: Dict[str, List[torch.Tensor]] = {name: [] for name in RETRIEVAL_BRANCH_NAMES}
    reliability_rows = []
    for batch in data_loader:
        if batch is None:
            continue
        _logits, latent, aux_latents, batch_labels, _paths = _forward_latent_batch(
            model, batch, device=device, return_aux_latents=True
        )
        batch_labels_cpu = batch_labels.detach().long().cpu()

        main_valid = torch.isfinite(latent).all(dim=1) & (torch.norm(latent, p=2, dim=1) >= 1e-8)
        if main_valid.any():
            latents.append(F.normalize(latent[main_valid], dim=1, eps=1e-8).detach().float().cpu())
            labels.append(batch_labels_cpu[main_valid.detach().cpu()])

        if isinstance(aux_latents, dict):
            if aux_latents.get("reliability_weights") is not None:
                reliability_rows.append(aux_latents["reliability_weights"].detach().float().cpu())
            for branch_name in RETRIEVAL_BRANCH_NAMES:
                branch_latent = aux_latents.get(branch_name)
                if branch_latent is None or branch_latent.numel() == 0:
                    continue
                branch_valid = (
                    torch.isfinite(branch_latent).all(dim=1)
                    & (torch.norm(branch_latent, p=2, dim=1) >= 1e-8)
                )
                if branch_valid.any():
                    branch_latents[branch_name].append(
                        F.normalize(branch_latent[branch_valid], dim=1, eps=1e-8).detach().float().cpu()
                    )
                    branch_labels[branch_name].append(batch_labels_cpu[branch_valid.detach().cpu()])
    if was_training:
        model.train()
    if not latents:
        return {}

    features_t = torch.cat(latents, dim=0)
    labels_t = torch.cat(labels, dim=0)
    results = _compute_pair_and_retrieval_metrics(features_t, labels_t, prefix, k_values)
    if reliability_rows:
        reliability = torch.cat(reliability_rows, dim=0)
        names = [name for name, enabled in (("tile", USE_VISUAL_MOD), ("glyph", USE_CHAR_MOD), ("word", USE_WORD_MOD)) if enabled]
        for idx, name in enumerate(names):
            results[f"{prefix}/fusion/reliability_{name}_mean"] = float(reliability[:, idx].mean())
    for branch_name in RETRIEVAL_BRANCH_NAMES:
        if not branch_latents[branch_name]:
            continue
        branch_features_t = torch.cat(branch_latents[branch_name], dim=0)
        branch_labels_t = torch.cat(branch_labels[branch_name], dim=0)
        results.update(
            _compute_pair_and_retrieval_metrics(
                branch_features_t,
                branch_labels_t,
                f"{prefix}/branches/{branch_name}",
                k_values,
            )
        )

    return results


def _resolve_cluster_pairs_csv(csv_path: Optional[str] = None) -> Optional[str]:
    """Resolve CLUSTER_PAIRS_CSV_PATH to an absolute path that exists, or None."""
    candidate = csv_path if csv_path else CLUSTER_PAIRS_CSV_PATH
    if not candidate:
        return None
    if not os.path.isabs(candidate):
        # Resolve relative to the project root (parent of this file's package).
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidate = os.path.join(project_root, candidate)
    return candidate if os.path.exists(candidate) else None


def _resolve_project_path(path: str) -> str:
    """Resolve a repository-relative data path without depending on the CWD."""
    if os.path.isabs(path):
        return os.path.normpath(path)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.normpath(os.path.join(project_root, path))


def load_cluster_test_members(
    source_name: str,
    source_path: str,
    *,
    sheet_name: Optional[str] = None,
    expected_rows: Optional[int] = None,
    expected_clusters: Optional[int] = None,
) -> "object":
    """Load and strictly validate a fixed retrieval test as member-level rows.

    The returned DataFrame always has ``image_path``, ``xml_path``, and
    ``cluster_id`` columns. The Geniza ``cluster_members.xlsx`` schema is
    converted in memory; derived pair workbooks and stored similarity scores
    are deliberately not consumed.
    """
    import pandas as pd

    resolved_path = _resolve_project_path(source_path)
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(
            f"Cluster test {source_name!r} was not found: {resolved_path}"
        )

    suffix = os.path.splitext(resolved_path)[1].lower()
    if suffix == ".csv":
        df = pd.read_csv(resolved_path)
        required = {"image_path", "xml_path", "cluster_id"}
        id_column = "picture_id" if "picture_id" in df.columns else "image_path"
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(resolved_path, sheet_name=sheet_name or 0)
        required = {"cluster_id", "image_id", "manuscript_id", "relative_path"}
        id_column = "image_id"
    else:
        raise ValueError(
            f"Cluster test {source_name!r} must be CSV or Excel, got {suffix!r}"
        )

    missing_columns = sorted(required.difference(df.columns))
    if missing_columns:
        raise ValueError(
            f"Cluster test {source_name!r} is missing required columns: {missing_columns}"
        )
    if df.empty:
        raise ValueError(f"Cluster test {source_name!r} has no rows")

    df = df.copy()
    if suffix in {".xlsx", ".xls"}:
        image_paths = []
        xml_paths = []
        for row in df.itertuples(index=False):
            relative = os.path.normpath(str(row.relative_path).strip())
            relative_parts = list(relative.split(os.sep))
            if relative_parts and relative_parts[0] == os.path.basename(
                os.path.normpath(GENIZA_IMAGE_BASE)
            ):
                relative_parts = relative_parts[1:]
            if len(relative_parts) < 3:
                raise ValueError(
                    f"Cluster test {source_name!r} has an invalid relative_path: "
                    f"{row.relative_path!r}"
                )
            manuscript_id = str(row.manuscript_id).strip()
            if relative_parts[0] != manuscript_id:
                raise ValueError(
                    f"Cluster test {source_name!r} manuscript mismatch: "
                    f"row={manuscript_id!r}, path={relative_parts[0]!r}"
                )
            image_path = os.path.normpath(
                os.path.join(GENIZA_IMAGE_BASE, *relative_parts)
            )
            parent_directory = relative_parts[-2]
            picture_stem = os.path.splitext(relative_parts[-1])[0]
            xml_path = os.path.normpath(
                os.path.join(
                    GENIZA_XML_BASE,
                    manuscript_id,
                    parent_directory,
                    f"{picture_stem}{GENIZA_XML_FILENAME_SUFFIX}",
                )
            )
            image_paths.append(image_path)
            xml_paths.append(xml_path)
        df["image_path"] = image_paths
        df["xml_path"] = xml_paths

    for column in ("image_path", "xml_path", "cluster_id", id_column):
        if df[column].isna().any():
            raise ValueError(
                f"Cluster test {source_name!r} has null values in {column!r}"
            )
    df["image_path"] = df["image_path"].astype(str).str.strip().map(os.path.normpath)
    df["xml_path"] = df["xml_path"].astype(str).str.strip().map(os.path.normpath)
    df["cluster_id"] = df["cluster_id"].astype(str).str.strip()
    if (df[["image_path", "xml_path", "cluster_id"]].apply(lambda col: col.str.len()) == 0).any().any():
        raise ValueError(f"Cluster test {source_name!r} contains blank required values")

    duplicate_paths = df.loc[df["image_path"].duplicated(keep=False), "image_path"].unique()
    if len(duplicate_paths):
        raise ValueError(
            f"Cluster test {source_name!r} has duplicate image paths: "
            f"{duplicate_paths[:3].tolist()}"
        )
    duplicate_ids = df.loc[df[id_column].duplicated(keep=False), id_column].unique()
    if len(duplicate_ids):
        raise ValueError(
            f"Cluster test {source_name!r} has duplicate image IDs in {id_column!r}: "
            f"{duplicate_ids[:3].tolist()}"
        )

    cluster_sizes = df.groupby("cluster_id", sort=False).size()
    singleton_clusters = cluster_sizes[cluster_sizes < 2]
    if not singleton_clusters.empty:
        raise ValueError(
            f"Cluster test {source_name!r} contains clusters with fewer than two members: "
            f"{singleton_clusters.index[:5].tolist()}"
        )
    actual_rows = int(len(df))
    actual_clusters = int(cluster_sizes.size)
    if expected_rows is not None and actual_rows != int(expected_rows):
        raise ValueError(
            f"Cluster test {source_name!r} expected {expected_rows} rows, found {actual_rows}"
        )
    if expected_clusters is not None and actual_clusters != int(expected_clusters):
        raise ValueError(
            f"Cluster test {source_name!r} expected {expected_clusters} clusters, "
            f"found {actual_clusters}"
        )

    missing_images = [path for path in df["image_path"] if not os.path.isfile(path)]
    missing_xml = [path for path in df["xml_path"] if not os.path.isfile(path)]
    if missing_images or missing_xml:
        raise FileNotFoundError(
            f"Cluster test {source_name!r} failed path preflight: "
            f"missing_images={len(missing_images)} sample={missing_images[:2]}, "
            f"missing_xml={len(missing_xml)} sample={missing_xml[:2]}"
        )

    normalized = df[["image_path", "xml_path", "cluster_id"]].reset_index(drop=True)
    normalized.attrs.update({
        "source_name": source_name,
        "source_path": resolved_path,
        "num_rows": actual_rows,
        "num_clusters": actual_clusters,
    })
    logger.info(
        "[CLUSTER TEST][PREFLIGHT] source=%s rows=%d clusters=%d images=%d xml=%d path=%s",
        source_name,
        actual_rows,
        actual_clusters,
        actual_rows,
        actual_rows,
        resolved_path,
    )
    return normalized


def build_cluster_test_loader(
    members_df,
    *,
    device: torch.device,
    batch_size: int = CLUSTER_PAIRS_EVAL_BATCH_SIZE,
    num_workers: int = 2,
):
    """Build a reusable, deterministic loader for a normalized member table."""
    from torch.utils.data import DataLoader as _DataLoader
    from torchvision import transforms as _transforms

    cluster_ids = members_df["cluster_id"].astype(str).tolist()
    unique_cluster_ids = list(dict.fromkeys(cluster_ids))
    label2idx = {cluster_id: idx for idx, cluster_id in enumerate(unique_cluster_ids)}
    transform = _transforms.Compose([
        _transforms.ToTensor(),
        _transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])
    dataset = ManuscriptDataset(
        members_df["image_path"].tolist(),
        cluster_ids,
        transform,
        label2idx,
        xml_paths=members_df["xml_path"].tolist(),
        patch_size=CLUSTERING_TILE_SIZE,
        stride=CLUSTERING_TILE_STRIDE,
        max_tiles_per_image=CLUSTERING_MAX_TILES_EVAL,
        split="test",
    )
    dataset.cluster_test_name = members_df.attrs["source_name"]
    dataset.cluster_test_num_rows = int(members_df.attrs["num_rows"])
    dataset.cluster_test_num_clusters = int(members_df.attrs["num_clusters"])
    return _DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(int(num_workers) > 0),
        collate_fn=tile_collate_with_padding,
    )


def build_cluster_test_loaders(device: torch.device) -> Dict[str, object]:
    """Preflight and construct both fixed cluster retrieval test loaders."""
    specs = (
        ("clusters_images_metadata", CLUSTER_PAIRS_CSV_PATH, None),
        ("cluster_members", CLUSTER_MEMBERS_XLSX_PATH, CLUSTER_MEMBERS_XLSX_SHEET),
    )
    loaders = {}
    for source_name, source_path, sheet_name in specs:
        expected_rows, expected_clusters = CLUSTER_TEST_EXPECTED_COUNTS[source_name]
        members_df = load_cluster_test_members(
            source_name,
            source_path,
            sheet_name=sheet_name,
            expected_rows=expected_rows,
            expected_clusters=expected_clusters,
        )
        loaders[source_name] = build_cluster_test_loader(members_df, device=device)
    return loaders


def _compute_cluster_branch_maps(
    branch_features: Mapping[str, torch.Tensor],
    labels: torch.Tensor | Mapping[str, torch.Tensor],
    source_name: str,
) -> Dict[str, float]:
    """Return only mAP and KNN@1/5/10 for each retrieval representation.

    ``labels`` may be branch-specific because glyph and word representations
    legitimately have fewer rows when an image contains no usable OCR evidence.
    """
    missing = [name for name in RETRIEVAL_BRANCH_NAMES if name not in branch_features]
    if missing:
        raise RuntimeError(
            f"Cluster test {source_name!r} did not produce branches: {missing}"
        )
    results = {}
    for branch_name in RETRIEVAL_BRANCH_NAMES:
        features = branch_features[branch_name]
        branch_labels = labels[branch_name] if isinstance(labels, Mapping) else labels
        if features.shape[0] != branch_labels.shape[0]:
            raise RuntimeError(
                f"Cluster test {source_name!r}/{branch_name} coverage mismatch: "
                f"features={features.shape[0]} labels={branch_labels.shape[0]}"
            )
        metric = compute_retrieval_metrics(features, branch_labels)
        if metric.num_queries != int(branch_labels.shape[0]):
            logger.warning(
                "[CLUSTER TEST][QUERY COVERAGE] source=%s branch=%s "
                "queries_with_relevant=%d embedded_rows=%d; rows left without a "
                "same-cluster peer after modality filtering are not retrieval queries",
                source_name,
                branch_name,
                metric.num_queries,
                int(branch_labels.shape[0]),
            )
        branch_metrics = {
            "mAP": float(metric.mean_average_precision),
            "knn_at_1": float(metric.knn_at_1),
            "knn_at_5": float(metric.knn_at_5),
            "knn_at_10": float(metric.knn_at_10),
        }
        non_finite = [name for name, value in branch_metrics.items() if not math.isfinite(value)]
        if non_finite:
            raise RuntimeError(
                f"Cluster test {source_name!r}/{branch_name} produced non-finite "
                f"metrics: {non_finite}"
            )
        for metric_name, value in branch_metrics.items():
            results[f"test/{source_name}/{metric_name}/{branch_name}"] = value
    return results


def _validate_cluster_test_wandb_metrics(metrics: Mapping[str, float]) -> None:
    """Allow only mAP and KNN@1/5/10 for four branches and two fixed tests."""
    actual_keys = set(metrics)
    if actual_keys != CLUSTER_TEST_WANDB_KEYS:
        raise RuntimeError(
            "Cluster-test W&B fields must contain only branch mAP and KNN@1/5/10 "
            "values for each fixed source: "
            f"missing={sorted(CLUSTER_TEST_WANDB_KEYS - actual_keys)}, "
            f"unexpected={sorted(actual_keys - CLUSTER_TEST_WANDB_KEYS)}"
        )


@torch.no_grad()
def evaluate_cluster_test_map(
    model,
    data_loader,
    *,
    device: torch.device,
    source_name: str,
) -> Dict[str, float]:
    """Evaluate a fixed cluster collection with branch-specific OCR coverage.

    Missing glyph/word evidence is skipped and logged for that branch. Missing
    required fusion/tile embeddings and all non-finite outputs remain fatal.
    """
    expected_rows = int(getattr(data_loader.dataset, "cluster_test_num_rows", len(data_loader.dataset)))
    expected_clusters = int(getattr(data_loader.dataset, "cluster_test_num_clusters", 0))
    was_training = model.training
    model.eval()
    base_model = model.module if hasattr(model, "module") else model
    prev_mod_dropout = getattr(base_model, "modality_dropout_enabled", None)
    prev_token_subs = getattr(base_model, "token_subsample_enabled", None)
    if hasattr(base_model, "modality_dropout_enabled"):
        base_model.modality_dropout_enabled = False
    if hasattr(base_model, "token_subsample_enabled"):
        base_model.token_subsample_enabled = False

    branch_rows: Dict[str, List[torch.Tensor]] = {
        name: [] for name in RETRIEVAL_BRANCH_NAMES
    }
    branch_label_rows: Dict[str, List[torch.Tensor]] = {
        name: [] for name in RETRIEVAL_BRANCH_NAMES
    }
    seen_rows = 0
    unavailable_counts = {name: 0 for name in RETRIEVAL_BRANCH_NAMES}
    nonfinite_counts = {name: 0 for name in RETRIEVAL_BRANCH_NAMES}
    try:
        for batch in data_loader:
            if batch is None:
                continue
            _logits, latent, aux_latents, labels, paths = _forward_latent_batch(
                model, batch, device=device, return_aux_latents=True
            )
            if not isinstance(aux_latents, dict):
                raise RuntimeError(
                    f"Cluster test {source_name!r} did not receive auxiliary branch latents"
                )
            batch_features = {"fusion": latent}
            batch_features.update({
                name: aux_latents.get(name) for name in ("tile", "glyph", "word")
            })
            batch_size = int(labels.shape[0])
            seen_rows += batch_size
            labels_cpu = labels.detach().long().cpu()
            for branch_name in RETRIEVAL_BRANCH_NAMES:
                features = batch_features.get(branch_name)
                if features is None or features.ndim != 2 or features.shape[0] != batch_size:
                    raise RuntimeError(
                        f"Cluster test {source_name!r} has missing/invalid {branch_name!r} "
                        f"output for a batch of {batch_size} rows"
                    )
                finite = torch.isfinite(features).all(dim=1)
                nonzero = torch.norm(features.float(), p=2, dim=1) >= 1e-8
                valid = finite & nonzero
                unavailable = finite & ~nonzero
                nonfinite = ~finite
                unavailable_counts[branch_name] += int(unavailable.sum().item())
                nonfinite_counts[branch_name] += int(nonfinite.sum().item())
                if valid.any():
                    branch_rows[branch_name].append(
                        F.normalize(features[valid].float(), dim=1, eps=1e-8).detach().cpu()
                    )
                    branch_label_rows[branch_name].append(
                        labels_cpu[valid.detach().cpu()]
                    )
                unavailable_indices = unavailable.nonzero(as_tuple=True)[0].detach().cpu().tolist()
                for index in unavailable_indices:
                    path = paths[index] if paths is not None else "<unknown>"
                    logger.warning(
                        "[CLUSTER TEST][SKIP] source=%s branch=%s "
                        "reason=no_usable_modality path=%s",
                        source_name,
                        branch_name,
                        path,
                    )
                if nonfinite.any():
                    bad_indices = nonfinite.nonzero(as_tuple=True)[0].detach().cpu().tolist()
                    bad_paths = [paths[i] for i in bad_indices[:3]] if paths is not None else []
                    logger.error(
                        "[CLUSTER TEST] source=%s branch=%s nonfinite=%d paths=%s",
                        source_name,
                        branch_name,
                        len(bad_indices),
                        bad_paths,
                    )
    finally:
        if hasattr(base_model, "modality_dropout_enabled") and prev_mod_dropout is not None:
            base_model.modality_dropout_enabled = prev_mod_dropout
        if hasattr(base_model, "token_subsample_enabled") and prev_token_subs is not None:
            base_model.token_subsample_enabled = prev_token_subs
        if was_training:
            model.train()

    logger.info(
        "[CLUSTER TEST][COVERAGE] source=%s expected=%d seen=%d clusters=%d "
        "embedded=%s skipped_unavailable=%s nonfinite=%s",
        source_name,
        expected_rows,
        seen_rows,
        expected_clusters,
        {name: seen_rows - unavailable_counts[name] - nonfinite_counts[name]
         for name in RETRIEVAL_BRANCH_NAMES},
        unavailable_counts,
        nonfinite_counts,
    )
    required_unavailable = {
        name: unavailable_counts[name] for name in ("fusion", "tile")
        if unavailable_counts[name]
    }
    if seen_rows != expected_rows or required_unavailable or any(nonfinite_counts.values()):
        raise RuntimeError(
            f"Cluster test {source_name!r} produced partial coverage: "
            f"expected={expected_rows}, seen={seen_rows}, "
            f"required_unavailable={required_unavailable}, "
            f"nonfinite={nonfinite_counts}"
        )
    if any(not rows for rows in branch_rows.values()):
        raise RuntimeError(f"Cluster test {source_name!r} produced no usable embeddings")

    branch_features_t = {
        name: torch.cat(rows, dim=0) for name, rows in branch_rows.items()
    }
    branch_labels_t = {
        name: torch.cat(rows, dim=0) for name, rows in branch_label_rows.items()
    }
    metrics = _compute_cluster_branch_maps(
        branch_features_t, branch_labels_t, source_name
    )
    logger.info(
        "[CLUSTER TEST][mAP] source=%s fusion=%.6f tile=%.6f glyph=%.6f word=%.6f",
        source_name,
        metrics[f"test/{source_name}/mAP/fusion"],
        metrics[f"test/{source_name}/mAP/tile"],
        metrics[f"test/{source_name}/mAP/glyph"],
        metrics[f"test/{source_name}/mAP/word"],
    )
    return metrics


def evaluate_cluster_test_suite(
    model,
    cluster_test_loaders: Mapping[str, object],
    *,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate every configured fixed cluster collection exactly once."""
    results: Dict[str, float] = {}
    for source_name, loader in cluster_test_loaders.items():
        source_metrics = evaluate_cluster_test_map(
            model,
            loader,
            device=device,
            source_name=source_name,
        )
        overlap = results.keys() & source_metrics.keys()
        if overlap:
            raise RuntimeError(f"Duplicate cluster-test metric keys: {sorted(overlap)}")
        results.update(source_metrics)
    _validate_cluster_test_wandb_metrics(results)
    return results


@torch.no_grad()
def evaluate_cluster_pairs_csv(
    model,
    csv_path: str,
    device: torch.device,
    prefix: str = "test",
    batch_size: int = CLUSTER_PAIRS_EVAL_BATCH_SIZE,
    k_values: tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """
    Per-epoch "test" evaluation that mirrors
    results_analysis/test_set/compare_clusters_pairs.py on the curated cluster
    metadata CSV (manuscript_id, picture_id, image_path, xml_path, cluster_id).

    Returns the same pair separation + image retrieval metrics that
    compare_clusters_pairs.py prints, under ``prefix/pair/...`` and
    ``prefix/cluster_retrieval/...`` keys.
    """
    if not csv_path or not os.path.exists(csv_path):
        logger.warning(
            "[CLUSTER PAIRS] CSV not found at %s — skipping cluster-pairs test eval.",
            csv_path,
        )
        return {}

    # Local imports keep the heavy dependencies out of module-level load.
    import pandas as pd
    from torch.utils.data import DataLoader as _DataLoader
    from torchvision import transforms as _transforms

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        logger.exception("[CLUSTER PAIRS] Failed to read %s: %s", csv_path, exc)
        return {}

    required_cols = {"image_path", "cluster_id"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.error("[CLUSTER PAIRS] %s missing required columns %s — skipping.", csv_path, sorted(missing))
        return {}

    df = df.copy()
    df["image_path"] = df["image_path"].astype(str).str.strip()
    df = df[df["image_path"].str.len() > 0].reset_index(drop=True)
    if df.empty:
        logger.warning("[CLUSTER PAIRS] %s has no usable rows after filtering — skipping.", csv_path)
        return {}

    image_paths: List[str] = df["image_path"].tolist()
    if "xml_path" in df.columns:
        xml_paths = [
            (str(p).strip() if (pd.notna(p) and str(p).strip()) else None)
            for p in df["xml_path"].tolist()
        ]
    else:
        xml_paths = [None] * len(image_paths)

    # ManuscriptDataset wants integer labels via label2idx; we use a synthetic
    # placeholder ("0") so the model's classification head sees a valid index.
    # The real cluster_id is tracked separately and only used for metrics.
    dummy_labels = ["0"] * len(image_paths)
    label2idx = {"0": 0}

    transform = _transforms.Compose([
        _transforms.ToTensor(),
        _transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])

    dataset = ManuscriptDataset(
        image_paths,
        dummy_labels,
        transform,
        label2idx,
        xml_paths=xml_paths,
        patch_size=CLUSTERING_TILE_SIZE,
        stride=CLUSTERING_TILE_STRIDE,
        max_tiles_per_image=CLUSTERING_MAX_TILES_EVAL,
        split="test",
    )
    loader = _DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        collate_fn=tile_collate_with_padding,
    )

    # Stable cluster_id -> integer index so labels_t is a torch.long.
    cluster_id_to_idx: Dict[str, int] = {}
    def _cluster_idx(cid) -> int:
        key = str(cid)
        if key not in cluster_id_to_idx:
            cluster_id_to_idx[key] = len(cluster_id_to_idx)
        return cluster_id_to_idx[key]

    # Map image_path (raw and normpath) -> cluster index, for robust lookup.
    cluster_idx_by_path: Dict[str, int] = {}
    for _, row in df.iterrows():
        raw = str(row["image_path"]).strip()
        cid = _cluster_idx(row["cluster_id"])
        cluster_idx_by_path[raw] = cid
        cluster_idx_by_path[os.path.normpath(raw)] = cid

    # Eval-time forward: disable train-only regularizers; restore on exit.
    was_training = model.training
    model.eval()
    base_model = model.module if hasattr(model, "module") else model
    prev_mod_dropout = getattr(base_model, "modality_dropout_enabled", None)
    prev_token_subs = getattr(base_model, "token_subsample_enabled", None)
    if hasattr(base_model, "modality_dropout_enabled"):
        base_model.modality_dropout_enabled = False
    if hasattr(base_model, "token_subsample_enabled"):
        base_model.token_subsample_enabled = False

    latents_per_image: List[torch.Tensor] = []
    cluster_indices: List[int] = []
    branch_latents: Dict[str, List[torch.Tensor]] = {name: [] for name in RETRIEVAL_BRANCH_NAMES}
    branch_cluster_indices: Dict[str, List[int]] = {name: [] for name in RETRIEVAL_BRANCH_NAMES}
    skipped_nonfinite = 0
    skipped_zero_norm = 0

    try:
        for batch in loader:
            if batch is None:
                continue
            _logits, latent, aux_latents, _labels_t, batch_paths = _forward_latent_batch(
                model, batch, device=device, return_aux_latents=True
            )
            if latent is None or latent.numel() == 0:
                continue

            # Per-row finite + nonzero filter, matching evaluate()/eval semantics.
            finite_mask = torch.isfinite(latent).all(dim=1)
            norms = torch.norm(latent.float(), p=2, dim=1)
            nonzero_mask = norms >= 1e-8
            valid_mask = finite_mask & nonzero_mask
            if not valid_mask.any():
                skipped_nonfinite += int((~finite_mask).sum().item())
                skipped_zero_norm += int((finite_mask & ~nonzero_mask).sum().item())
                continue
            skipped_nonfinite += int((~finite_mask).sum().item())
            skipped_zero_norm += int((finite_mask & ~nonzero_mask).sum().item())

            valid_idx = valid_mask.nonzero(as_tuple=True)[0].tolist()
            normalized = F.normalize(latent[valid_mask], dim=1, eps=1e-8).detach().float().cpu()

            for local_i, vec in zip(valid_idx, normalized):
                if batch_paths is None or local_i >= len(batch_paths):
                    continue
                path = str(batch_paths[local_i]).strip()
                cid = cluster_idx_by_path.get(path)
                if cid is None:
                    cid = cluster_idx_by_path.get(os.path.normpath(path))
                if cid is None:
                    # Image was forwarded but is not in the CSV (shouldn't happen
                    # because the dataset was built from the CSV) — skip it.
                    continue
                latents_per_image.append(vec)
                cluster_indices.append(cid)

            if isinstance(aux_latents, dict) and batch_paths is not None:
                for branch_name in RETRIEVAL_BRANCH_NAMES:
                    branch_tensor = aux_latents.get(branch_name)
                    if branch_tensor is None or branch_tensor.numel() == 0:
                        continue
                    branch_finite = torch.isfinite(branch_tensor).all(dim=1)
                    branch_norms = torch.norm(branch_tensor.float(), p=2, dim=1)
                    branch_valid = branch_finite & (branch_norms >= 1e-8)
                    if not branch_valid.any():
                        continue
                    branch_normalized = F.normalize(
                        branch_tensor[branch_valid],
                        dim=1,
                        eps=1e-8,
                    ).detach().float().cpu()
                    branch_valid_idx = branch_valid.nonzero(as_tuple=True)[0].tolist()
                    for local_i, vec in zip(branch_valid_idx, branch_normalized):
                        if local_i >= len(batch_paths):
                            continue
                        path = str(batch_paths[local_i]).strip()
                        cid = cluster_idx_by_path.get(path)
                        if cid is None:
                            cid = cluster_idx_by_path.get(os.path.normpath(path))
                        if cid is None:
                            continue
                        branch_latents[branch_name].append(vec)
                        branch_cluster_indices[branch_name].append(cid)
    finally:
        if hasattr(base_model, "modality_dropout_enabled") and prev_mod_dropout is not None:
            base_model.modality_dropout_enabled = prev_mod_dropout
        if hasattr(base_model, "token_subsample_enabled") and prev_token_subs is not None:
            base_model.token_subsample_enabled = prev_token_subs
        if was_training:
            model.train()

    if not latents_per_image:
        logger.warning(
            "[CLUSTER PAIRS] No usable latents were produced (rows=%d, skipped_nonfinite=%d, skipped_zero_norm=%d).",
            len(df), skipped_nonfinite, skipped_zero_norm,
        )
        return {}

    features_t = torch.stack(latents_per_image, dim=0)
    labels_t = torch.tensor(cluster_indices, dtype=torch.long)
    results = _compute_pair_and_retrieval_metrics(features_t, labels_t, prefix, k_values)
    for branch_name in RETRIEVAL_BRANCH_NAMES:
        if not branch_latents[branch_name]:
            continue
        branch_features_t = torch.stack(branch_latents[branch_name], dim=0)
        branch_labels_t = torch.tensor(branch_cluster_indices[branch_name], dtype=torch.long)
        results.update(
            _compute_pair_and_retrieval_metrics(
                branch_features_t,
                branch_labels_t,
                f"{prefix}/branches/{branch_name}",
                k_values,
            )
        )

    # Surface a few diagnostic counters so the wandb panel makes it obvious
    # this eval ran (and on how many rows).
    results[f"{prefix}/n_rows_csv"] = float(len(df))
    results[f"{prefix}/n_rows_evaluated"] = float(len(latents_per_image))
    results[f"{prefix}/n_unique_clusters"] = float(len(cluster_id_to_idx))
    results[f"{prefix}/skipped_nonfinite"] = float(skipped_nonfinite)
    results[f"{prefix}/skipped_zero_norm"] = float(skipped_zero_norm)
    return results


# ============================================================================
# ATTENTION VISUALIZATION FUNCTIONS
# ============================================================================

def extract_attention_from_model(model, batch):
    """
    Extract attention weights from the model for all modalities.
    
    Args:
        model: MultiModal model instance
        batch: Full multi-modal batch tuple
        
    Returns:
        Tuple of (attention_dict, processed_batch_data) where:
        - attention_dict: Dict with keys 'visual', 'character', 'word' containing attention tensors
        - processed_batch_data: Dict with processed tensors (tiles, valid_mask, coords, etc.)
    """
    if batch is None:
        return {}, {}

    # Unwrap DDP for custom methods like forward_with_attention()
    base_model = model.module if hasattr(model, "module") else model
    was_training = base_model.training
    base_model.eval()
    # Unpack full multi-modal batch
    tiles, valid_mask, coords, tile_page_segments, char_patches, char_valid_mask, glyph_coords, glyph_page_segments, char_class_ids, char_metadata, words, word_metadata, labels, paths = batch
    device = next(base_model.parameters()).device
    tiles = tiles.to(device, non_blocking=True)
    valid_mask = valid_mask.to(device, non_blocking=True)
    coords = coords.to(device, non_blocking=True)
    tile_page_segments = tile_page_segments.to(device, non_blocking=True)
    char_patches = char_patches.to(device, non_blocking=True)
    char_valid_mask = char_valid_mask.to(device, non_blocking=True)
    glyph_coords = glyph_coords.to(device, non_blocking=True)
    glyph_page_segments = glyph_page_segments.to(device, non_blocking=True)
    char_class_ids = char_class_ids.to(device, non_blocking=True)

    batch_element_indices = torch.arange(
        tiles.shape[0], dtype=torch.long, device=tiles.device
    )
    with torch.no_grad():
        _, _, attention_dict = base_model.forward_with_attention(
            tiles=tiles,
            tile_coords=coords,
            tile_valid_mask=valid_mask,
            tile_page_segments=tile_page_segments,
            glyph_patches=char_patches,
            glyph_coords=glyph_coords,
            glyph_valid_mask=char_valid_mask,
            glyph_page_segments=glyph_page_segments,
            char_class_ids=char_class_ids,
            words=words,
            word_metadata=word_metadata,
            paths=list(paths) if paths is not None else None,
            batch_element_indices=batch_element_indices,
        )
    
    # Restore original training mode (Bug fix: previously left model in eval mode)
    if was_training:
        base_model.train()
    
    processed_data = {
        'tiles': tiles,
        'valid_mask': valid_mask,
        'coords': coords,
        'tile_page_segments': tile_page_segments,
        'char_patches': char_patches,
        'char_valid_mask': char_valid_mask,
        'glyph_page_segments': glyph_page_segments,
        'char_metadata': char_metadata,
        'words': words,
        'word_metadata': word_metadata,
        'labels': labels,
        'paths': paths
    }
    
    return attention_dict, processed_data


def load_image_and_xml(image_path, xml_path_override=None):
    """
    Load image and XML data for attention visualization.
    
    Args:
        image_path: Path to the image file
        xml_path_override: Optional explicit XML path (from the dataset / DB).
            When provided, this is used directly instead of re-discovering
            the XML from the filesystem via find_xml_path_pretrain().
        
    Returns:
        Tuple of (img, img_w, img_h, text_regions, text_bounds, xml_path) or None if error
    """
    from utilities.VisionModule.xml_patch_extraction import find_xml_path, parse_alto_text_regions, get_text_region_bounds
    from utilities.xml_loader import find_xml_path_pretrain
    from system import XML_PATCH_DETECT_ROTATION
    from utilities.page_rotation import detect_page_rotation, apply_rotation_correction
    
    try:
        with Image.open(image_path).convert("RGB") as img:
            img_w, img_h = img.size
            
            # Load XML for character/word positions.
            # Prefer the explicit override (from the dataset's DB-backed xml_paths);
            # fall back to filesystem discovery when not provided.
            xml_path = xml_path_override or find_xml_path_pretrain(image_path)
            text_regions = []
            text_bounds = None
            if xml_path:
                text_regions, xml_image_size = parse_alto_text_regions(xml_path)
                if xml_image_size and xml_image_size[0] > 0 and xml_image_size[1] > 0:
                    if xml_image_size != (img_w, img_h):
                        scale_x = img_w / xml_image_size[0]
                        scale_y = img_h / xml_image_size[1]
                        for region in text_regions:
                            region['hpos'] *= scale_x
                            region['vpos'] *= scale_y
                            region['width'] *= scale_x
                            region['height'] *= scale_y
                            region['center_x'] = region['hpos'] + region['width'] / 2.0
                            region['center_y'] = region['vpos'] + region['height'] / 2.0
                            if region.get('baseline') is not None:
                                x1, y1, x2, y2 = region['baseline']
                                region['baseline'] = (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)

                # IMPORTANT: patch/char extraction may apply rotation correction; apply the same
                # correction here so that metadata/coords align with the displayed image.
                if XML_PATCH_DETECT_ROTATION and text_regions:
                    rotation_angle = detect_page_rotation(text_regions, img_w, img_h)
                    if rotation_angle != 0:
                        img, text_regions = apply_rotation_correction(img, text_regions, rotation_angle)
                        img_w, img_h = img.size
                text_bounds = get_text_region_bounds(text_regions)
            
            # Return a copy of the image since we're closing the context manager
            return img.copy(), img_w, img_h, text_regions, text_bounds, xml_path
    except Exception as e:
        logger.exception(f"Error loading image {image_path}")
        print(f"Error loading image {image_path}: {e}")
        return None


def get_save_path_info(image_path, epoch, img_idx):
    """
    Extract manuscript ID and image name from path and create save directory.
    
    Args:
        image_path: Path to the image file
        epoch: Current epoch number
        img_idx: Image index in batch
        
    Returns:
        Tuple of (save_dir, manuscript_id, image_name)
    """
    path_parts = image_path.split('/')
    manuscript_id = path_parts[-3] if len(path_parts) >= 3 else "unknown"
    image_name = path_parts[-1].rsplit('.', 1)[0]
    save_dir = os.path.join(ATTENTION_IMAGES_DIR, f"epoch_{epoch:02d}", f"img_{img_idx:02d}_{manuscript_id}")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir, manuscript_id, image_name


def save_and_log_figure(fig, save_path, wandb_key, caption, *, step: int | None = None, commit: bool = False):
    """
    Save figure to disk and log to wandb.
    
    Args:
        fig: Matplotlib figure object
        save_path: Path to save the figure
        wandb_key: Wandb key for logging
        caption: Caption for wandb
    """
    import io
    from PIL import Image
    
    # Get absolute path for logging
    abs_save_path = os.path.abspath(save_path)
    
    # Render to buffer first to prevent file system race conditions
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    
    # Materialize the image before closing the buffer. PIL can otherwise keep
    # a lazy reference to `buf`, which makes W&B media logging unreliable.
    with Image.open(buf) as opened:
        image = opened.convert("RGB").copy()
    
    # Save to disk (for persistence/debugging)
    try:
        image.save(save_path)
        logger.info(f"Saved attention map: {abs_save_path} (wandb key: {wandb_key})")
    except Exception as e:
        logger.error(f"Failed to save attention map to disk: {e}")
        
    # Log to wandb directly from PIL image (avoids reading truncated file from disk).
    # In DDP, only rank0 initializes wandb. Guard logging for safety.
    if getattr(wandb, "run", None) is not None:
        # Use explicit `step` so media aligns to epochs (when provided).
        wandb.log({wandb_key: wandb.Image(image, caption=caption)}, step=step, commit=commit)
        
    # Close buffer and figure
    buf.close()
    plt.close(fig)


def _attention_overlay_display_values(attention_scores: np.ndarray, *, flat_tol: float = 0.05):
    """
    Convert raw attention scores to overlay colors without making uniform
    attention look maximally important.

    Previous overlays used score / max(score), so a nearly-flat distribution
    made every tile/glyph white-hot. Here, flat distributions map to zero
    display intensity; non-flat distributions are min/percentile normalized.
    """
    scores = np.asarray(attention_scores, dtype=np.float32)
    if scores.size == 0:
        return scores, 0.0, 1.0, False

    finite = np.isfinite(scores)
    if not finite.any():
        return np.zeros_like(scores), 0.0, 1.0, True

    clean = scores[finite]
    raw_min = float(clean.min())
    raw_max = float(clean.max())
    raw_mean = float(clean.mean())
    spread = raw_max - raw_min

    if raw_max <= 1e-12 or spread <= max(1e-12, abs(raw_mean) * flat_tol):
        return np.zeros_like(scores), raw_min, raw_max, True

    robust_hi = float(np.percentile(clean, 95))
    denom = max(robust_hi - raw_min, 1e-12)
    display = (scores - raw_min) / denom
    display = np.clip(display, 0.0, 1.0)
    display[~finite] = 0.0
    return display, raw_min, raw_max, False


def save_glyph_grid_by_letter(
    char_patches: torch.Tensor,
    char_valid_mask: torch.Tensor,
    img_idx: int,
    epoch: int,
    save_dir: str,
    manuscript_id: str,
    image_name: str,
    char_metadata: Optional[List[List[Dict]]] = None,
):
    """
    Save a grid of glyphs grouped by letter (with title per letter) and log to wandb.
    """
    try:
        from torchvision.utils import make_grid
        from system import NORMALIZE_MEAN, NORMALIZE_STD

        if char_patches.ndim != 5 or char_patches.shape[1] == 0:
            return
        valid = char_valid_mask[img_idx]
        if valid.numel() == 0 or valid.sum().item() == 0:
            return

        patches = char_patches[img_idx][valid].detach().cpu().float()
        if char_metadata is None or len(char_metadata) <= img_idx:
            return
        img_metadata = char_metadata[img_idx]
        valid_indices = torch.nonzero(valid.detach().cpu(), as_tuple=False).flatten().tolist()
        valid_metadata = [img_metadata[j] for j in valid_indices if j < len(img_metadata)]
        num_valid = min(patches.shape[0], len(valid_metadata))
        patches = patches[:num_valid]
        chars = [str((valid_metadata[j] or {}).get("char", "") or "").strip() for j in range(num_valid)]

        if CHAR_PATCH_APPLY_IMAGENET_NORM:
            mean = torch.tensor(NORMALIZE_MEAN, dtype=patches.dtype).view(1, 3, 1, 1)
            std = torch.tensor(NORMALIZE_STD, dtype=patches.dtype).view(1, 3, 1, 1)
            patches = patches * std + mean
        patches = patches.clamp(0.0, 1.0)

        char_to_order = {ch: i for i, ch in enumerate(HEBREW_ALPHABET)}
        indexed = [(char_to_order.get(c, 999), c, i) for i, c in enumerate(chars)]
        indexed.sort(key=lambda x: (x[0], x[2]))
        sorted_patches = [patches[i] for _, _, i in indexed]
        sorted_chars = [c for _, c, _ in indexed]

        groups = []
        for char, g in groupby(zip(sorted_patches, sorted_chars), key=lambda x: x[1]):
            tensors = [t for t, _ in g]
            groups.append((char or "?", tensors))

        if not groups:
            return

        n_rows = len(groups)
        fig, axes = plt.subplots(n_rows, 1, figsize=(14, 4 * n_rows))
        if n_rows == 1:
            axes = [axes]
        for ax, (char, letter_patches) in zip(axes, groups):
            tensors = torch.stack(letter_patches, dim=0)
            grid = make_grid(tensors, nrow=min(8, len(letter_patches)), padding=2)
            grid_np = grid.permute(1, 2, 0).numpy()
            ax.imshow(grid_np)
            ax.set_title(f" {char}  (n={len(letter_patches)})", fontsize=14)
            ax.axis("off")
        plt.tight_layout()

        filename = f"{image_name}_glyph_grid.png"
        save_path = os.path.join(save_dir, filename)
        # Epoch-independent wandb key so epochs become the slider axis for this image.
        wandb_key = f"attention_maps/img_{img_idx:02d}_{manuscript_id}/glyph_grid"
        caption = f"Glyph grid by letter: {manuscript_id}/{image_name}"
        save_and_log_figure(fig, save_path, wandb_key, caption, step=epoch, commit=False)
    except Exception:
        logger.exception("[Attention Maps] Failed to save glyph grid by letter")


def visualize_visual_patches_attention(ax, img, visual_attn, valid_mask, coords, img_w, img_h, epoch, img_idx):
    """
    Visualize visual patches attention on an image.
    
    IMPORTANT: This function uses batch coordinates (coords) because those are the ACTUAL patches
    the model saw during training. The attention scores correspond to these patches.
    If patches appear center-sampled instead of XML-based, the issue is in the dataset
    (falling back to center-based extraction instead of using XML extraction).
    
    Args:
        ax: Matplotlib axis to draw on
        img: PIL Image object
        visual_attn: Visual attention tensor [B, N+1, N+1]
        valid_mask: Valid mask tensor [B, N]
        coords: Normalized coordinates tensor [B, N, 2]
        img_w: Image width
        img_h: Image height
        epoch: Current epoch number
        img_idx: Image index in batch
    """
    from system import TILE_SIZE
    from matplotlib.patches import Rectangle
    
    ax.imshow(img)
    
    # Get CLS token attention to patches
    # visual_attn shape: [B, N+1, N+1] where first dim is CLS token
    cls_attn_vis = visual_attn[img_idx, 0, 1:].cpu().float().numpy()  # [N]
    valid_indices = valid_mask[img_idx].cpu().numpy()
    tile_coords_norm = coords[img_idx].cpu().float().numpy()[valid_indices]
    attention_scores = cls_attn_vis[valid_indices]
    
    # Log attention statistics for debugging
    if len(attention_scores) > 0:
        logger.debug(f"[Attention Debug] Image {img_idx}: min={attention_scores.min():.6f}, max={attention_scores.max():.6f}, "
                    f"mean={attention_scores.mean():.6f}, std={attention_scores.std():.6f}, "
                    f"range={attention_scores.max() - attention_scores.min():.6f}")
    
    display_scores, raw_min, raw_max, is_flat = _attention_overlay_display_values(attention_scores)
    
    # Draw patches with attention
    # IMPORTANT: Coordinates in batch are actual spatial positions (left=0, right=1)
    # These are the patches the model actually saw, so we must use them for correct visualization
    patches_drawn = []
    for j, (cx_norm, cy_norm) in enumerate(tile_coords_norm):
        # Use coordinates directly - they represent actual spatial positions
        cx = cx_norm * img_w
        cy = cy_norm * img_h
        left = max(0, int(cx - TILE_SIZE / 2))
        top = max(0, int(cy - TILE_SIZE / 2))
        right = min(left + TILE_SIZE, img_w)
        bottom = min(top + TILE_SIZE, img_h)
        
        display_score = display_scores[j]
        color = cm.hot(display_score)
        rect = Rectangle((left, top), right-left, bottom-top,
                       linewidth=2, edgecolor='white', facecolor=color, alpha=0.5)
        ax.add_patch(rect)
        patches_drawn.append((display_score, color))
    
    # Add colorbar
    if len(attention_scores) > 0:
        sm = plt.cm.ScalarMappable(cmap=cm.hot, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Relative Contribution Display', rotation=270, labelpad=20)
    
    flat_note = " | flat/raw-uniform" if is_flat else ""
    ax.set_title(
        f"Visual Patches (Global Contribution) - Epoch {epoch}{flat_note}\n"
        f"raw min={raw_min:.6g}, raw max={raw_max:.6g}, sum={attention_scores.sum():.6g}",
        fontsize=16,
    )
    ax.axis('off')


def create_visual_heatmap(visual_attn, valid_mask, img_idx, epoch, save_dir, manuscript_id, image_name):
    """
    Create and save visual patch-to-patch attention heatmap.
    
    Args:
        visual_attn: Visual attention tensor [B, N+1, N+1]
        valid_mask: Valid mask tensor [B, N]
        img_idx: Image index in batch
        epoch: Current epoch number
        save_dir: Directory to save the heatmap
        manuscript_id: Manuscript ID
        image_name: Image name without extension
    """
    fig_heatmap, ax_heatmap = plt.subplots(1, 1, figsize=(12, 10))
    attn_matrix = visual_attn[img_idx, 1:, 1:].cpu().float().numpy()  # [N, N] (exclude CLS)
    valid_indices = valid_mask[img_idx].cpu().numpy()
    attn_matrix_valid = attn_matrix[np.ix_(valid_indices, valid_indices)]  # [num_valid, num_valid]
    
    im = ax_heatmap.imshow(attn_matrix_valid, cmap='viridis', aspect='auto')
    ax_heatmap.set_title(f"Patch-to-Patch Attention Matrix - Epoch {epoch}\n({attn_matrix_valid.shape[0]} patches)", fontsize=14)
    ax_heatmap.set_xlabel('Attended Patches (columns)', fontsize=12)
    ax_heatmap.set_ylabel('Attending Patches (rows)', fontsize=12)
    plt.colorbar(im, ax=ax_heatmap, label='Attention Weight')
    plt.tight_layout()
    
    filename_heatmap = f"{image_name}_visual_heatmap.png"
    save_path_heatmap = os.path.join(save_dir, filename_heatmap)
    abs_save_path_heatmap = os.path.abspath(save_path_heatmap)
    # Epoch-independent wandb key so all epochs for this image show up on one slider.
    wandb_key_heatmap = f"attention_maps/img_{img_idx:02d}_{manuscript_id}/visual_heatmap"
    caption = f"Patch-to-patch attention heatmap: {manuscript_id}/{image_name}"
    logger.info(f"  Visual heatmap: {abs_save_path_heatmap}")
    save_and_log_figure(fig_heatmap, save_path_heatmap, wandb_key_heatmap, caption, step=epoch, commit=False)


def visualize_character_patches_attention(ax, img, char_attn, char_valid_mask, char_metadata, img_idx, epoch):
    """
    Visualize character patches attention on an image.
    
    Args:
        ax: Matplotlib axis to draw on
        img: PIL Image object
        char_attn: Character attention tensor [B, M+1, M+1]
        char_valid_mask: Character valid mask tensor [B, M]
        img_idx: Image index in batch
        image_path: Path to the image file
        xml_path: Path to XML file (optional)
        epoch: Current epoch number
        
    Returns:
        Number of valid characters visualized
    """
    from system import CHAR_PATCH_SIZE
    from matplotlib.patches import Rectangle
    
    ax.imshow(img)
    
    # Get CLS token attention to characters
    cls_attn_char = char_attn[img_idx, 0, 1:].cpu().float().numpy()  # [M] (may include 1 extra null token)
    char_valid = char_valid_mask[img_idx].cpu().numpy()
    # Defensive: if attention length doesn't match mask length (e.g. null token added), trim to mask length.
    if cls_attn_char.shape[0] != char_valid.shape[0]:
        min_len = min(int(cls_attn_char.shape[0]), int(char_valid.shape[0]))
        cls_attn_char = cls_attn_char[:min_len]
        char_valid = char_valid[:min_len]
    char_attention_scores = cls_attn_char[char_valid]
    
    # Log attention statistics for debugging
    if len(char_attention_scores) > 0:
        logger.debug(f"[Character Attention Debug] Image {img_idx}: min={char_attention_scores.min():.6f}, "
                    f"max={char_attention_scores.max():.6f}, mean={char_attention_scores.mean():.6f}, "
                    f"std={char_attention_scores.std():.6f}, range={char_attention_scores.max() - char_attention_scores.min():.6f}")
    
    # Use the exact per-sample metadata produced by the dataset (no re-extraction).
    # This guarantees token<->box alignment (up to padding).
    try:
        md_list = char_metadata[img_idx] if char_metadata is not None and img_idx < len(char_metadata) else []

        display_scores, raw_min, raw_max, is_flat = _attention_overlay_display_values(char_attention_scores)

        num_valid = int(char_valid.sum()) if hasattr(char_valid, "sum") else int(sum(bool(x) for x in char_valid))
        valid_indices = [idx for idx, is_valid in enumerate(char_valid) if bool(is_valid)]
        valid_md = [md_list[idx] for idx in valid_indices if idx < len(md_list)]
        num_draw = min(len(valid_md), num_valid, len(display_scores))

        # Sort by character class (alphabetical) for visualization
        draw_items = []
        for j in range(num_draw):
            meta = valid_md[j] or {}
            draw_items.append((meta.get('char_class_id', 999), j, meta))
        draw_items.sort(key=lambda x: x[0])

        for _, j, meta in draw_items:
            hpos = meta.get('hpos')
            vpos = meta.get('vpos')
            width = meta.get('width', CHAR_PATCH_SIZE)
            height = meta.get('height', CHAR_PATCH_SIZE)
            if hpos is None or vpos is None:
                continue
            display_score = display_scores[j]
            color = cm.viridis(display_score)
            rect = Rectangle((hpos, vpos), width, height,
                             linewidth=1, edgecolor='green', facecolor=color, alpha=0.6)
            ax.add_patch(rect)
            char_text = meta.get('char', '?')
            ax.text(hpos + width/2, vpos + height/2, char_text,
                    ha='center', va='center', fontsize=6, color='white', weight='bold')

        # Add colorbar
        if len(char_attention_scores) > 0:
            sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Relative Contribution Display', rotation=270, labelpad=20)
    except Exception as e:
        logger.exception("Could not visualize character positions from dataset metadata")
        print(f"Warning: Could not visualize character positions from dataset metadata: {e}")
    
    flat_note = " | flat/raw-uniform" if 'is_flat' in locals() and is_flat else ""
    raw_note = (
        f"\nraw min={raw_min:.6g}, raw max={raw_max:.6g}"
        if 'raw_min' in locals() and 'raw_max' in locals()
        else ""
    )
    ax.set_title(
        f"Character Patches (Global Contribution) - Epoch {epoch}{flat_note}\n"
        f"{len(char_attention_scores)} characters{raw_note}, sum={char_attention_scores.sum():.6g}",
        fontsize=16,
    )
    ax.axis('off')
    
    return len(char_attention_scores)


def create_character_heatmap(char_attn, char_valid_mask, img_idx, epoch, save_dir, manuscript_id, image_name, char_metadata=None):
    """
    Create and save character attention heatmap.
    
    IMPORTANT: In the current architecture, we do NOT have a true character-to-character
    self-attention matrix over glyph tokens. The glyph branch uses a set summarizer:
      queries (Q = GLYPH_NUM_SUMMARY_TOKENS) attend to glyph tokens (M).
    
    So the *real* attention is query->glyph: [B, Q, M].
    We can optionally also visualize a synthetic [B, M+1, M+1] matrix, but it is a proxy.
    
    Args:
        char_attn: Either:
          - glyph query attention: [B, Q, M]  (preferred, real)
          - synthetic char matrix: [B, M+1, M+1] (legacy/proxy)
        char_valid_mask: Character valid mask tensor [B, M]
        char_metadata: Optional list-of-lists of glyph metadata dicts (for alphabetical labeling)
    """
    # Prefer real glyph attention: [B, Q, M], but avoid confusing it with the legacy
    # synthetic/proxy character matrix which is also 3D: [B, M+1, M+1].
    if hasattr(char_attn, "dim") and char_attn.dim() == 3:
        # If it's square and matches (M or M+1), it's the legacy/proxy matrix → fall through.
        M_mask = int(char_valid_mask.shape[1])
        is_square = int(char_attn.shape[1]) == int(char_attn.shape[2])
        looks_like_legacy = is_square and int(char_attn.shape[1]) in (M_mask, M_mask + 1)

        if not looks_like_legacy:
            # Real glyph summarizer attention: [B, Q, M]
            attn_qm = char_attn[img_idx].cpu().float().numpy()
            char_valid = char_valid_mask[img_idx].cpu().numpy()

            # Filter to valid glyphs on the M dimension
            if attn_qm.shape[1] != len(char_valid):
                logger.warning(f"[Glyph Heatmap] Shape mismatch! attn_qm={attn_qm.shape}, valid_mask={len(char_valid)}")
            attn_qm_valid = attn_qm[:, char_valid] if len(char_valid) == attn_qm.shape[1] else attn_qm
            if attn_qm_valid.size == 0:
                return

            # Sort columns alphabetically by char_class_id for visualization
            md_list = char_metadata[img_idx] if char_metadata is not None and img_idx < len(char_metadata) else []
            valid_md = [md_list[j] for j in range(len(md_list)) if j < len(char_valid) and char_valid[j]]
            glyph_labels = None
            if len(valid_md) == attn_qm_valid.shape[1]:
                sort_keys = [(m.get('char_class_id', 999) if m else 999, idx) for idx, m in enumerate(valid_md)]
                sort_order = [idx for _, idx in sorted(sort_keys)]
                attn_qm_valid = attn_qm_valid[:, sort_order]
                glyph_labels = [(m.get('char', '?') if m else '?') for m in valid_md]
                glyph_labels = [glyph_labels[idx] for idx in sort_order]

            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            im = ax.imshow(attn_qm_valid, cmap='viridis', aspect='auto')
            ax.set_title(
                f"Glyph summarizer attention (queries→glyphs) - Epoch {epoch}\n"
                f"(Q={attn_qm_valid.shape[0]}, M_valid={attn_qm_valid.shape[1]}, sorted by letter)",
                fontsize=14,
            )
            if glyph_labels is not None and len(glyph_labels) <= 120:
                ax.set_xticks(range(len(glyph_labels)))
                ax.set_xticklabels(glyph_labels, fontsize=5, rotation=90)
            ax.set_xlabel('Glyphs (valid, alphabetical)', fontsize=12)
            ax.set_ylabel('Summary Queries', fontsize=12)
            plt.colorbar(im, ax=ax, label='Attention Weight')
            plt.tight_layout()

            filename = f"{image_name}_glyph_query_heatmap.png"
            save_path = os.path.join(save_dir, filename)
            # Epoch-independent wandb key so glyph query maps for this image are grouped.
            wandb_key = f"attention_maps/img_{img_idx:02d}_{manuscript_id}/glyph_query_heatmap"
            caption = f"Glyph summarizer attention (queries→glyphs): {manuscript_id}/{image_name}"
            logger.info(f"  Glyph query heatmap: {os.path.abspath(save_path)}")
            save_and_log_figure(fig, save_path, wandb_key, caption, step=epoch, commit=False)
            return

    # Fallback: legacy synthetic char matrix [B, M+1, M+1]
    # Extract attention matrix excluding CLS token
    # char_attn shape should be [B, M+1, M+1] where first token is CLS
    attn_matrix_char = char_attn[img_idx, 1:, 1:].cpu().float().numpy()  # [M, M] (exclude CLS)
    char_valid = char_valid_mask[img_idx].cpu().numpy()
    
    # Verify shapes match
    if attn_matrix_char.shape[0] != len(char_valid):
        logger.warning(f"[Character Heatmap Debug] Shape mismatch! Attention matrix shape={attn_matrix_char.shape}, "
                      f"valid mask length={len(char_valid)}")
    
    # Log attention statistics before filtering
    logger.debug(f"[Character Heatmap Debug] Image {img_idx}: Full matrix shape={attn_matrix_char.shape}, "
                f"valid chars={char_valid.sum()}/{len(char_valid)}, "
                f"full matrix range=[{attn_matrix_char.min():.8f}, {attn_matrix_char.max():.8f}], "
                f"full matrix mean={attn_matrix_char.mean():.8f}, std={attn_matrix_char.std():.8f}")
    
    # Check if attention sums to 1 per row (should be true for softmax)
    row_sums = attn_matrix_char.sum(axis=1)
    logger.debug(f"[Character Heatmap Debug] Row sums: min={row_sums.min():.6f}, max={row_sums.max():.6f}, "
                f"mean={row_sums.mean():.6f} (should be ~1.0 for softmax)")
    
    # Check if padding positions have systematically different attention
    if len(char_valid) > 0:
        padding_positions = ~char_valid
        if padding_positions.any():
            padding_attn = attn_matrix_char[:, padding_positions]  # Attention TO padding positions
            valid_attn = attn_matrix_char[:, char_valid]  # Attention TO valid positions
            logger.debug(f"[Character Heatmap Debug] Attention TO padding: mean={padding_attn.mean():.8f}, "
                       f"std={padding_attn.std():.8f}, min={padding_attn.min():.8f}, max={padding_attn.max():.8f}")
            logger.debug(f"[Character Heatmap Debug] Attention TO valid: mean={valid_attn.mean():.8f}, "
                       f"std={valid_attn.std():.8f}, min={valid_attn.min():.8f}, max={valid_attn.max():.8f}")
    
    # Filter to only valid characters
    # IMPORTANT: np.ix_ creates a meshgrid for indexing, which correctly filters both rows and columns
    attn_matrix_char_valid = attn_matrix_char[np.ix_(char_valid, char_valid)]  # [num_valid, num_valid]
    
    if attn_matrix_char_valid.shape[0] == 0:
        return
    
    # Sort rows and columns alphabetically by char_class_id for visualization
    md_list = char_metadata[img_idx] if char_metadata is not None and img_idx < len(char_metadata) else []
    valid_md = [md_list[j] for j in range(len(md_list)) if j < len(char_valid) and char_valid[j]]
    heatmap_labels = None
    if len(valid_md) == attn_matrix_char_valid.shape[0]:
        sort_keys = [(m.get('char_class_id', 999) if m else 999, idx) for idx, m in enumerate(valid_md)]
        sort_order = [idx for _, idx in sorted(sort_keys)]
        attn_matrix_char_valid = attn_matrix_char_valid[np.ix_(sort_order, sort_order)]
        heatmap_labels = [(m.get('char', '?') if m else '?') for m in valid_md]
        heatmap_labels = [heatmap_labels[idx] for idx in sort_order]
    
    # Log statistics after filtering
    logger.debug(f"[Character Heatmap Debug] Filtered matrix shape={attn_matrix_char_valid.shape}, "
                f"range=[{attn_matrix_char_valid.min():.8f}, {attn_matrix_char_valid.max():.8f}], "
                f"mean={attn_matrix_char_valid.mean():.8f}, std={attn_matrix_char_valid.std():.8f}")
    
    # Check if filtered attention sums to 1 per row (should be true for softmax)
    row_sums_valid = attn_matrix_char_valid.sum(axis=1)
    logger.debug(f"[Character Heatmap Debug] Filtered row sums: min={row_sums_valid.min():.6f}, "
                f"max={row_sums_valid.max():.6f}, mean={row_sums_valid.mean():.6f}")
    
    # Check for any columns that are consistently low (potential padding artifacts)
    col_means = attn_matrix_char_valid.mean(axis=0)
    col_stds = attn_matrix_char_valid.std(axis=0)
    logger.debug(f"[Character Heatmap Debug] Column means: min={col_means.min():.8f}, max={col_means.max():.8f}, "
                f"std={col_means.std():.8f}")
    logger.debug(f"[Character Heatmap Debug] Column stds: min={col_stds.min():.8f}, max={col_stds.max():.8f}, "
                f"mean={col_stds.mean():.8f}")
    
    fig_char_heatmap, ax_char_heatmap = plt.subplots(1, 1, figsize=(12, 10))
    im = ax_char_heatmap.imshow(attn_matrix_char_valid, cmap='viridis', aspect='auto')
    ax_char_heatmap.set_title(f"Character-to-Character Attention Matrix - Epoch {epoch}\n({attn_matrix_char_valid.shape[0]} characters, sorted by letter)", fontsize=14)
    if heatmap_labels is not None and len(heatmap_labels) <= 120:
        ax_char_heatmap.set_xticks(range(len(heatmap_labels)))
        ax_char_heatmap.set_xticklabels(heatmap_labels, fontsize=5, rotation=90)
        ax_char_heatmap.set_yticks(range(len(heatmap_labels)))
        ax_char_heatmap.set_yticklabels(heatmap_labels, fontsize=5)
    ax_char_heatmap.set_xlabel('Attended Characters (columns, alphabetical)', fontsize=12)
    ax_char_heatmap.set_ylabel('Attending Characters (rows, alphabetical)', fontsize=12)
    plt.colorbar(im, ax=ax_char_heatmap, label='Attention Weight')
    plt.tight_layout()
    
    filename_char_heatmap = f"{image_name}_character_heatmap.png"
    save_path_char_heatmap = os.path.join(save_dir, filename_char_heatmap)
    abs_save_path_char_heatmap = os.path.abspath(save_path_char_heatmap)
    wandb_key_char_heatmap = f"attention_maps/img_{img_idx:02d}_{manuscript_id}/character_heatmap"
    caption = f"Character-to-character attention heatmap: {manuscript_id}/{image_name}"
    logger.info(f"  Character heatmap: {abs_save_path_char_heatmap}")
    save_and_log_figure(fig_char_heatmap, save_path_char_heatmap, wandb_key_char_heatmap, caption, step=epoch, commit=False)


def visualize_words_attention(ax, img, word_attn, words, word_metadata, img_idx, img_w, img_h, epoch, global_min=None, global_max=None):
    """
    Visualize word attention on an image.
    
    Args:
        ax: Matplotlib axis to draw on
        img: PIL Image object
        word_attn: Word attention tensor [B, max_words+1, max_words+1]
        words: List of word lists per image
        word_metadata: List of metadata lists per image
        img_idx: Image index in batch
        img_w: Image width
        img_h: Image height
        epoch: Current epoch number
        global_min: Optional global minimum for normalization (across all images)
        global_max: Optional global maximum for normalization (across all images)
        
    Returns:
        Number of words visualized
    """
    from matplotlib.patches import Rectangle
    
    ax.imshow(img)
    
    # Get CLS token attention to words
    cls_attn_word = word_attn[img_idx, 0, 1:].cpu().float().numpy()  # [max_words]
    img_words = words[img_idx]
    num_words = len(img_words)
    word_attention_scores = cls_attn_word[:num_words]

    # Use the same flat-distribution-aware display transform as tiles/glyphs.
    # Raw contribution mass remains visible in the title; colors only express
    # within-modality contrast and never turn a uniform map into a hot map.
    norm_scores, min_score, max_score, is_flat = _attention_overlay_display_values(
        word_attention_scores
    )
    
    # Get word positions from metadata
    if word_metadata and img_idx < len(word_metadata) and word_metadata[img_idx]:
        for j, (word, meta) in enumerate(zip(img_words, word_metadata[img_idx])):
            if j >= len(word_attention_scores):
                break
            
            # Try to get position from metadata
            hpos = meta.get('hpos')
            vpos = meta.get('vpos')
            width = meta.get('width', 50)
            height = meta.get('height', 20)
            
            # If normalized coordinates available, convert to pixel coordinates
            if hpos is None and meta.get('normalized_center_x') is not None:
                center_x_norm = meta.get('normalized_center_x')
                center_y_norm = meta.get('normalized_center_y', 0.5)
                hpos = center_x_norm * img_w - width / 2
                vpos = center_y_norm * img_h - height / 2
            
            if hpos is not None and vpos is not None:
                norm_score = norm_scores[j]
                color = cm.cool(norm_score)
                rect = Rectangle((hpos, vpos), width, height,
                               linewidth=2, edgecolor='yellow', facecolor=color, alpha=0.6)
                ax.add_patch(rect)
                # Truncate long words for display
                display_word = word[:15] if len(word) > 15 else word
                # For Hebrew RTL text, matplotlib displays it LTR by default
                # Reverse the display string so it appears correctly in the visualization
                # (The actual word passed to AlephBERT is correct - this is just for display)
                from system import XML_PATCH_READING_DIRECTION_RTL
                from utilities.VisionModule.alto_parser import _is_hebrew
                if XML_PATCH_READING_DIRECTION_RTL and _is_hebrew(display_word):
                    display_word = display_word[::-1]  # Reverse for display only
                ax.text(hpos + width/2, vpos + height/2, display_word,
                       ha='center', va='center', fontsize=8, color='black', weight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    # Add colorbar
    if len(word_attention_scores) > 0:
        sm = plt.cm.ScalarMappable(cmap=cm.cool, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Relative Contribution Display', rotation=270, labelpad=20)
    
    flat_note = " | flat/raw-uniform" if is_flat else ""
    ax.set_title(
        f"Words (Global Contribution) - Epoch {epoch}{flat_note}\n"
        f"{num_words} words, raw min={min_score:.6g}, raw max={max_score:.6g}, "
        f"sum={word_attention_scores.sum():.6g}",
        fontsize=16,
    )
    ax.axis('off')
    
    return num_words


def create_word_heatmap(word_attn, words, img_idx, epoch, save_dir, manuscript_id, image_name):
    """
    Create and save word-to-word attention heatmap.
    
    Args:
        word_attn: Word attention tensor [B, max_words+1, max_words+1]
        words: List of word lists per image
        img_idx: Image index in batch
        epoch: Current epoch number
        save_dir: Directory to save the heatmap
        manuscript_id: Manuscript ID
        image_name: Image name without extension
    """
    img_words = words[img_idx]
    num_words = len(img_words)
    
    if num_words == 0:
        return
    
    fig_word_heatmap, ax_word_heatmap = plt.subplots(1, 1, figsize=(12, 10))
    attn_matrix_word = word_attn[img_idx, 1:, 1:].cpu().float().numpy()  # [max_words, max_words] (exclude CLS)
    
    # Log attention statistics before filtering
    logger.debug(f"[Word Heatmap Debug] Image {img_idx}: Full matrix shape={attn_matrix_word.shape}, "
                f"num_words={num_words}, "
                f"full matrix range=[{attn_matrix_word.min():.8f}, {attn_matrix_word.max():.8f}], "
                f"full matrix mean={attn_matrix_word.mean():.8f}, std={attn_matrix_word.std():.8f}")
    
    # Check if padding positions have systematically different attention
    if attn_matrix_word.shape[0] > num_words:
        padding_attn = attn_matrix_word[:, num_words:]  # Attention TO padding positions
        valid_attn = attn_matrix_word[:, :num_words]  # Attention TO valid positions
        logger.debug(f"[Word Heatmap Debug] Attention TO padding: mean={padding_attn.mean():.8f}, "
                   f"std={padding_attn.std():.8f}, min={padding_attn.min():.8f}, max={padding_attn.max():.8f}")
        logger.debug(f"[Word Heatmap Debug] Attention TO valid: mean={valid_attn.mean():.8f}, "
                   f"std={valid_attn.std():.8f}, min={valid_attn.min():.8f}, max={valid_attn.max():.8f}")
    
    # Get valid word indices (first num_words are valid)
    attn_matrix_word_valid = attn_matrix_word[:num_words, :num_words]  # [num_words, num_words]

    # Improve contrast when most values are 0 due to hard word filtering.
    cls_attn_word = word_attn[img_idx, 0, 1:].cpu().float().numpy()[:num_words]
    nonzero_mask = cls_attn_word > 1e-12
    if nonzero_mask.any():
        vmax = float(attn_matrix_word_valid[np.ix_(nonzero_mask, nonzero_mask)].max())
        vmax = vmax if vmax > 0 else 1.0
        vmin = 0.0
    else:
        vmax = float(attn_matrix_word_valid.max()) if attn_matrix_word_valid.size else 1.0
        vmin = 0.0
    
    # Log statistics after filtering
    logger.debug(f"[Word Heatmap Debug] Filtered matrix shape={attn_matrix_word_valid.shape}, "
                f"range=[{attn_matrix_word_valid.min():.8f}, {attn_matrix_word_valid.max():.8f}], "
                f"mean={attn_matrix_word_valid.mean():.8f}, std={attn_matrix_word_valid.std():.8f}")
    
    im = ax_word_heatmap.imshow(
        attn_matrix_word_valid,
        cmap='viridis',
        aspect='auto',
        vmin=vmin,
        vmax=vmax
    )
    ax_word_heatmap.set_title(f"Word-to-Word Attention Matrix - Epoch {epoch}\n({num_words} words)", fontsize=14)
    ax_word_heatmap.set_xlabel('Attended Words (columns)', fontsize=12)
    ax_word_heatmap.set_ylabel('Attending Words (rows)', fontsize=12)
    plt.colorbar(im, ax=ax_word_heatmap, label='Attention Weight')
    plt.tight_layout()
    
    filename_word_heatmap = f"{image_name}_word_heatmap.png"
    save_path_word_heatmap = os.path.join(save_dir, filename_word_heatmap)
    abs_save_path_word_heatmap = os.path.abspath(save_path_word_heatmap)
    wandb_key_word_heatmap = f"attention_maps/img_{img_idx:02d}_{manuscript_id}/word_heatmap"
    caption = f"Word-to-word attention heatmap: {manuscript_id}/{image_name}"
    logger.info(f"  Word heatmap: {abs_save_path_word_heatmap}")
    save_and_log_figure(fig_word_heatmap, save_path_word_heatmap, wandb_key_word_heatmap, caption, step=epoch, commit=False)


def create_word_summary_to_line_heatmap(
    word_query_to_line, img_idx, epoch, save_dir, manuscript_id, image_name
):
    """
    Create and save a heatmap of word-summary-token → line attention.

    This is the direct analogue of the glyph_query_to_glyph heatmap and is
    the most informative word-branch diagnostic because it shows how the
    learned summary queries attend to each text line — a signal that evolves
    with training even when within-line attention (frozen AlephBERT) does not.

    Args:
        word_query_to_line: [B, Q, L] tensor — summary tokens attending to lines
        img_idx: batch image index
        epoch: current epoch
        save_dir / manuscript_id / image_name: path helpers
    """
    attn = word_query_to_line[img_idx].cpu().float().numpy()  # [Q, L]
    Q, L = attn.shape
    if Q == 0 or L == 0:
        return

    fig, ax = plt.subplots(1, 1, figsize=(max(8, L * 0.5), max(6, Q * 0.15)))
    im = ax.imshow(attn, cmap="viridis", aspect="auto")
    ax.set_title(
        f"Word Summary→Line Attention — Epoch {epoch}\n"
        f"({Q} summary tokens × {L} lines)",
        fontsize=14,
    )
    ax.set_xlabel("Line index", fontsize=12)
    ax.set_ylabel("Summary token index", fontsize=12)
    plt.colorbar(im, ax=ax, label="Attention Weight")
    plt.tight_layout()

    fname = f"{image_name}_word_summary_to_line.png"
    save_path = os.path.join(save_dir, fname)
    wandb_key = f"attention_maps/img_{img_idx:02d}_{manuscript_id}/word_summary_to_line"
    caption = f"Word summary→line attention: {manuscript_id}/{image_name}"
    logger.info(f"  Word summary→line heatmap: {os.path.abspath(save_path)}")
    save_and_log_figure(fig, save_path, wandb_key, caption, step=epoch, commit=False)


def log_attention_maps(model, batch, epoch, num_images=5, xml_paths=None):
    """
    Generate and log multi-modal attention maps showing attention for patches, characters, and words.
    Uses XML-based text regions when available to focus on OCR areas.
    
    Args:
        xml_paths: Optional list of XML paths (one per image in batch).
            When provided, these are passed to load_image_and_xml so it
            doesn't have to re-discover the XML from the filesystem.
    
    This is the main orchestration function that coordinates all the modular attention visualization functions.
    """
    if batch is None:
        logger.warning("[Attention Maps] Skipping attention map generation because all selected samples failed to load.")
        return

    logger.info(f"[Attention Maps] Starting attention map generation for epoch {epoch}, {num_images} images")
    logger.info(f"[Attention Maps] Extracting attention weights from model...")
    # Extract attention from model
    attention_dict, processed_data = extract_attention_from_model(model, batch)
    logger.info(f"[Attention Maps] Attention weights extracted successfully")
    
    # Extract attention for each modality
    # Modality-local attention used to exist; in the current architecture the meaningful attention is
    # from the fusion transformer. We derive per-modality views from the fusion attention when needed.
    visual_attn = attention_dict.get('visual')  # [B, N+1, N+1] if available
    char_attn = attention_dict.get('character')  # [B, M+1, M+1] if available
    word_attn = attention_dict.get('word')  # [B, W+1, W+1] if available
    word_cls_to_word = attention_dict.get('word_cls_to_word')  # [B, W] if available
    word_query_to_line = attention_dict.get('word_query_to_line')  # [B, Q, L] if available
    fusion_attn = attention_dict.get('fusion')  # [B, L, L] from last fusion layer (avg over heads)
    logger.info(
        "[Attention Maps] sources: word=%s, word_local=%s, word_cls_to_word=%s, word_query_to_line=%s",
        "present" if word_attn is not None else "missing",
        "present" if attention_dict.get('word_local') is not None else "missing",
        "present" if word_cls_to_word is not None else "missing",
        "present" if word_query_to_line is not None else "missing",
    )

    base_model = model.module if hasattr(model, "module") else model

    if fusion_attn is not None and (visual_attn is None or char_attn is None or word_attn is None):
        # Derive per-modality attention from fusion attention.
        #
        # IMPORTANT: fusion_attn from forward_with_attention() is [B, num_latents, total_modality_tokens]
        # where CLS row has already been extracted and CLS column already stripped.
        # Token layout in dim=2: [visual tokens] + [glyph summary tokens] + [word tokens]
        # (tokens exist only if the modality is enabled and non-empty)
        from system import GLYPH_NUM_SUMMARY_TOKENS
        
        # Detect actual batch size and modality presence safely from fusion attention
        B = fusion_attn.shape[0]
        device = fusion_attn.device
        
        v_tokens_raw = processed_data.get("tiles")
        v_present = (v_tokens_raw is not None and getattr(base_model, "use_visual_mod", False))
        v_len = int(processed_data["valid_mask"].shape[1]) if v_present else 0
        
        c_tokens_raw = processed_data.get("char_patches")
        c_present = (c_tokens_raw is not None and getattr(base_model, "use_char_mod", False))
        c_len = GLYPH_NUM_SUMMARY_TOKENS if c_present else 0
        
        # Word branch outputs line tokens (not raw word tokens) to fusion.
        # We derive the count from the remainder of the fusion sequence.
        w_tokens_raw = processed_data.get("words")
        w_present = (w_tokens_raw is not None and getattr(base_model, "use_word_mod", False))
        w_len = max(0, fusion_attn.shape[2] - v_len - c_len) if w_present else 0

        # Offsets in the fusion token sequence (no CLS — already stripped)
        v_start = 0
        c_start = v_start + v_len
        w_start = c_start + c_len

        # Average over latent queries (dim=1) to get [B, total_tokens] attention scores,
        # then build per-modality [B, K+1, K+1] matrices for the visualization functions
        # (which expect CLS-row format: attn[b, 0, 1:] = CLS→token attention).
        avg_fusion = fusion_attn.mean(dim=1)  # [B, total_tokens]

        def _build_modality_attn(scores: torch.Tensor) -> torch.Tensor:
            """Build a visualization carrier whose row 0 stores global token scores."""
            K = scores.shape[1]
            out = torch.zeros(B, K + 1, K + 1, device=device, dtype=scores.dtype)
            out[:, 0, 1:] = scores
            return out

        if visual_attn is None and v_len > 0:
            visual_attn = _build_modality_attn(avg_fusion[:, v_start:v_start + v_len])
        if char_attn is None and c_len > 0:
            char_attn = _build_modality_attn(avg_fusion[:, c_start:c_start + c_len])
        if word_attn is None and w_len > 0:
            word_attn = _build_modality_attn(avg_fusion[:, w_start:w_start + w_len])
    
    paths = processed_data['paths']
    valid_mask = processed_data['valid_mask']
    coords = processed_data['coords']
    char_valid_mask = processed_data['char_valid_mask']
    char_patches = processed_data['char_patches']
    char_metadata = processed_data.get('char_metadata')
    words = processed_data['words']
    word_metadata = processed_data['word_metadata']
    
    # Compute global word attention min/max across all images for consistent normalization
    word_global_min = None
    word_global_max = None
    if word_attn is not None:
        all_word_scores = []
        for i in range(min(len(paths), num_images)):
            if words and i < len(words) and len(words[i]) > 0:
                cls_attn_word = word_attn[i, 0, 1:].cpu().float().numpy()
                num_words = len(words[i])
                word_scores = cls_attn_word[:num_words]
                if len(word_scores) > 0:
                    all_word_scores.append(word_scores)
        if len(all_word_scores) > 0:
            all_word_scores = np.concatenate(all_word_scores)
            word_global_min = all_word_scores.min()
            word_global_max = all_word_scores.max()

    # Sanity-check: log numeric stats of CLS->word attention used for visualization.
    # This helps detect "non-changing-looking" heatmaps caused by logging bugs.
    if word_attn is not None and words is not None:
        stats_count = min(len(paths), int(num_images), 3)
        for i in range(stats_count):
            if not (words and i < len(words) and words[i] is not None and len(words[i]) > 0):
                continue
            num_words = len(words[i])
            cls_scores = word_attn[i, 0, 1:].detach().cpu().float()
            cls_scores = cls_scores[:num_words]
            if cls_scores.numel() == 0:
                continue
            min_v = float(cls_scores.min().item())
            max_v = float(cls_scores.max().item())
            mean_v = float(cls_scores.mean().item())
            std_v = float(cls_scores.std().item())
            # Entropy of the (normalized) attention distribution; guard against zeros.
            s = float(cls_scores.sum().item())
            if s > 0:
                p = cls_scores / (s + 1e-12)
                entropy_v = float((-(p * (p + 1e-12).log()).sum()).item())
            else:
                entropy_v = 0.0

            nonzero_cnt = int((cls_scores > 1e-12).sum().item())
            topk = min(5, cls_scores.numel())
            top_vals = torch.topk(cls_scores, k=topk).values
            top_vals_list = [float(v.item()) for v in top_vals]
            logger.info(
                f"[Word Attention Stats] epoch={epoch} img_idx={i} words={num_words} "
                f"min={min_v:.6f} max={max_v:.6f} mean={mean_v:.6f} std={std_v:.6f} "
                f"entropy={entropy_v:.4f} nonzero_cnt={nonzero_cnt} top5={top_vals_list}"
            )
            if word_attn.shape[1] > 2:
                ww = word_attn[i, 1:num_words + 1, 1:num_words + 1].detach().cpu().float()
                row_std = float(ww.std(dim=1).mean().item()) if ww.numel() > 0 else 0.0
                logger.info(
                    f"[Word Heatmap Structure] epoch={epoch} img_idx={i} "
                    f"row_std_mean={row_std:.6f}"
                )
            if word_query_to_line is not None and i < word_query_to_line.shape[0]:
                s2l = word_query_to_line[i].detach().cpu().float()  # [Q, L]
                s2l_std = float(s2l.std().item())
                s2l_max = float(s2l.max().item())
                s2l_entropy = 0.0
                if s2l.numel() > 0:
                    p = s2l / (s2l.sum(dim=1, keepdim=True) + 1e-12)
                    s2l_entropy = float((-(p * (p + 1e-12).log()).sum(dim=1)).mean().item())
                logger.info(
                    f"[Word Summary→Line Stats] epoch={epoch} img_idx={i} "
                    f"Q={s2l.shape[0]} L={s2l.shape[1]} "
                    f"std={s2l_std:.6f} max={s2l_max:.6f} "
                    f"avg_row_entropy={s2l_entropy:.4f}"
                )
            # Log the fusion importance — the one signal that actually
            # changes per epoch (the word branch internals are frozen).
            if word_cls_to_word is not None and i < word_cls_to_word.shape[0]:
                fimp = word_cls_to_word[i].detach().cpu().float()
                topk = min(5, fimp.numel())
                top_vals = torch.topk(fimp, k=topk).values
                top_idxs = torch.topk(fimp, k=topk).indices
                logger.info(
                    f"[Word Fusion Signal] epoch={epoch} img_idx={i} "
                    f"std={float(fimp.std().item()):.6f} "
                    f"max={float(fimp.max().item()):.6f} "
                    f"top5_vals={[round(float(v), 6) for v in top_vals]} "
                    f"top5_idxs={top_idxs.tolist()}"
                )
    
    # Ensure we don't exceed batch size *or* any available attention tensor batch dimension.
    # (This can happen if attention tensors are computed/returned with a different batch dim under DP/DDP.)
    batch_size = len(paths)
    limits = [int(batch_size), int(num_images)]
    if visual_attn is not None:
        limits.append(int(visual_attn.shape[0]))
    if char_attn is not None:
        limits.append(int(char_attn.shape[0]))
    if word_attn is not None:
        limits.append(int(word_attn.shape[0]))
    if fusion_attn is not None:
        limits.append(int(fusion_attn.shape[0]))
    num_images_to_process = int(min(limits)) if limits else 0
    if num_images_to_process < min(int(batch_size), int(num_images)):
        logger.warning(
            f"[Attention Maps] Reducing num_images from {min(int(batch_size), int(num_images))} to {num_images_to_process} "
            f"due to attention tensor batch-size mismatch. "
            f"paths={batch_size}, visual_attn={getattr(visual_attn, 'shape', None)}, "
            f"char_attn={getattr(char_attn, 'shape', None)}, word_attn={getattr(word_attn, 'shape', None)}, "
            f"fusion_attn={getattr(fusion_attn, 'shape', None)}"
        )
    logger.info(f"[Attention Maps] Processing {num_images_to_process} images (batch_size={batch_size})...")
    logger.info(f"[Attention Maps] Available modalities: Visual={visual_attn is not None}, Character={char_attn is not None}, Word={word_attn is not None}")
    for i in range(num_images_to_process):
        # Safety check: ensure index is within bounds
        if i >= batch_size:
            logger.warning(f"[Attention Maps] Index {i} exceeds batch size {batch_size}. Skipping.")
            break
        image_path = paths[i]
        logger.info(f"[Attention Maps] --- Processing image {i+1}/{num_images_to_process}: {os.path.basename(image_path)} ---")
        try:
            # Load image and XML data
            xml_override = xml_paths[i] if xml_paths and i < len(xml_paths) else None
            logger.info(f"[Attention Maps]   Loading image and XML data...")
            result = load_image_and_xml(image_path, xml_path_override=xml_override)
            if result is None:
                logger.warning(f"[Attention Maps]   Failed to load image/XML. Skipping.")
                continue
            img, img_w, img_h, text_regions, text_bounds, xml_path = result
            logger.info(f"[Attention Maps]   Image loaded: {img_w}x{img_h}, XML: {'Yes' if xml_path else 'No'}")
            
            # Get save path information
            save_dir, manuscript_id, image_name = get_save_path_info(image_path, epoch, i)
            logger.info(f"[Attention Maps]   Save directory: {save_dir}")
            
            # Create panels only for modalities that have evidence for this image.
            has_visual = bool(
                visual_attn is not None
                and i < valid_mask.shape[0]
                and bool(valid_mask[i].any().item())
            )
            has_char = bool(
                char_attn is not None
                and char_patches is not None
                and i < char_patches.shape[0]
                and char_patches.shape[1] > 0
                and i < char_valid_mask.shape[0]
                and bool(char_valid_mask[i].any().item())
            )
            has_word = bool(
                word_attn is not None
                and words
                and i < len(words)
                and len(words[i]) > 0
            )
            num_modalities = sum((has_visual, has_char, has_word))
            if num_modalities == 0:
                logger.warning(f"[Attention Maps]   No modalities available. Skipping.")
                continue
            logger.info(f"[Attention Maps]   Creating visualization with {num_modalities} modalities...")
            
            fig, axes = plt.subplots(1, num_modalities, figsize=(20 * num_modalities, 16))
            if num_modalities == 1:
                axes = [axes]
            
            ax_idx = 0
            
            # 1. Visual Patches Attention
            if has_visual:
                logger.info(f"[Attention Maps]   [1/3] Processing Visual Patches attention...")
                ax = axes[ax_idx]
                # IMPORTANT: We must use batch coordinates (coords) because those are the ACTUAL patches
                # the model saw. The attention scores correspond to those patches, not to re-extracted ones.
                visualize_visual_patches_attention(
                    ax, img, visual_attn, valid_mask, coords, img_w, img_h, epoch, i
                )
                ax_idx += 1
                logger.info(f"[Attention Maps]   [1/3] Visual Patches attention completed")
            
            # 2. Character Patches Attention
            if has_char:
                logger.info(f"[Attention Maps]   [2/3] Processing Character Patches attention...")
                ax = axes[ax_idx] if num_modalities > 1 else axes[0]
                num_chars = visualize_character_patches_attention(
                    ax, img, char_attn, char_valid_mask, char_metadata, i, epoch
                )
                ax_idx += 1
                logger.info(f"[Attention Maps]   [2/3] Character Patches attention completed ({num_chars} characters)")
            
            # 3. Words Attention
            if has_word:
                logger.info(f"[Attention Maps]   [3/3] Processing Words attention...")
                ax = axes[ax_idx] if num_modalities > 1 else axes[0]
                num_words = visualize_words_attention(
                    ax, img, word_attn, words, word_metadata, i, img_w, img_h, epoch,
                    global_min=word_global_min, global_max=word_global_max
                )
                ax_idx += 1
                logger.info(f"[Attention Maps]   [3/3] Words attention completed ({num_words} words)")
            
            plt.tight_layout()
            
            # Save and log multimodal overlay
            filename = f"{image_name}_multimodal_cls.png"
            save_path = os.path.join(save_dir, filename)
            abs_save_path = os.path.abspath(save_path)
            abs_save_dir = os.path.abspath(save_dir)
            # Epoch-independent wandb key so multimodal overlays for this image share a slider.
            wandb_key = f"attention_maps/img_{i:02d}_{manuscript_id}/multimodal_cls"
            caption = f"Multi-modal global contribution: {manuscript_id}/{image_name}"
            logger.info(f"[Attention Maps]   Saving multimodal visualization...")
            logger.info(f"[Attention Maps]   Save directory: {abs_save_dir}")
            logger.info(f"[Attention Maps]   Multimodal CLS map: {abs_save_path}")
            save_and_log_figure(fig, save_path, wandb_key, caption, step=epoch, commit=False)

            # Log every available modality under its own stable key as well as
            # in the combined panel. This makes tile/glyph/word histories
            # independently inspectable in W&B and keeps local PNGs for probing.
            if has_visual:
                tile_fig, tile_ax = plt.subplots(1, 1, figsize=(20, 16))
                visualize_visual_patches_attention(
                    tile_ax, img, visual_attn, valid_mask, coords,
                    img_w, img_h, epoch, i,
                )
                save_and_log_figure(
                    tile_fig,
                    os.path.join(save_dir, f"{image_name}_tiles.png"),
                    f"attention_maps/img_{i:02d}_{manuscript_id}/tiles",
                    f"Tile contribution: {manuscript_id}/{image_name}",
                    step=epoch,
                    commit=False,
                )

            if has_char:
                glyph_fig, glyph_ax = plt.subplots(1, 1, figsize=(20, 16))
                visualize_character_patches_attention(
                    glyph_ax, img, char_attn, char_valid_mask,
                    char_metadata, i, epoch,
                )
                save_and_log_figure(
                    glyph_fig,
                    os.path.join(save_dir, f"{image_name}_glyphs.png"),
                    f"attention_maps/img_{i:02d}_{manuscript_id}/glyphs",
                    f"Glyph contribution: {manuscript_id}/{image_name}",
                    step=epoch,
                    commit=False,
                )

            if has_word:
                word_fig, word_ax = plt.subplots(1, 1, figsize=(20, 16))
                visualize_words_attention(
                    word_ax, img, word_attn, words, word_metadata, i,
                    img_w, img_h, epoch,
                    global_min=word_global_min,
                    global_max=word_global_max,
                )
                save_and_log_figure(
                    word_fig,
                    os.path.join(save_dir, f"{image_name}_words.png"),
                    f"attention_maps/img_{i:02d}_{manuscript_id}/words",
                    f"Word contribution: {manuscript_id}/{image_name}",
                    step=epoch,
                    commit=False,
                )
            logger.info(f"[Attention Maps] ✓ Completed image {i+1}/{num_images_to_process}")

        except FileNotFoundError:
            logger.warning(f"[Attention Maps] Image file not found at {image_path}. Skipping attention map.")
            print(f"Warning: Image file not found at {image_path}. Skipping attention map.")
        except Exception as e:
            logger.exception(f"[Attention Maps] An error occurred while processing {image_path} for attention map")
            print(f"An error occurred while processing {image_path}: {e}")
    
    logger.info(f"[Attention Maps] ========================================")
    logger.info(f"[Attention Maps] Attention map generation completed for epoch {epoch}")
    logger.info(f"[Attention Maps] ========================================")

def prepare_fixed_val_batch(val_loader, num_images=NUM_FIXED_VALIDATION_SAMPLES):
    """
    Collect exactly num_images validation samples for consistent tracking across epochs.
    
    Args:
        val_loader: Validation DataLoader
        num_images: Number of fixed samples to collect
        
    Returns:
        List of image paths (strings) for fixed validation samples
    """
    # IMPORTANT: do NOT iterate val_loader here.
    # Iterating val_loader triggers __getitem__ which loads full images and performs patch/XML extraction,
    # which can take minutes before training even starts. We only need stable PATHS.
    fixed_paths = []
    dataset = getattr(val_loader, "dataset", None)
    if dataset is not None and hasattr(dataset, "image_paths"):
        all_paths = list(dataset.image_paths)
        n = len(all_paths)
        k = min(int(num_images), n)
        if k <= 0:
            fixed_paths = []
        elif k >= n:
            fixed_paths = all_paths
        else:
            # Evenly-spaced deterministic selection across the FULL val set.
            # This avoids "all fixed samples are in the first batch", which is brittle
            # if an early batch is skipped due to non-finite outputs.
            idxs = np.linspace(0, n - 1, num=k, dtype=int)
            fixed_paths = [all_paths[i] for i in idxs.tolist()]
    else:
        # Fallback (older datasets/loaders): iterate, but this will be slow.
        collected_samples = 0
        for batch in val_loader:
            if batch is None:
                continue
            paths = batch[-1]
            for path in paths:
                if collected_samples < num_images:
                    fixed_paths.append(path)
                    collected_samples += 1
                else:
                    break
            if collected_samples >= num_images:
                break
    
    logger.info(f"Prepared fixed validation batch with {len(fixed_paths)} samples for consistent tracking across epochs")
    print(f"Prepared fixed validation batch with {len(fixed_paths)} samples for consistent tracking.")
    return fixed_paths

def select_diverse_attention_samples(fixed_paths, num_attention_images=NUM_ATTENTION_IMAGES):
    """
    Select attention samples from different manuscripts for better diversity.
    
    Args:
        fixed_paths: List of fixed validation sample paths
        num_attention_images: Number of diverse attention samples to select
        
    Returns:
        List of selected image paths (strings) for attention visualization
    """
    if len(fixed_paths) <= num_attention_images:
        return fixed_paths[:num_attention_images]
    
    # Group paths by manuscript ID
    manuscript_groups = {}
    for path in fixed_paths:
        path_parts = path.split('/')
        manuscript_id = path_parts[-3] if len(path_parts) >= 3 else "unknown"
        if manuscript_id not in manuscript_groups:
            manuscript_groups[manuscript_id] = []
        manuscript_groups[manuscript_id].append(path)
    
    # Select one image from each manuscript until we have enough
    selected_paths = []
    manuscript_ids = list(manuscript_groups.keys())
    
    # Round-robin selection from different manuscripts
    manuscript_index = 0
    while len(selected_paths) < num_attention_images and manuscript_ids:
        current_manuscript = manuscript_ids[manuscript_index]
        
        if manuscript_groups[current_manuscript]:
            # Take the first available image from this manuscript
            selected_paths.append(manuscript_groups[current_manuscript].pop(0))
            
            # If this manuscript has no more images, remove it from the list
            if not manuscript_groups[current_manuscript]:
                manuscript_ids.remove(current_manuscript)
                if manuscript_index >= len(manuscript_ids) and manuscript_ids:
                    manuscript_index = 0
            else:
                manuscript_index = (manuscript_index + 1) % len(manuscript_ids)
        else:
            # This shouldn't happen, but just in case
            manuscript_ids.remove(current_manuscript)
            if manuscript_index >= len(manuscript_ids) and manuscript_ids:
                manuscript_index = 0
    
    # Count unique manuscripts (use same extraction method as grouping)
    unique_manuscripts = set()
    for path in selected_paths:
        path_parts = path.split('/')
        manuscript_id = path_parts[-3] if len(path_parts) >= 3 else "unknown"
        unique_manuscripts.add(manuscript_id)
    
    num_manuscripts = len(unique_manuscripts)
    logger.info(f"Selected {len(selected_paths)} attention images from {num_manuscripts} different manuscripts for visualization")
    print(f"Selected {len(selected_paths)} attention images from {num_manuscripts} different manuscripts:")
    for i, path in enumerate(selected_paths):
        path_parts = path.split('/')
        manuscript_id = path_parts[-3] if len(path_parts) >= 3 else "unknown"
        image_name = path_parts[-1]
        logger.info(f"  Attention sample {i+1}: {path}")
        print(f"  {i+1}. {manuscript_id}/{image_name}")
    
    return selected_paths


# ============================================================================
# PCA ANALYSIS FUNCTIONS
# ============================================================================

def collect_validation_latents(model, val_loader, max_samples=PCA_MAX_SAMPLES):
    """
    Collect latent representations from validation set.
    
    Args:
        model: MultiModal model instance
        val_loader: Validation DataLoader
        max_samples: Maximum number of samples to collect
        
    Returns:
        numpy array of shape [N, LATENT_DIM] containing latent representations
    """
    model.eval()
    all_latents = []
    
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            if len(all_latents) >= max_samples:
                break
                
            # Unpack full multi-modal batch
            tiles, valid_mask, coords, tile_page_segments, char_patches, char_valid_mask, glyph_coords, glyph_page_segments, char_class_ids, _char_metadata, words, word_metadata, labels, paths = batch
            device = next(model.parameters()).device
            tiles = tiles.to(device, non_blocking=True)
            valid_mask = valid_mask.to(device, non_blocking=True)
            coords = coords.to(device, non_blocking=True)
            tile_page_segments = tile_page_segments.to(device, non_blocking=True)
            char_patches = char_patches.to(device, non_blocking=True)
            char_valid_mask = char_valid_mask.to(device, non_blocking=True)
            glyph_coords = glyph_coords.to(device, non_blocking=True)
            glyph_page_segments = glyph_page_segments.to(device, non_blocking=True)
            char_class_ids = char_class_ids.to(device, non_blocking=True)
            
            batch_element_indices = torch.arange(
                tiles.shape[0], dtype=torch.long, device=tiles.device
            )
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, latent, _ = model(
                    tiles=tiles,
                    tile_coords=coords,
                    tile_valid_mask=valid_mask,
                    tile_page_segments=tile_page_segments,
                    glyph_patches=char_patches,
                    glyph_coords=glyph_coords,
                    glyph_valid_mask=char_valid_mask,
                    glyph_page_segments=glyph_page_segments,
                    char_class_ids=char_class_ids,
                    words=words,
                    word_metadata=word_metadata,
                    batch_element_indices=batch_element_indices,
                )

            # If model produced non-finite latents, skip them (PCA can't handle NaNs anyway).
            if not torch.isfinite(latent).all():
                logger.error("[PCA] Non-finite latent detected while collecting validation latents. Skipping batch.")
                if paths:
                    logger.error(f"[PCA] Example path: {paths[0]}")
                continue
            
            # Convert to numpy and collect (convert BFloat16 to float32 first)
            latent_np = latent.cpu().float().numpy()
            remaining = max_samples - len(all_latents)
            if remaining < len(latent_np):
                latent_np = latent_np[:remaining]
            all_latents.append(latent_np)
    
    if len(all_latents) == 0:
        return None
    
    return np.vstack(all_latents)


def compute_pca_elbow_dimension(latents, max_components=None, explained_variance_threshold=PCA_EXPLAINED_VARIANCE_THRESHOLD):
    """
    Compute optimal PCA dimension using elbow method and 95% variance threshold.
    
    Important: The result is estimated from the *sample* provided (e.g. up to PCA_MAX_SAMPLES).
    When the number of samples is smaller than the number of classes, the effective dimension
    may be underestimated because we cannot resolve more directions than min(n_samples, latent_dim).
    The model still uses the full LATENT_DIM for classification; this is a diagnostic metric only.
    
    To get the *real* optimal dimension per epoch (not capped at PCA_DIMENSION), fit PCA with
    full rank: pass max_components=None so n_components = min(n_samples, n_features). Then
    elbow and 95% threshold reflect the true effective dimension.
    
    Methods:
    1. Elbow (threshold): first component where relative gain in cumulative variance < 0.01%.
    2. Elbow (curvature): classical elbow = argmax of 2nd derivative of cumulative variance curve.
    3. 95% threshold: fewest components needed to explain >= 95% of variance (most interpretable).
    
    Args:
        latents: numpy array of shape [N, LATENT_DIM]
        max_components: Maximum number of components to fit (default None = full rank for diagnostic).
            Use None to get true optimal dimension; use e.g. PCA_MAX_COMPONENTS to cap for speed.
        explained_variance_threshold: Threshold for variance method (default 0.95)
        
    The PCA is fitted on the raw embeddings. Scikit-learn centers them but does
    not scale individual coordinates, matching the production PCA/search path
    and preserving the variance geometry used by cosine retrieval.

    Returns:
        Tuple of (optimal_dim, threshold_dim, elbow_curvature_dim, explained_variances,
                  cumulative_variances, pca, scaler). ``scaler`` is retained as
                  a compatibility return value and is always ``None``.
    """
    if latents is None or len(latents) == 0:
        return None, None, None, None, None, None, None
    
    # Rank of data is at most min(n_samples, n_features); PCA can't produce more non-trivial components.
    # Use full rank (max_components=None) so optimal_dim reflects the real effective dimension per epoch.
    n_samples, n_features = latents.shape[0], latents.shape[1]
    n_components = min(n_samples, n_features)
    if max_components is not None:
        n_components = min(n_components, max_components)
    pca = PCA(n_components=n_components)
    # PCA centers automatically. Do not standardize each coordinate: production
    # search PCA also uses raw centered latents with ``whiten=False``.
    pca.fit(latents)
    
    explained_variances = pca.explained_variance_ratio_
    cumulative_variances = np.cumsum(explained_variances)
    
    # Method 1: Elbow (threshold) - first component where relative gain < 0.01%
    denom = np.maximum(cumulative_variances[:-1], 1e-12)
    percentage_changes = np.diff(cumulative_variances) / denom
    elbow_threshold = 0.0001
    elbow_indices = np.where(percentage_changes < elbow_threshold)[0]
    if len(elbow_indices) > 0:
        elbow_idx = elbow_indices[0] + 1
        optimal_dim = min(elbow_idx, n_components)
    else:
        lenient_indices = np.where(percentage_changes < 0.001)[0]
        if len(lenient_indices) > 0:
            elbow_idx = lenient_indices[0] + 1
            optimal_dim = min(elbow_idx, n_components)
        else:
            optimal_dim = n_components
    
    # Method 2: Elbow (curvature) - classical elbow = where 2nd derivative of cumulative variance is most negative
    if len(cumulative_variances) >= 3:
        d2 = np.diff(cumulative_variances, n=2)  # length n_components-2; d2[i] ~ curvature at component i+2
        elbow_curvature_idx = np.argmin(d2)      # index in d2 (0-based)
        elbow_curvature_dim = min(elbow_curvature_idx + 2, n_components)  # +2 to get 1-based component count
    else:
        elbow_curvature_dim = optimal_dim
    
    # Method 3: 95% variance threshold (most interpretable for "effective dimension")
    threshold_indices = np.where(cumulative_variances >= explained_variance_threshold)[0]
    if len(threshold_indices) > 0:
        threshold_dim = threshold_indices[0] + 1
    else:
        threshold_dim = n_components
    
    return optimal_dim, threshold_dim, elbow_curvature_dim, explained_variances, cumulative_variances, pca, None


def visualize_pca_95_variance_threshold(explained_variances, cumulative_variances, threshold_dim, epoch=0, save_dir=None):
    """
    Visualize the 95% variance threshold method for PCA dimension selection.
    
    Creates a plot showing cumulative explained variance with the 95% threshold marked.
    
    Args:
        explained_variances: numpy array of explained variance ratios
        cumulative_variances: numpy array of cumulative explained variances
        threshold_dim: Dimension that captures 95% of variance
        epoch: Current epoch number
        save_dir: Directory to save the plot
        
    Returns:
        matplotlib figure object
    """
    if explained_variances is None or cumulative_variances is None or threshold_dim is None:
        return None
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    n_components = len(explained_variances)
    component_range = np.arange(1, n_components + 1)
    
    # Plot cumulative explained variance
    ax.plot(component_range, cumulative_variances, 'g-', linewidth=2, alpha=0.7, label='Cumulative Explained Variance')
    ax.axvline(x=threshold_dim, color='r', linestyle='--', linewidth=2, label=f'95% Threshold: {threshold_dim} components')
    ax.axhline(y=0.95, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='95% Variance Threshold')
    ax.scatter([threshold_dim], [cumulative_variances[threshold_dim-1]], 
               color='red', s=100, zorder=5, marker='o', edgecolors='black', linewidths=2)
    
    # Add text annotation for threshold dimension
    ax.text(threshold_dim, cumulative_variances[threshold_dim-1] + 0.02,
            f'{threshold_dim} components\n{cumulative_variances[threshold_dim-1]:.2%} variance',
            fontsize=11, ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Explained Variance', fontsize=12, fontweight='bold')
    ax.set_title(f'PCA Dimension Selection: 95% Variance Threshold - Epoch {epoch}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, min(n_components + 10, component_range[-1]))
    
    plt.tight_layout()
    
    # Save if save_dir is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"pca_95_variance_threshold_epoch_{epoch:02d}.png"
        save_path = os.path.join(save_dir, filename)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig


def visualize_pca_elbow_method(explained_variances, cumulative_variances, optimal_dim, epoch=0, save_dir=None):
    """
    Visualize the elbow method for PCA dimension selection.
    
    Creates a plot showing:
    1. Explained variance ratio per component
    2. Cumulative explained variance
    3. Marked elbow point (optimal dimension)
    
    Args:
        explained_variances: numpy array of explained variance ratios
        cumulative_variances: numpy array of cumulative explained variances
        optimal_dim: Optimal dimension found by elbow method
        epoch: Current epoch number
        save_dir: Directory to save the plot
        
    Returns:
        matplotlib figure object
    """
    if explained_variances is None or cumulative_variances is None or optimal_dim is None:
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    n_components = len(explained_variances)
    component_range = np.arange(1, n_components + 1)
    
    # Plot 1: Explained variance ratio per component
    ax1.plot(component_range, explained_variances, 'b-', linewidth=2, alpha=0.7, label='Explained Variance Ratio')
    ax1.axvline(x=optimal_dim, color='r', linestyle='--', linewidth=2, label=f'Elbow Point: {optimal_dim} components')
    ax1.scatter([optimal_dim], [explained_variances[optimal_dim-1]], 
                color='red', s=100, zorder=5, marker='o', edgecolors='black', linewidths=2)
    ax1.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12, fontweight='bold')
    ax1.set_title(f'Explained Variance per Component - Epoch {epoch}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10)
    ax1.set_xlim(0, min(n_components + 10, component_range[-1]))
    
    # Plot 2: Cumulative explained variance
    ax2.plot(component_range, cumulative_variances, 'g-', linewidth=2, alpha=0.7, label='Cumulative Explained Variance')
    ax2.axvline(x=optimal_dim, color='r', linestyle='--', linewidth=2, label=f'Elbow Point: {optimal_dim} components')
    ax2.axhline(y=cumulative_variances[optimal_dim-1], color='r', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.scatter([optimal_dim], [cumulative_variances[optimal_dim-1]], 
                color='red', s=100, zorder=5, marker='o', edgecolors='black', linewidths=2)
    
    # Add text annotation for optimal dimension
    ax2.text(optimal_dim, cumulative_variances[optimal_dim-1] + 0.02,
             f'{optimal_dim} components\n{cumulative_variances[optimal_dim-1]:.2%} variance',
             fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax2.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=12, fontweight='bold')
    ax2.set_title(f'PCA Dimension Selection: Elbow Method - Epoch {epoch}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlim(0, min(n_components + 10, component_range[-1]))
    
    plt.tight_layout()
    
    # Save if save_dir is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"pca_elbow_method_epoch_{epoch:02d}.png"
        save_path = os.path.join(save_dir, filename)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig


def visualize_pca_dimension_utilization(explained_variances, original_latent_dim, epoch=0, save_dir=None):
    """
    Visualize what percentage of the original latent dimension we actually utilize.
    
    Shows how much variance each PCA component explains, indicating which dimensions
    are actually being used and how efficiently we're utilizing the latent space.
    
    Args:
        explained_variances: numpy array of explained variance ratios per component
        original_latent_dim: Original dimension of the latent space.
        epoch: Current epoch number
        save_dir: Directory to save the plot
        
    Returns:
        matplotlib figure object
    """
    if explained_variances is None or len(explained_variances) == 0:
        return None
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    n_components = len(explained_variances)
    component_range = np.arange(1, n_components + 1)
    
    # Plot 1: Explained variance per component (bar chart)
    colors = plt.cm.viridis(explained_variances / explained_variances.max())
    bars = ax1.bar(component_range, explained_variances * 100, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('PCA Component Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Explained Variance (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Latent Dimension Utilization - Epoch {epoch}\n'
                  f'Variance Explained by Each PCA Component (out of {original_latent_dim} original dimensions)',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax1.set_xlim(0, n_components + 1)
    
    # Add percentage labels on top bars (only for top components to avoid clutter)
    top_n = min(20, n_components)
    top_indices = np.argsort(explained_variances)[-top_n:][::-1]
    for idx in top_indices:
        height = explained_variances[idx] * 100
        if height > 0.1:  # Only label if > 0.1%
            ax1.text(component_range[idx], height, f'{height:.2f}%',
                    ha='center', va='bottom', fontsize=8, rotation=90)
    
    # Plot 2: Cumulative utilization percentage
    cumulative_variance = np.cumsum(explained_variances) * 100
    utilization_percentage = (component_range / original_latent_dim) * 100
    
    ax2.plot(component_range, cumulative_variance, 'g-', linewidth=2, alpha=0.7, 
             label='Cumulative Variance Explained (%)')
    ax2.plot(component_range, utilization_percentage, 'r--', linewidth=2, alpha=0.7,
             label=f'Dimension Utilization (% of {original_latent_dim} dims)')
    
    # Fill area between curves to show efficiency
    ax2.fill_between(component_range, cumulative_variance, utilization_percentage, 
                     where=(cumulative_variance >= utilization_percentage),
                     alpha=0.3, color='green', label='Efficient Utilization')
    ax2.fill_between(component_range, cumulative_variance, utilization_percentage,
                     where=(cumulative_variance < utilization_percentage),
                     alpha=0.3, color='red', label='Underutilized')
    
    ax2.set_xlabel('Number of PCA Components', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Latent Space Efficiency: Variance Explained vs Dimension Utilization - Epoch {epoch}',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10)
    ax2.set_xlim(0, min(n_components + 10, component_range[-1]))
    ax2.set_ylim(0, max(100, cumulative_variance[-1] * 1.1))
    
    plt.tight_layout()
    
    # Save if save_dir is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"pca_dimension_utilization_epoch_{epoch:02d}.png"
        save_path = os.path.join(save_dir, filename)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig


def create_pca_correlation_map(latents, pca=None, scaler=None, pca_dim=PCA_DIMENSION, epoch=0, save_dir=None, max_dims_for_display=256):
    """
    Create a correlation map of **original latent dimensions** (raw latents).
    
    This computes the correlation between latent dimensions (columns of latents) across samples.
    High off-diagonal correlation indicates redundancy in the learned representation (PCA would help).
    Low off-diagonal correlation indicates the latent space is already well decorrelated.
    
    Note: PCA-transformed data would always show ~0 correlation (orthogonal by construction),
    so we intentionally use the raw latent dimensions to assess whether the model has redundancy.
    
    Args:
        latents: numpy array of shape [N, LATENT_DIM]
        pca: Pre-fitted PCA object (optional, unused; kept for API compatibility)
        scaler: Pre-fitted StandardScaler object (optional, unused; kept for API compatibility)
        pca_dim: Unused; kept for API compatibility
        epoch: Current epoch number
        save_dir: Directory to save the correlation map
        max_dims_for_display: Max number of latent dimensions to include in the plot (default 256).
            If LATENT_DIM is larger, we take the first max_dims_for_display dimensions to keep the plot readable.
        
    Returns:
        Tuple of (fig, correlation_matrix, n_dims_shown, latents_pca) where:
        - fig: matplotlib figure object
        - correlation_matrix: numpy array of correlation matrix [n_dims_shown, n_dims_shown] for raw latent dims
        - n_dims_shown: number of dimensions in the correlation matrix
        - latents_pca: None (kept for API compatibility; callers that need PCA transform should use pca separately)
    """
    if latents is None or len(latents) == 0:
        return None, None, None, None
    
    n_samples, n_features = latents.shape[0], latents.shape[1]
    # Use at most max_dims_for_display dimensions for the correlation plot.
    n_dims_to_plot = min(n_features, max_dims_for_display)
    latents_sub = latents[:, :n_dims_to_plot]  # [N, n_dims_to_plot]
    
    # Correlation between **original latent dimensions** (columns = dimensions, rows = samples)
    # This shows redundancy in the learned representation; high off-diagonal = correlated dims
    correlation_matrix = np.corrcoef(latents_sub.T)  # [n_dims_to_plot, n_dims_to_plot]

    # region agent log (debug-4580da)
    try:
        import json, time
        stds = np.std(latents_sub, axis=0)
        payload = {
            "sessionId": "4580da",
            "runId": f"epoch_{int(epoch)}",
            "hypothesisId": "H_nan_corrcoef",
            "location": "train/trainer.py:create_pca_correlation_map",
            "message": "PCA correlation diagnostics",
            "data": {
                "latents_shape": [int(n_samples), int(n_features)],
                "latents_sub_shape": [int(latents_sub.shape[0]), int(latents_sub.shape[1])],
                "latents_sub_std_min": float(np.nanmin(stds)) if stds.size else None,
                "latents_sub_std_zeros": int(np.sum(stds == 0.0)) if stds.size else 0,
                "corr_nan_count": int(np.isnan(correlation_matrix).sum()),
                "corr_inf_count": int(np.isinf(correlation_matrix).sum()),
                "corr_min": float(np.nanmin(correlation_matrix)),
                "corr_max": float(np.nanmax(correlation_matrix)),
            },
            "timestamp": int(time.time() * 1000),
        }
        with open("/home/omerv/JoinsFinder/.cursor/debug-4580da.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass
    # endregion agent log (debug-4580da)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    
    im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax.set_title(f'Latent Dimension Correlation Map - Epoch {epoch}\n'
                 f'Correlation between {n_dims_to_plot} latent dimensions (first {n_dims_to_plot} of {n_features}). '
                 f'High off-diagonal = redundancy.',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Latent Dimension Index', fontsize=12)
    ax.set_ylabel('Latent Dimension Index', fontsize=12)
    
    ax.text(0.02, 0.98, 'Diagonal = 1.0 (self-correlation).\nOff-diagonal: high = redundant dims, low = decorrelated.',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    cbar = plt.colorbar(im, ax=ax, label='Correlation Coefficient', fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    
    tick_step = max(1, n_dims_to_plot // 10)
    ax.set_xticks(np.arange(0, n_dims_to_plot, tick_step))
    ax.set_yticks(np.arange(0, n_dims_to_plot, tick_step))
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"pca_correlation_map_epoch_{epoch:02d}.png"
        save_path = os.path.join(save_dir, filename)
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig, correlation_matrix, n_dims_to_plot, None


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    combined_loss,
    num_epochs=10,
    tile_branch_freeze_from_epoch=-1,
    tile_branch_freeze_until_epoch=-1,
    glyph_branch_freeze_from_epoch=-1,
    glyph_branch_freeze_until_epoch=-1,
    word_branch_freeze_from_epoch=-1,
    word_branch_freeze_until_epoch=-1,
    lr_stage2=3e-4,
    project_name='manuscript_tiles',
    run_start_info=None,
    gradient_accumulation_steps=1,
    idx2label=None,
    training_mode=None,
    device: Optional[torch.device] = None,
    is_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    train_sampler=None,
    geniza_train_loader=None,
    geniza_val_loader=None,
    geniza_train_sampler=None,
    geniza_contrastive_loss=None,
    geniza_feature_queue=None,
    geniza_contrastive_weight: float = 0.0,
    cluster_test_loaders: Optional[Mapping[str, object]] = None,
):
    """
    Main training loop for MultiModal model.

    Per-branch freeze ranges (tile/glyph/word_branch_freeze_from_epoch..until_epoch inclusive)
    control which branches are trainable at each epoch.
    When epoch is inside a branch's range, that branch is frozen; outside the range it trains.
    E.g. tile (0,2) = freeze tile branch first 3 epochs; set from/until to -1 for no freeze.
    """
    is_main = (rank == 0)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cluster_test_loaders = cluster_test_loaders or {}
    if is_main and cluster_test_loaders:
        expected_test_sources = set(CLUSTER_TEST_SOURCE_NAMES)
        configured_test_sources = set(cluster_test_loaders.keys())
        if configured_test_sources != expected_test_sources:
            raise RuntimeError(
                "When fixed cluster tests are provided, both sources are required: "
                f"expected={sorted(expected_test_sources)}, "
                f"configured={sorted(configured_test_sources)}"
            )

    # If something hangs (dataloader / wandb / barrier), dump Python tracebacks periodically.
    if is_main:
        try:
            if bool(ENABLE_TRAIN_HANG_WATCHDOG):
                faulthandler.enable()
                # Dump once after N seconds (helps debug hangs without spamming output).
                faulthandler.dump_traceback_later(int(TRAIN_HANG_WATCHDOG_SECONDS), repeat=False)
        except Exception:
            logger.exception("Failed to enable faulthandler watchdog")

    def ddp_barrier():
        """Synchronize all ranks if running under DDP."""
        if is_distributed and dist.is_available() and dist.is_initialized():
            try:
                if hasattr(dist, "monitored_barrier"):
                    dist.monitored_barrier(timeout=datetime.timedelta(minutes=30))
                else:
                    dist.barrier()
            except Exception:
                logger.exception("[DDP] barrier failed/hung")
                raise

    def collect_memory_usage() -> Dict[str, float]:
        """Collect current memory usage for the epoch-level W&B payload."""
        if not is_main:
            return {}
        memory_metrics: Dict[str, float] = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                memory_metrics[f"memory.gpu_{i}_allocated_gb"] = allocated
                memory_metrics[f"memory.gpu_{i}_reserved_gb"] = reserved
        # System memory
        system_mem = psutil.virtual_memory()
        memory_metrics["memory.system_used_gb"] = system_mem.used / 1024**3
        memory_metrics["memory.system_available_gb"] = system_mem.available / 1024**3
        return memory_metrics

    def clip_accumulated_gradients() -> None:
        max_norm = float(GRADIENT_CLIP_NORM)
        params = []
        seen = set()
        for group in optimizer.param_groups:
            for param in group.get("params", []):
                if param.grad is None:
                    continue
                param_id = id(param)
                if param_id in seen:
                    continue
                seen.add(param_id)
                params.append(param)
        if not params:
            return
        if max_norm > 0.0:
            # Never let a non-finite accumulated gradient reach optimizer.step.
            # ``error_if_nonfinite`` turns silent multi-day corruption into an
            # immediate, actionable failure at the responsible update.
            torch.nn.utils.clip_grad_norm_(
                params, max_norm=max_norm, error_if_nonfinite=True
            )
        else:
            bad = [name for name, param in model.named_parameters()
                   if param.grad is not None and not torch.isfinite(param.grad).all()]
            if bad:
                raise RuntimeError(
                    "Non-finite accumulated gradients before optimizer.step: "
                    + ", ".join(bad[:20])
                )
    if is_main:
        wandb.init(project=project_name, name=RUN_NAME, config=run_start_info)
        try:
            wandb.define_metric("epoch")
            for metric_pattern in (
                "train/*",
                "val/*",
                "test/*",
                "clusters/*",
                "pca/*",
                "memory/*",
                "learning_rate",
                "learning_rate/*",
                "validation_errors",
                "validation_batch_predictions",
                "attention_maps/*",
            ):
                wandb.define_metric(metric_pattern, step_metric="epoch")
        except Exception:
            logger.exception("[W&B] Failed to define epoch-based metrics")

    # Initialize tables for logging across epochs
    # NOTE: W&B Tables can be finicky if you mutate + re-log the *same* Table object over time.
    # To ensure the UI always shows rows, we construct fresh per-epoch tables at logging time.
    error_table = None
    val_batch_table = None

    best_val_accuracy = float("-inf")
    best_geniza_map = float("-inf")
    best_val_retrieval_map = float("-inf")
    best_checkpoint_metric_source = None
    best_checkpoint_score = float("-inf")
    best_epoch = 0
    use_geniza_contrastive = (
        geniza_train_loader is not None
        and geniza_contrastive_loss is not None
        and geniza_feature_queue is not None
        and float(geniza_contrastive_weight) > 0.0
    )
    # Use provided training_mode or fall back to system default
    mode = training_mode if training_mode is not None else TRAINING_MODE
    # Build path: Results/best_model with filename = modes enabled + time
    base_model = model.module if hasattr(model, "module") else model
    modes_parts = []
    if base_model.use_visual_mod:
        modes_parts.append("V")
    if base_model.use_char_mod:
        modes_parts.append("C")
    if base_model.use_word_mod:
        modes_parts.append("W")
    modes_str = "_".join(modes_parts) if modes_parts else "none"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    best_model_path = f"{BEST_MODEL_PATH_BASE}/best_model_{mode}_{modes_str}_{timestamp}.pth"
    last_model_path = f"{BEST_MODEL_PATH_BASE}/last_model_{mode}_{modes_str}_{timestamp}.pth"
    os.makedirs(BEST_MODEL_PATH_BASE, exist_ok=True)

    # Prepare a fixed set of validation samples for consistent tracking across epochs.
    # NOTE: This should be FAST and must not iterate the DataLoader.
    fixed_val_paths = prepare_fixed_val_batch(val_loader, num_images=NUM_FIXED_VALIDATION_SAMPLES) if (is_main and val_loader is not None) else []
    
    # Select diverse attention samples, capped by available paths
    attention_target = min(NUM_ATTENTION_IMAGES, len(fixed_val_paths))
    attention_vis_paths = select_diverse_attention_samples(fixed_val_paths, num_attention_images=attention_target) if is_main else []
    if is_main:
        logger.info(f"Selected {len(attention_vis_paths)} fixed images for attention visualization across all epochs")
        print(f"Selected {len(attention_vis_paths)} fixed images for attention visualization across all epochs:")
        for i, path in enumerate(attention_vis_paths, 1):
            logger.info(f"  Fixed attention sample {i}: {path}")
            print(f"  {i}. {path}")

    # IMPORTANT: cache the padded attention batch once (rank0 only).
    # Do NOT iterate the full val_loader: it loads images/tiles/words/chars. We only need ~5 samples.
    cached_attention_indices = None
    cached_attention_paths = []
    if is_main and val_loader is not None and len(attention_vis_paths) > 0:
        try:
            dataset = getattr(val_loader, "dataset", None)
            if dataset is None or not hasattr(dataset, "image_paths"):
                raise RuntimeError("val_loader.dataset.image_paths is not available; cannot cache attention batch cheaply.")

            # Build a path -> index map once.
            idx_by_path = {os.path.normpath(p): i for i, p in enumerate(list(dataset.image_paths))}
            indices = []
            for p in attention_vis_paths[:NUM_ATTENTION_IMAGES]:
                k = os.path.normpath(p) if isinstance(p, str) else os.path.normpath(str(p))
                if k in idx_by_path:
                    indices.append(idx_by_path[k])
            if len(indices) != min(NUM_ATTENTION_IMAGES, len(attention_vis_paths)):
                logger.warning(f"[Attention Maps] Could only map {len(indices)}/{NUM_ATTENTION_IMAGES} attention paths to dataset indices.")

            # Defer actual loading - only cache the indices, load on-demand during training
            # This avoids loading all images immediately at startup (which was causing memory issues)
            if len(indices) > 0:
                cached_attention_indices = indices
                cached_attention_paths = [dataset.image_paths[i] for i in indices]
                logger.info(f"[Attention Maps] Cached attention indices for {len(cached_attention_paths)} images (will load on-demand each epoch).")
            else:
                logger.warning("[Attention Maps] No samples found for caching; attention maps may be skipped.")
        except Exception:
            logger.exception("[Attention Maps] Failed to build cached attention batch; will fall back to per-epoch search.")

    # Track per-branch freeze state (pretrain and finetune; from main.py / previous epoch)
    def _epoch_in_freeze_range(epoch, from_ep, until_ep):
        return from_ep >= 0 and from_ep <= epoch <= until_ep

    branch_frozen = {
        "tile": _epoch_in_freeze_range(0, tile_branch_freeze_from_epoch, tile_branch_freeze_until_epoch),
        "glyph": _epoch_in_freeze_range(0, glyph_branch_freeze_from_epoch, glyph_branch_freeze_until_epoch),
        "word": _epoch_in_freeze_range(0, word_branch_freeze_from_epoch, word_branch_freeze_until_epoch),
    }

    for epoch in range(num_epochs):
        model.train()
        if hasattr(combined_loss, "set_training_epoch"):
            combined_loss.set_training_epoch(epoch)
            if is_main and hasattr(combined_loss, "current_word_aux_weight"):
                logger.info(
                    "[Word Aux] epoch %d effective weight %.6f",
                    epoch + 1,
                    combined_loss.current_word_aux_weight,
                )
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        if geniza_train_sampler is not None and hasattr(geniza_train_sampler, "set_epoch"):
            geniza_train_sampler.set_epoch(epoch)

        # ArcFace margin warmup: linearly ramp from 0 to target over N full epochs.
        if (ARCFACE_WEIGHT > 0
                and ARCFACE_MARGIN_WARMUP_EPOCHS > 0
                and hasattr(combined_loss, "set_arcface_margin")):
            warmup_frac = min(1.0, epoch / ARCFACE_MARGIN_WARMUP_EPOCHS)
            current_margin = ARCFACE_MARGIN * warmup_frac
            combined_loss.set_arcface_margin(current_margin)
            if is_main:
                logger.info(f"[ArcFace] margin warmup: epoch {epoch+1}, margin={current_margin:.4f} "
                            f"(target={ARCFACE_MARGIN}, warmup_epochs={ARCFACE_MARGIN_WARMUP_EPOCHS})")

        base_model = model.module if hasattr(model, "module") else model

        # Apply per-branch freeze/unfreeze by epoch range (pretrain and finetune)
        freeze_tile = _epoch_in_freeze_range(epoch, tile_branch_freeze_from_epoch, tile_branch_freeze_until_epoch)
        freeze_glyph = _epoch_in_freeze_range(epoch, glyph_branch_freeze_from_epoch, glyph_branch_freeze_until_epoch)
        freeze_word = _epoch_in_freeze_range(epoch, word_branch_freeze_from_epoch, word_branch_freeze_until_epoch)
        new_branch_frozen = {"tile": freeze_tile, "glyph": freeze_glyph, "word": freeze_word}
        freeze_state_changed = new_branch_frozen != branch_frozen

        if freeze_state_changed:
            if is_main:
                logger.info(f"\n{'='*80}")
                logger.info(f"Branch freeze state (epoch {epoch+1}): tile={freeze_tile}, glyph={freeze_glyph}, word={freeze_word}")
                logger.info(f"{'='*80}")
            for name, param in base_model.named_parameters():
                if "head.classifier" in name or "head.arcface_head" in name:
                    param.requires_grad = True
                elif name.startswith("tile_branch."):
                    param.requires_grad = not freeze_tile
                elif name.startswith("glyph_branch."):
                    param.requires_grad = not freeze_glyph
                elif name.startswith("word_branch."):
                    param.requires_grad = not freeze_word
                else:
                    param.requires_grad = True
                if not param.requires_grad:
                    param.grad = None
            # Optimizer groups intentionally retain frozen parameters. AdamW
            # skips params with grad=None, preserving moments and differential
            # LR schedules across later unfreezes.
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            trainable_params += [p for p in combined_loss.parameters() if p.requires_grad]
            if is_main:
                logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,} (incl. loss-module params)\n")
            branch_frozen = new_branch_frozen

        # Use GPU tensors for running sums to avoid .item() sync every batch.
        # Moved to GPU on first batch (after device is known).
        total_loss = torch.tensor(0.0, device=device)
        total_arcface_loss = torch.tensor(0.0, device=device)
        total_tile_aux_loss = torch.tensor(0.0, device=device)
        total_glyph_aux_loss = torch.tensor(0.0, device=device)
        total_fusion_aux_loss = torch.tensor(0.0, device=device)
        total_word_aux_loss = torch.tensor(0.0, device=device)
        total_geniza_contrastive_loss = torch.tensor(0.0, device=device)
        total_geniza_valid_anchor_frac = 0.0
        total_geniza_positive_pairs = 0.0
        total_geniza_top1_acc = 0.0
        geniza_steps = 0
        correct = torch.tensor(0, dtype=torch.long, device=device)
        total = 0
        train_diversity_total_batches = 0
        train_diversity_homogeneous_batches = 0
        train_diversity_low_diversity_batches = 0
        geniza_iter = iter(geniza_train_loader) if use_geniza_contrastive else None
        accumulated_backward_steps = 0
        successful_train_batches = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}") if is_main else train_loader):
            if batch is None:
                continue
            tiles, valid_mask, coords, tile_page_segments, char_patches, char_valid_mask, glyph_coords, glyph_page_segments, char_class_ids, _char_metadata, words, word_metadata, labels, paths = batch
            
            # DEBUG: Validate labels immediately after unpacking from batch
            if batch_idx == 0:
                logger.info(f"[DEBUG] Training loop - First batch after unpacking:")
                logger.info(f"  Labels type: {type(labels)}")
                if isinstance(labels, torch.Tensor):
                    logger.info(f"  Labels shape: {labels.shape}")
                    logger.info(f"  Labels dtype: {labels.dtype}")
                    logger.info(f"  Labels device: {labels.device}")
                    logger.info(f"  Labels range: [{labels.min().item()}, {labels.max().item()}]")
                    logger.info(f"  Labels sample: {labels[:min(5, len(labels))].tolist()}")
                else:
                    logger.info(f"  Labels is not a tensor: {labels}")
                    logger.info(f"  Labels content: {labels}")
            
            # DEBUG: Check for invalid labels before moving to device
            if isinstance(labels, torch.Tensor):
                if (labels < 0).any():
                    invalid_indices = (labels < 0).nonzero(as_tuple=True)[0]
                    logger.error(f"[DEBUG] Negative labels detected at batch {batch_idx} (before device move):")
                    logger.error(f"  Invalid indices: {invalid_indices.tolist()}")
                    logger.error(f"  Invalid label values: {labels[invalid_indices].tolist()}")
                    logger.error(f"  Corresponding paths: {[paths[i] for i in invalid_indices.tolist()]}")
                if torch.isnan(labels.float()).any():
                    nan_indices = torch.isnan(labels.float()).nonzero(as_tuple=True)[0]
                    logger.error(f"[DEBUG] NaN labels detected at batch {batch_idx} (before device move):")
                    logger.error(f"  NaN indices: {nan_indices.tolist()}")
                    logger.error(f"  Corresponding paths: {[paths[i] for i in nan_indices.tolist()]}")
                label_values = labels.detach().cpu().tolist()
                if label_values:
                    unique_labels = len(set(int(x) for x in label_values))
                    batch_size_for_diversity = len(label_values)
                    diversity_ratio = unique_labels / batch_size_for_diversity
                    train_diversity_total_batches += 1
                    if unique_labels == 1:
                        train_diversity_homogeneous_batches += 1
                    elif diversity_ratio < 0.3:
                        train_diversity_low_diversity_batches += 1
                
            # Move to device
            tiles = tiles.to(device, non_blocking=True)
            valid_mask = valid_mask.to(device, non_blocking=True)
            coords = coords.to(device, non_blocking=True)
            tile_page_segments = tile_page_segments.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            char_patches = char_patches.to(device, non_blocking=True)
            char_valid_mask = char_valid_mask.to(device, non_blocking=True)
            glyph_coords = glyph_coords.to(device, non_blocking=True)
            glyph_page_segments = glyph_page_segments.to(device, non_blocking=True)
            char_class_ids = char_class_ids.to(device, non_blocking=True)
            
            batch_element_indices = torch.arange(
                tiles.shape[0], dtype=torch.long, device=device
            )
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, latent, aux_latents = model(
                    tiles=tiles,
                    tile_coords=coords,
                    tile_valid_mask=valid_mask,
                    tile_page_segments=tile_page_segments,
                    glyph_patches=char_patches,
                    glyph_coords=glyph_coords,
                    glyph_valid_mask=char_valid_mask,
                    glyph_page_segments=glyph_page_segments,
                    char_class_ids=char_class_ids,
                    words=words,
                    word_metadata=word_metadata,
                    paths=list(paths) if paths is not None else None,
                    batch_element_indices=batch_element_indices,
                    return_aux_latents=True,
                )
            
            # Check for non-finite model outputs early (before normalization).
            # Use torch.isfinite (NOT torch.isnan) so we also catch +/-Inf, which
            # bf16 autocast can produce on exploded attention scores or post-clip
            # blow-ups. This matches the eval-time guard in evaluate().
            if not (torch.isfinite(logits).all() and torch.isfinite(latent).all()):
                bad_aux = [name for name, value in aux_latents.items()
                           if torch.is_tensor(value) and not torch.isfinite(value).all()]
                bad_params = [name for name, value in model.named_parameters()
                              if not torch.isfinite(value).all()]
                raise RuntimeError(
                    f"Non-finite model output at train batch {batch_idx}: "
                    f"logits_finite={bool(torch.isfinite(logits).all())}, "
                    f"latent_finite={bool(torch.isfinite(latent).all())}, "
                    f"bad_aux={bad_aux}, bad_parameters={bad_params[:20]}"
                )
            
            # Ensure all tensors are on the same device as the loss module
            # With DataParallel, model outputs are gathered on output_device (device 0),
            # but we need to ensure the loss module and inputs are on the same device
            
            # CRITICAL: Check for zero vectors before normalization (would produce NaN).
            # Cast to fp32 to avoid bf16 underflow falsely flagging small-but-valid latents;
            # matches the eval-time guard in evaluate().
            latent_norms = torch.norm(latent.float(), p=2, dim=1)
            if (latent_norms < 1e-8).any():
                zero_count = (latent_norms < 1e-8).sum().item()
                logger.error(f"Found {zero_count} zero/near-zero latent vectors at batch {batch_idx}. Skipping batch.")
                print(f"ERROR: Zero latent vectors detected. This indicates model produced invalid outputs.")
                continue
            
            # Use eps=1e-8 to match evaluation (extra safety beyond the zero-vector guard above).
            normalized_latent = F.normalize(latent, dim=1, eps=1e-8)
            
            # Validate labels are in valid range (CRITICAL: prevents index errors)
            num_classes = logits.shape[1]
            if (labels < 0).any() or (labels >= num_classes).any():
                invalid_mask = (labels < 0) | (labels >= num_classes)
                invalid_indices = invalid_mask.nonzero(as_tuple=True)[0]
                invalid_label_values = labels[invalid_indices]
                
                logger.error(f"[DEBUG] Invalid labels detected at batch {batch_idx}:")
                logger.error(f"  Invalid indices: {invalid_indices.tolist()}")
                logger.error(f"  Invalid label values: {invalid_label_values.tolist()}")
                logger.error(f"  Expected range: [0, {num_classes-1}]")
                logger.error(f"  Num classes (from logits): {num_classes}")
                logger.error(f"  Corresponding paths: {[paths[i] for i in invalid_indices.tolist()]}")
                logger.error(f"  Label2idx size: {len(idx2label) if idx2label else 'N/A'}")
                if idx2label:
                    logger.error(f"  Label2idx sample: {dict(list(idx2label.items())[:5])}")
                
                print(f"ERROR: Invalid labels at batch {batch_idx}. This indicates a data loading bug.")
                print(f"  Invalid labels: {invalid_label_values.tolist()}")
                print(f"  Expected range: [0, {num_classes-1}]")
                continue
            
            # All tensors should already be on the correct device (from model outputs)
            # No need to move them - loss functions don't have parameters, they work with any device
            loss_res = combined_loss(logits, normalized_latent, labels, aux_latents=aux_latents)
            loss, arcface_loss, sparsity_loss, effective_logits, aux_loss_dict = loss_res

            # Check for NaN classification loss before adding the Geniza objective.
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(
                    f"Non-finite classification loss at train batch {batch_idx}: "
                    f"loss={loss.detach().float().cpu().item()}, "
                    f"aux_losses={{{', '.join(f'{k}: {v.detach().float().cpu().item()}' for k, v in aux_loss_dict.items())}}}"
                )

            # Backprop the classification graph first to avoid keeping both the
            # classification and Geniza forward graphs resident at the same time.
            (loss / gradient_accumulation_steps).backward()
            accumulated_backward_steps += 1

            geniza_loss = torch.tensor(0.0, device=device)
            geniza_stats = {"valid_anchor_frac": 0.0, "positive_pairs": 0.0, "contrast_count": 0.0}
            geniza_latent_for_queue = None
            geniza_labels_for_queue = None
            if use_geniza_contrastive:
                try:
                    geniza_batch = next(geniza_iter)
                except StopIteration:
                    geniza_iter = iter(geniza_train_loader)
                    geniza_batch = next(geniza_iter)

                if geniza_batch is not None:
                    _g_logits, g_latent, _g_aux, g_labels, _g_paths = _forward_latent_batch(
                        model,
                        geniza_batch,
                        device=device,
                        return_aux_latents=False,
                    )
                    if torch.isfinite(g_latent).all() and (torch.norm(g_latent, p=2, dim=1) >= 1e-8).all():
                        g_norm = F.normalize(g_latent, dim=1, eps=1e-8)
                        bank_features, bank_labels = geniza_feature_queue.get()
                        geniza_loss, geniza_stats = geniza_contrastive_loss(
                            g_norm,
                            g_labels,
                            bank_features=bank_features,
                            bank_labels=bank_labels,
                        )
                        geniza_latent_for_queue = g_norm
                        geniza_labels_for_queue = g_labels
                    else:
                        logger.warning(f"[GENIZA CONTRASTIVE] Skipping non-finite/zero Geniza latent batch at train batch {batch_idx}.")
                else:
                    logger.warning(f"[GENIZA CONTRASTIVE] Skipping empty Geniza batch at train batch {batch_idx}.")

                weighted_geniza_loss = float(geniza_contrastive_weight) * geniza_loss
                if torch.isnan(weighted_geniza_loss) or torch.isinf(weighted_geniza_loss):
                    logger.error(f"NaN/Inf Geniza contrastive loss detected at batch {batch_idx}. Skipping Geniza backward for this batch.")
                    geniza_loss = torch.tensor(0.0, device=device)
                    geniza_latent_for_queue = None
                    geniza_labels_for_queue = None
                else:
                    (weighted_geniza_loss / gradient_accumulation_steps).backward()
            if geniza_latent_for_queue is not None and geniza_labels_for_queue is not None:
                geniza_feature_queue.enqueue(geniza_latent_for_queue, geniza_labels_for_queue)
            total_step_loss = loss.detach() + (float(geniza_contrastive_weight) * geniza_loss.detach())
            
            # Update weights every gradient_accumulation_steps
            if accumulated_backward_steps >= gradient_accumulation_steps:
                clip_accumulated_gradients()
                optimizer.step()
                optimizer.zero_grad()
                accumulated_backward_steps = 0
            
            batch_size = tiles.shape[0]
            # Accumulate losses/metrics as GPU tensors — avoid .item() every batch
            # (.item() synchronises the CUDA stream and stalls the pipeline).
            total_loss += total_step_loss.detach() * batch_size
            total_arcface_loss += arcface_loss.detach() * batch_size
            total_tile_aux_loss += aux_loss_dict.get('tile', torch.tensor(0.0, device=device)).detach() * batch_size
            total_glyph_aux_loss += aux_loss_dict.get('glyph', torch.tensor(0.0, device=device)).detach() * batch_size
            total_fusion_aux_loss += aux_loss_dict.get('fusion', torch.tensor(0.0, device=device)).detach() * batch_size
            total_word_aux_loss += aux_loss_dict.get('word', torch.tensor(0.0, device=device)).detach() * batch_size
            if use_geniza_contrastive:
                total_geniza_contrastive_loss += geniza_loss.detach()
                total_geniza_valid_anchor_frac += float(geniza_stats.get("valid_anchor_frac", 0.0))
                total_geniza_positive_pairs += float(geniza_stats.get("positive_pairs", 0.0))
                total_geniza_top1_acc += float(geniza_stats.get("top1_acc", 0.0))
                geniza_steps += 1
            # Keep training accuracy and logged predictions aligned with `effective_logits`.
            preds = effective_logits.argmax(dim=1)
            correct += (preds == labels).sum()
            total += batch_size
            successful_train_batches += 1
            
            # Clear CUDA cache infrequently to limit GPU synchronisation overhead.
            # Doing this every optimizer step or every 5 batches was a major drag on
            # multi-GPU throughput (empty_cache forces a full device sync).
            if torch.cuda.is_available() and successful_train_batches % (gradient_accumulation_steps * 4) == 0:
                torch.cuda.empty_cache()

        # Apply any remaining accumulated gradients from the last incomplete accumulation window.
        # Without this, up to (gradient_accumulation_steps - 1) batches of gradients are silently lost.
        if accumulated_backward_steps > 0:
            clip_accumulated_gradients()
            optimizer.step()
            optimizer.zero_grad()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

        if total == 0:
            raise RuntimeError(
                f"Epoch {epoch + 1} produced no valid training samples. "
                "All batches were empty or skipped before loss computation; check image/XML loading and label mappings."
            )

        # Single .item() sync at epoch boundary (instead of every batch)
        train_loss = (total_loss / total).item()
        train_arcface_loss = (total_arcface_loss / total).item()
        train_tile_aux_loss = (total_tile_aux_loss / total).item()
        train_glyph_aux_loss = (total_glyph_aux_loss / total).item()
        train_fusion_aux_loss = (total_fusion_aux_loss / total).item()
        train_word_aux_loss = (total_word_aux_loss / total).item()
        train_acc = (correct / total).item()
        train_geniza_contrastive_loss = (
            (total_geniza_contrastive_loss / max(1, geniza_steps)).item()
            if use_geniza_contrastive else 0.0
        )
        train_geniza_weighted_contrastive_loss = (
            float(geniza_contrastive_weight) * train_geniza_contrastive_loss
            if use_geniza_contrastive else 0.0
        )
        train_geniza_valid_anchor_frac = (
            total_geniza_valid_anchor_frac / max(1, geniza_steps)
            if use_geniza_contrastive else 0.0
        )
        train_geniza_positive_pairs = (
            total_geniza_positive_pairs / max(1, geniza_steps)
            if use_geniza_contrastive else 0.0
        )
        train_geniza_top1_acc = (
            total_geniza_top1_acc / max(1, geniza_steps)
            if use_geniza_contrastive else 0.0
        )

        # IMPORTANT (DDP): keep ranks in sync at epoch boundaries.
        # Rank0 may do validation/visualization while other ranks must wait.
        ddp_barrier()
        # Rank0-only: validation, wandb logging, attention viz, PCA analysis.
        geniza_metrics = {}
        val_retrieval_metrics = {}
        val_checkpoint_score = None
        val_checkpoint_map = None
        if is_main and val_loader is not None:
            eval_res = evaluate(
                model,
                val_loader,
                combined_loss,
                idx2label,
                fixed_val_paths,
                device=device,
                return_stats=True,
            )
            val_loss, val_arcface_loss, val_tile_aux_loss, val_glyph_aux_loss, val_fusion_aux_loss, val_word_aux_loss, val_acc, misclassified, val_samples, val_stats = eval_res
            val_retrieval_metrics = evaluate_retrieval(
                model,
                val_loader,
                device=device,
                metric_prefix="val/retrieval",
            )
            val_checkpoint = _validation_checkpoint_score(val_retrieval_metrics, val_acc)
            val_checkpoint_score = val_checkpoint[0] if val_checkpoint is not None else None
            val_checkpoint_map = val_checkpoint[1] if val_checkpoint is not None else None
            test_cluster_metrics = (
                evaluate_cluster_test_suite(
                    model,
                    cluster_test_loaders,
                    device=device,
                )
                if cluster_test_loaders else {}
            )
            geniza_metrics = (
                evaluate_retrieval(
                    model,
                    geniza_val_loader,
                    device=device,
                    contrastive_loss_fn=geniza_contrastive_loss,
                    contrastive_weight=float(geniza_contrastive_weight),
                    metric_prefix="geniza/val/retrieval",
                )
                if use_geniza_contrastive else {}
            )

            # Attention maps: load on-demand (not at startup) to avoid memory issues
            if cached_attention_indices is not None:
                # Load images on-demand each epoch (not at startup) to avoid memory issues
                dataset = getattr(val_loader, "dataset", None)
                if dataset is not None:
                    samples = [dataset[i] for i in cached_attention_indices]
                    cached_attention_batch = tile_collate_with_padding(samples)

                    if cached_attention_batch is None:
                        logger.warning("[Attention Maps] All cached attention samples failed to load; skipping attention maps this epoch.")
                    else:
                        # Look up XML paths from the dataset for these samples
                        # (so attention visualization can find character/word regions)
                        attn_xml_paths = None
                        if hasattr(dataset, "xml_paths") and dataset.xml_paths is not None:
                            xml_by_path = {
                                os.path.normpath(dataset.image_paths[i]): dataset.xml_paths[i]
                                for i in cached_attention_indices
                            }
                            attn_xml_paths = [
                                xml_by_path.get(os.path.normpath(path))
                                for path in cached_attention_batch[-1]
                            ]
                        actual_attention_paths = list(cached_attention_batch[-1])
                        
                        logger.info(f"[Attention Maps] ========================================")
                        logger.info(f"[Attention Maps] Generating attention maps for epoch {epoch + 1}")
                        logger.info(f"[Attention Maps] Processing {len(actual_attention_paths)}/{NUM_ATTENTION_IMAGES} cached images:")
                        for idx, path in enumerate(actual_attention_paths, 1):
                            logger.info(f"[Attention Maps]   {idx}. {path}")
                        logger.info(f"[Attention Maps] ========================================")
                        
                        # NOTE: cached_attention_batch may contain fewer samples than NUM_ATTENTION_IMAGES.
                        # Always pass the actual count to avoid indexing mismatches inside visualization.
                        log_attention_maps(model, cached_attention_batch, epoch + 1,
                                           num_images=len(actual_attention_paths),
                                           xml_paths=attn_xml_paths)
                        
                        # Clear the batch from memory after use to free GPU memory
                        del cached_attention_batch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                else:
                    logger.warning("[Attention Maps] Dataset not available; skipping attention maps.")
            else:
                logger.warning("[Attention Maps] No cached attention indices available; skipping attention maps this epoch to keep DDP in sync.")

            current_lr = optimizer.param_groups[0]['lr']
            current_lrs = {
                str(group.get("name", f"group_{index}")): float(group["lr"])
                for index, group in enumerate(optimizer.param_groups)
            }

            # Build per-epoch tables (fresh objects) so W&B always renders updated rows.
            # Standard columns for all tables
            table_columns = ["epoch", "image_path", "true_label", "predicted_label", "correct", "confidence", "top_predictions"]
            
            error_table_epoch = wandb.Table(columns=table_columns)
            val_batch_table_epoch = wandb.Table(columns=table_columns)

            # Add top N misclassified samples from THIS epoch (SORTED BY CONFIDENCE DESCENDING)
            if misclassified:
                # Sort by confidence descending (High confidence errors first - "Arrogant but Wrong")
                misclassified_sorted = sorted(misclassified, key=lambda x: x['confidence'], reverse=True)
                top_misclassified = misclassified_sorted[:NUM_TOP_ERRORS_PER_EPOCH]
                for item in top_misclassified:
                    error_table_epoch.add_data(
                        epoch + 1,
                        item['path'],
                        item['true_label'],
                        item['predicted_label'],
                        item['correct'],
                        f"{item['confidence']:.3f}",
                        item['top_predictions']
                    )

            # Add fixed validation batch predictions from THIS epoch
            if val_samples:
                for sample in val_samples:
                    val_batch_table_epoch.add_data(
                        epoch + 1,
                        sample["path"],
                        sample["true_label"],
                        sample["predicted_label"],
                        sample["correct"],
                        f"{sample['confidence']:.3f}",
                        sample["top_predictions"]
                    )
            
            epoch_step = epoch + 1
            epoch_log_data = {
                "epoch": epoch_step,
                "learning_rate": current_lr,
                **{
                    f"learning_rate/{name}": value
                    for name, value in current_lrs.items()
                },
                "train/loss": train_loss,
                "train/cls_loss": train_arcface_loss,
                "train/tile_aux_loss": train_tile_aux_loss,
                "train/glyph_aux_loss": train_glyph_aux_loss,
                "train/fusion_aux_loss": train_fusion_aux_loss,
                "train/word_aux_loss": train_word_aux_loss,
                "train/word_aux_weight": float(
                    getattr(combined_loss, "current_word_aux_weight", 0.0)
                ),
                "train/accuracy": train_acc,
                "val/loss": val_loss,
                "val/cls_loss": val_arcface_loss,
                "val/tile_aux_loss": val_tile_aux_loss,
                "val/glyph_aux_loss": val_glyph_aux_loss,
                "val/fusion_aux_loss": val_fusion_aux_loss,
                "val/word_aux_loss": val_word_aux_loss,
                "val/accuracy": val_acc,
                "val/checkpoint_score": val_checkpoint_score if val_checkpoint_score is not None else val_acc,
                "validation_errors": error_table_epoch,
                "validation_batch_predictions": val_batch_table_epoch,
            }
            if test_cluster_metrics:
                _validate_cluster_test_wandb_metrics(test_cluster_metrics)
                epoch_log_data.update(test_cluster_metrics)
            if val_retrieval_metrics:
                epoch_log_data.update(val_retrieval_metrics)
            if use_geniza_contrastive:
                epoch_log_data.update({
                    # Weighted loss is the actual contribution to the objective.
                    "train/contrastive_loss": train_geniza_weighted_contrastive_loss,
                    "train/contrastive_accuracy": train_geniza_top1_acc,
                    "train/contrastive_valid_anchor_frac": train_geniza_valid_anchor_frac,
                    "train/contrastive_positive_pairs_per_step": train_geniza_positive_pairs,
                })
            epoch_log_data.update(collect_memory_usage())
            if train_diversity_total_batches > 0:
                homogeneous_pct = train_diversity_homogeneous_batches / train_diversity_total_batches
                low_diversity_pct = train_diversity_low_diversity_batches / train_diversity_total_batches
                logger.info(
                    "[Train Batch Diversity] "
                    f"total_batches={train_diversity_total_batches}, "
                    f"homogeneous={train_diversity_homogeneous_batches} ({homogeneous_pct:.1%}), "
                    f"low_diversity={train_diversity_low_diversity_batches} ({low_diversity_pct:.1%})"
                )
            if use_geniza_contrastive:
                epoch_log_data.update(geniza_metrics)
                if not geniza_metrics:
                    logger.warning(
                        "[GENIZA RETRIEVAL] No validation retrieval metrics were produced. "
                        "Check that geniza_val_loader is non-empty and latents are finite/non-zero."
                    )

            # Also log a couple of high-signal counts to console for quick debugging.
            if isinstance(val_stats, dict):
                logger.info(
                    "[VAL Tables] "
                    f"fixed_requested={val_stats.get('eval.fixed_paths_requested')} "
                    f"fixed_seen={val_stats.get('eval.fixed_paths_seen')} "
                    f"fixed_matched={val_stats.get('eval.fixed_paths_matched')} "
                    f"rows_in_val_table={len(val_samples) if val_samples else 0} "
                    f"rows_in_error_table={len(misclassified) if misclassified else 0}"
                )
                print(
                    "[VAL Tables] "
                    f"fixed_requested={val_stats.get('eval.fixed_paths_requested')} "
                    f"fixed_seen={val_stats.get('eval.fixed_paths_seen')} "
                    f"fixed_matched={val_stats.get('eval.fixed_paths_matched')} "
                    f"rows_in_val_table={len(val_samples) if val_samples else 0} "
                    f"rows_in_error_table={len(misclassified) if misclassified else 0}"
                )

            geniza_msg = ""
            if geniza_metrics:
                geniza_msg = (
                    f", Geniza val mAP {geniza_metrics.get('geniza/val/retrieval/mAP', 0.0):.4f}, "
                    f"Geniza val KNN@1 {geniza_metrics.get('geniza/val/retrieval/knn_at_1', 0.0):.4f}"
                )
            print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}, Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}{geniza_msg}, LR {current_lr:.6f}")
            if test_cluster_metrics:
                for source_name in ("clusters_images_metadata", "cluster_members"):
                    prefix = f"test/{source_name}/mAP"
                    print(
                        f"  Cluster test [{source_name}]: "
                        f"fusion={test_cluster_metrics[f'{prefix}/fusion']:.4f}, "
                        f"tile={test_cluster_metrics[f'{prefix}/tile']:.4f}, "
                        f"glyph={test_cluster_metrics[f'{prefix}/glyph']:.4f}, "
                        f"word={test_cluster_metrics[f'{prefix}/word']:.4f}"
                    )

            # Save XML warnings and log batch diversity stats after first epoch (since first epoch iterates over all manuscripts)
            if epoch == 0:
                # Save XML warnings to logs directory (filename will include timestamp)
                save_xml_warnings_to_file(output_dir="logs")

            # PCA diagnostics are added to the same per-epoch W&B payload below.
            # Do not call wandb.log inside this block; that keeps one value/image
            # per key per epoch and prevents duplicate W&B panels.
            do_pca = bool(ENABLE_PCA_ANALYSIS) and (int(PCA_EVERY_EPOCHS) > 0) and (((epoch + 1) % int(PCA_EVERY_EPOCHS)) == 0)
            if do_pca:
                print(f"Computing PCA analysis for epoch {epoch+1}...")
                # Use more samples when we have many classes so PCA isn't limited by n_samples < num_classes
                base_model = model.module if hasattr(model, "module") else model
                num_classes_pca = getattr(base_model.head.classifier[-1], "out_features", None)
                if num_classes_pca is not None:
                    max_pca_samples = max(int(PCA_MAX_SAMPLES), min(2 * num_classes_pca, 3000))
                else:
                    max_pca_samples = int(PCA_MAX_SAMPLES)
                val_latents = collect_validation_latents(model, val_loader, max_samples=max_pca_samples)
                n_pca_samples = len(val_latents) if val_latents is not None else 0

                if val_latents is not None and len(val_latents) > 0:
                    # Initialize outputs so we don't crash if PCA is skipped.
                    elbow_dim = None
                    threshold_dim = None
                    elbow_curvature_dim = None
                    explained_vars = None
                    cumulative_vars = None
                    pca_fitted = None
                    scaler_fitted = None

                    # Check for NaN values before PCA
                    if np.isnan(val_latents).any():
                        logger.warning(f"PCA skipped: Found NaN values in validation latents. This may indicate training instability.")
                        print(f"Warning: PCA skipped due to NaN values in latents. Check training loss for NaN.")
                    else:
                        # Compute optimal dimensions (elbow threshold, 95% variance, elbow curvature).
                        # Pass max_components=None so PCA is fitted with full rank and optimal_dim reflects the real effective dimension per epoch.
                        elbow_dim, threshold_dim, elbow_curvature_dim, explained_vars, cumulative_vars, pca_fitted, scaler_fitted = compute_pca_elbow_dimension(
                            val_latents, max_components=None
                        )
                    # Warn when PCA sample count is below num_classes (optimal dim may be underestimated)
                    if num_classes_pca is not None and n_pca_samples < num_classes_pca:
                        logger.warning(
                            f"[PCA] Samples ({n_pca_samples}) < num_classes ({num_classes_pca}). "
                            "Optimal dimension may be underestimated; consider increasing PCA_MAX_SAMPLES or using more validation data."
                        )
                        print(f"  PCA: Using {n_pca_samples} samples for {num_classes_pca} classes; optimal dimension estimate may be conservative.")
                    
                    if elbow_dim is not None and threshold_dim is not None and cumulative_vars is not None:
                        # DEBUG: Validate inputs before computing percentage
                        logger.info(f"[DEBUG] PCA computation:")
                        logger.info(f"  threshold_dim: {threshold_dim} (type: {type(threshold_dim)})")
                        logger.info(f"  LATENT_DIM: {LATENT_DIM} (type: {type(LATENT_DIM)})")
                        
                        # Calculate utilization percentage: (optimal_dim_95 / LATENT_DIM) * 100
                        if threshold_dim is None or (isinstance(threshold_dim, (int, float, np.number)) and np.isnan(threshold_dim)):
                            logger.error(f"[DEBUG] Cannot compute utilization percentage: threshold_dim is None or NaN")
                            utilization_percentage = float('nan')
                        elif LATENT_DIM == 0:
                            logger.error(f"[DEBUG] Cannot compute utilization percentage: LATENT_DIM is 0")
                            utilization_percentage = float('nan')
                        else:
                            utilization_percentage = (threshold_dim / LATENT_DIM) * 100
                            logger.info(f"[DEBUG] Computed utilization_percentage: {utilization_percentage}%")
                        
                        # Log sample/class context so PCA metrics are interpretable
                        pca_log_dict = {
                            "pca/optimal_dimension_elbow": elbow_dim,
                            "pca/optimal_dimension_95_threshold": threshold_dim,
                            "pca/n_samples": n_pca_samples,
                            "pca/explained_variance_95": cumulative_vars[min(95, len(cumulative_vars)-1)] if len(cumulative_vars) > 95 else cumulative_vars[-1],
                            "pca/explained_variance_elbow": cumulative_vars[elbow_dim-1] if elbow_dim and elbow_dim <= len(cumulative_vars) else cumulative_vars[-1],
                            "pca/explained_variance_threshold": cumulative_vars[threshold_dim-1] if threshold_dim <= len(cumulative_vars) else cumulative_vars[-1],
                        }
                        if elbow_curvature_dim is not None:
                            pca_log_dict["pca/optimal_dimension_elbow_curvature"] = elbow_curvature_dim
                        if num_classes_pca is not None:
                            pca_log_dict["pca/num_classes"] = num_classes_pca
                        # Validate percentage before logging
                        if np.isnan(utilization_percentage) or np.isinf(utilization_percentage):
                            logger.error(f"[DEBUG] Invalid utilization_percentage: {utilization_percentage}")
                            logger.error(f"  threshold_dim: {threshold_dim}, LATENT_DIM: {LATENT_DIM}")
                            print(f"  PCA Elbow dimension: {elbow_dim} (explains {cumulative_vars[elbow_dim-1]:.2%} of variance)")
                            print(f"  PCA 95% threshold dimension: {threshold_dim} (explains {cumulative_vars[threshold_dim-1]:.2%} of variance)")
                            if elbow_curvature_dim is not None:
                                print(f"  PCA Elbow (curvature) dimension: {elbow_curvature_dim}")
                            print(f"  Warning: Could not compute dimension utilization percentage (threshold_dim={threshold_dim}, LATENT_DIM={LATENT_DIM})")
                        else:
                            pca_log_dict["pca/dimension_utilization_percentage"] = utilization_percentage
                            print(f"  PCA Elbow dimension: {elbow_dim} (explains {cumulative_vars[elbow_dim-1]:.2%} of variance)")
                            print(f"  PCA 95% threshold dimension: {threshold_dim} (explains {cumulative_vars[threshold_dim-1]:.2%} of variance)")
                            if elbow_curvature_dim is not None:
                                print(f"  PCA Elbow (curvature) dimension: {elbow_curvature_dim}")
                            print(f"  Dimension utilization: {utilization_percentage:.2f}% ({threshold_dim}/{LATENT_DIM} dimensions)")

                        # Save locally and attach each PCA graph once to the consolidated epoch payload.
                        threshold_fig = visualize_pca_95_variance_threshold(
                            explained_vars, cumulative_vars, threshold_dim, 
                            epoch=epoch+1, save_dir=PCA_SAVE_DIR
                        )
                        if threshold_fig is not None:
                            pca_log_dict["pca/variance_threshold_plot"] = wandb.Image(
                                threshold_fig,
                                caption=f"PCA 95% variance threshold - Epoch {epoch+1}",
                            )
                            plt.close(threshold_fig)  # Free memory

                        elbow_fig = visualize_pca_elbow_method(
                            explained_vars, cumulative_vars, elbow_dim, 
                            epoch=epoch+1, save_dir=PCA_SAVE_DIR
                        )
                        if elbow_fig is not None:
                            pca_log_dict["pca/elbow_plot"] = wandb.Image(
                                elbow_fig,
                                caption=f"PCA elbow method - Epoch {epoch+1}",
                            )
                            plt.close(elbow_fig)  # Free memory

                        utilization_fig = visualize_pca_dimension_utilization(
                            explained_vars, original_latent_dim=LATENT_DIM,
                            epoch=epoch+1, save_dir=PCA_SAVE_DIR
                        )
                        if utilization_fig is not None:
                            pca_log_dict["pca/dimension_utilization_plot"] = wandb.Image(
                                utilization_fig,
                                caption=f"PCA dimension utilization - Epoch {epoch+1}",
                            )
                            plt.close(utilization_fig)  # Free memory
                        epoch_log_data.update(pca_log_dict)

                    # Create correlation map of original latent dimensions (redundancy check)
                    pca_corr_fig, correlation_matrix, n_corr_dims, _ = create_pca_correlation_map(
                        val_latents, pca=pca_fitted, scaler=scaler_fitted,
                        pca_dim=PCA_DIMENSION, epoch=epoch+1, save_dir=PCA_SAVE_DIR
                    )

                    if pca_corr_fig is not None:
                        # Compute off-diagonal correlation (high = redundant latent dims, low = decorrelated)
                        mask = ~np.eye(n_corr_dims, dtype=bool)
                        off_diagonal_corr = correlation_matrix[mask]
                        
                        epoch_log_data.update({
                            "pca/correlation_map": wandb.Image(
                                pca_corr_fig,
                                caption=f"Latent dimension correlation - Epoch {epoch+1} (off-diagonal mean: {off_diagonal_corr.mean():.4f})"
                            ),
                            "pca/n_components": n_corr_dims,
                            "pca/off_diagonal_correlation_mean": off_diagonal_corr.mean(),
                            "pca/off_diagonal_correlation_std": off_diagonal_corr.std(),
                        })
                        plt.close(pca_corr_fig)  # Free memory

            # Commit one consolidated epoch payload so each scalar/table/image is
            # written once per epoch step. Attention media logged earlier uses
            # commit=False and is flushed by this call.
            try:
                unexpected_test_keys = {
                    key for key in epoch_log_data
                    if key.startswith("test/") and key not in CLUSTER_TEST_WANDB_KEYS
                }
                if unexpected_test_keys:
                    raise RuntimeError(
                        "Refusing to send unapproved test fields to W&B: "
                        f"{sorted(unexpected_test_keys)}"
                    )
                logger.info(
                    f"[W&B] Committing epoch {epoch_step} payload "
                    f"({len(epoch_log_data)} keys) at step={epoch_step}"
                )
                wandb.log(epoch_log_data, step=epoch_step, commit=True)
                logger.info(f"[W&B] Committed epoch {epoch_step} payload")
            except Exception:
                logger.exception(f"[W&B] Failed to log epoch {epoch_step} payload")
                raise
        else:
            # Non-rank0 (or no val loader): keep vars for any later rank0-only sections.
            current_lr = optimizer.param_groups[0]['lr']
            val_acc = 0.0

        # Sync again so rank0 finishes eval/logging before any rank proceeds to next epoch's training.
        ddp_barrier()
        
        # Step the scheduler at the end of the epoch
        scheduler.step()

        geniza_checkpoint_score = None
        if use_geniza_contrastive and geniza_metrics:
            geniza_retrieval_score = geniza_metrics.get("geniza/val/retrieval/mAP")
            if geniza_retrieval_score is not None and math.isfinite(float(geniza_retrieval_score)):
                geniza_checkpoint_score = (
                    VAL_RETRIEVAL_CHECKPOINT_WEIGHT * float(geniza_retrieval_score)
                    + VAL_ACCURACY_CHECKPOINT_WEIGHT * float(val_acc)
                )

        if is_main:
            # Get num_classes from the model's head classifier (unwrap DDP if needed)
            base_model = model.module if hasattr(model, "module") else model
            num_classes = base_model.head.classifier[-1].out_features

            # Save the latest checkpoint every epoch, overwriting the previous one for this run.
            checkpoint = {
                'state_dict': base_model.state_dict(),
                'combined_loss_state_dict': combined_loss.state_dict(),
                'model_config': {
                    'num_classes': num_classes,
                    'tile_size': TILE_SIZE,
                    'use_visual_mod': base_model.use_visual_mod,
                    'use_char_mod': base_model.use_char_mod,
                    'use_word_mod': base_model.use_word_mod,
                    'd_model': base_model.d_model,
                    'latent_dim': LATENT_DIM,
                    'use_branch_adapters_and_summarizers': (
                        base_model.use_branch_adapters_and_summarizers
                    ),
                    'symmetric_branch_dim': getattr(
                        base_model.fusion, 'branch_dim', None
                    ),
                    'tile_summary_tokens': (
                        base_model.tile_branch.set_summarizer.num_queries
                        if base_model.use_visual_mod
                        and getattr(base_model.tile_branch, 'set_summarizer', None) is not None
                        else 0
                    ),
                    'tile_summarizer_cross_attn_layers': (
                        base_model.tile_branch.set_summarizer.num_cross_attn_layers
                        if base_model.use_visual_mod
                        and getattr(base_model.tile_branch, 'set_summarizer', None) is not None
                        else 0
                    ),
                    'glyph_summary_tokens': (
                        base_model.glyph_branch.set_summarizer.num_queries
                        if base_model.use_char_mod
                        and getattr(base_model.glyph_branch, 'set_summarizer', None) is not None
                        else 0
                    ),
                    'word_summary_tokens': (
                        base_model.word_branch.word_set_summarizer.num_queries
                        if base_model.use_word_mod
                        and getattr(base_model.word_branch, 'word_set_summarizer', None) is not None
                        else 0
                    ),
                },
                'training_mode': mode,
                'epoch': epoch + 1,
                'val_accuracy': val_acc,
                'val_retrieval_map': val_checkpoint_map,
                'geniza_map': geniza_metrics.get("geniza/val/retrieval/mAP") if geniza_metrics else None,
                'checkpoint_score': (
                    geniza_checkpoint_score
                    if geniza_checkpoint_score is not None
                    else val_checkpoint_score if val_checkpoint_score is not None
                    else val_acc
                ),
                'best_val_accuracy': best_val_accuracy if best_epoch > 0 else None,
                'best_geniza_map': best_geniza_map if best_epoch > 0 and use_geniza_contrastive else None,
                'best_val_retrieval_map': best_val_retrieval_map if best_epoch > 0 and best_checkpoint_metric_source == "val_retrieval_accuracy" else None,
                'best_checkpoint_metric_source': best_checkpoint_metric_source,
                'best_checkpoint_score': best_checkpoint_score if best_epoch > 0 else None,
                'best_epoch': best_epoch,
                'best_model_path': best_model_path,
                'last_model_path': last_model_path,
            }
            torch.save(checkpoint, last_model_path)
            logger.info(f"[Checkpoint] Saved last checkpoint for epoch {epoch + 1}: {last_model_path}")
            wandb.summary['last_model_path'] = last_model_path
            wandb.summary['last_checkpoint_epoch'] = epoch + 1

        # Save the best model from validation-only signals. Test/cluster-pairs
        # metrics are logged for debugging but must not influence selection.
        if geniza_checkpoint_score is not None:
            current_best_metric = geniza_checkpoint_score
            current_best_source = "geniza_retrieval_accuracy"
        elif val_checkpoint_score is not None and math.isfinite(float(val_checkpoint_score)):
            current_best_metric = val_checkpoint_score
            current_best_source = "val_retrieval_accuracy"
        else:
            current_best_metric = val_acc
            current_best_source = "val_accuracy"
        previous_best_metric = best_checkpoint_score if current_best_source != "val_accuracy" else best_val_accuracy
        if is_main and current_best_metric > previous_best_metric:
            if current_best_source == "geniza_retrieval_accuracy":
                best_geniza_map = geniza_metrics.get("geniza/val/retrieval/mAP", float("-inf"))
                best_checkpoint_score = current_best_metric
            elif current_best_source == "val_retrieval_accuracy":
                best_val_retrieval_map = val_checkpoint_map if val_checkpoint_map is not None else float("-inf")
                best_checkpoint_score = current_best_metric
            else:
                best_checkpoint_score = current_best_metric
            best_val_accuracy = max(best_val_accuracy, val_acc)
            best_checkpoint_metric_source = current_best_source
            best_epoch = epoch + 1
            checkpoint['best_val_accuracy'] = best_val_accuracy
            checkpoint['best_geniza_map'] = best_geniza_map if current_best_source == "geniza_retrieval_accuracy" else None
            checkpoint['best_val_retrieval_map'] = best_val_retrieval_map if current_best_source == "val_retrieval_accuracy" else None
            checkpoint['best_checkpoint_metric_source'] = best_checkpoint_metric_source
            checkpoint['best_checkpoint_score'] = best_checkpoint_score
            checkpoint['best_epoch'] = best_epoch
            torch.save(checkpoint, best_model_path)
            if current_best_source == "geniza_retrieval_accuracy":
                print(
                    f"Epoch {epoch + 1}: New best model saved with Geniza validation retrieval/accuracy score: "
                    f"{current_best_metric:.4f} (Geniza mAP: {best_geniza_map:.4f}, val acc: {val_acc:.4f})"
                )
            elif current_best_source == "val_retrieval_accuracy":
                print(
                    f"Epoch {epoch + 1}: New best model saved with validation retrieval/accuracy score: "
                    f"{current_best_metric:.4f} (val mAP: {best_val_retrieval_map:.4f}, val acc: {val_acc:.4f})"
                )
            else:
                print(f"Epoch {epoch + 1}: New best model saved with validation accuracy: {val_acc:.4f}")
            # Log best metrics to wandb for easy tracking
            wandb.summary['best_val_accuracy'] = best_val_accuracy
            wandb.summary['best_checkpoint_metric_source'] = best_checkpoint_metric_source
            wandb.summary['best_checkpoint_score'] = best_checkpoint_score
            if current_best_source == "val_retrieval_accuracy":
                wandb.summary['best_val_retrieval_map'] = best_val_retrieval_map
            wandb.summary['best_model_path'] = best_model_path
            wandb.summary['last_model_path'] = last_model_path
            if current_best_source == "geniza_retrieval_accuracy":
                wandb.summary['best_geniza_map'] = best_geniza_map
            wandb.summary['best_epoch'] = epoch + 1

    # After training, load the best model for final evaluation
    if is_main:
        model_path_to_load = best_model_path if best_epoch > 0 and os.path.exists(best_model_path) else last_model_path
        if best_checkpoint_metric_source == "geniza_retrieval_accuracy":
            print(
                f"\nTraining finished. Loading best model from epoch {best_epoch} "
                f"with Geniza validation retrieval/accuracy score {best_checkpoint_score:.4f} "
                f"(Geniza mAP {best_geniza_map:.4f})."
            )
        elif best_checkpoint_metric_source == "val_retrieval_accuracy":
            print(
                f"\nTraining finished. Loading best model from epoch {best_epoch} "
                f"with validation retrieval/accuracy score {best_checkpoint_score:.4f} "
                f"(val mAP {best_val_retrieval_map:.4f})."
            )
        else:
            print(f"\nTraining finished. Loading best model from epoch {best_epoch} with validation accuracy {best_val_accuracy:.4f}.")
        load_training_checkpoint_for_evaluation(
            model,
            combined_loss,
            model_path_to_load,
            device=device,
            logger=logger,
        )

    # Re-run fixed cluster tests with the validation-selected best model only
    # when a caller explicitly supplied those loaders.
    if is_main and cluster_test_loaders:
        print("Evaluating both fixed cluster tests with the best model...")
        final_cluster_metrics = evaluate_cluster_test_suite(
            model,
            cluster_test_loaders,
            device=device,
        )
        _validate_cluster_test_wandb_metrics(final_cluster_metrics)
        wandb.summary.update(final_cluster_metrics)

    if is_main:
        wandb.finish()
    return model
