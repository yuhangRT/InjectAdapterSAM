"""HQ refinement decoder for WireCR-HQInstSAM."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["HQMaskDecoder"]


class HQMaskDecoder(nn.Module):
    """Minimal HQ-style decoder using prompt-aware fusion over adapted features."""

    def __init__(
        self,
        base_mask_decoder: nn.Module,
        *,
        in_channels: int = 256,
        prompt_dim: int = 256,
    ) -> None:
        super().__init__()
        self.transformer_dim = int(base_mask_decoder.transformer_dim)
        self.transformer = base_mask_decoder.transformer
        self.num_multimask_outputs = int(base_mask_decoder.num_multimask_outputs)
        self.iou_token = base_mask_decoder.iou_token
        self.num_mask_tokens = int(base_mask_decoder.num_mask_tokens)
        self.mask_tokens = base_mask_decoder.mask_tokens
        self.output_upscaling = base_mask_decoder.output_upscaling
        self.output_hypernetworks_mlps = base_mask_decoder.output_hypernetworks_mlps
        self.iou_prediction_head = base_mask_decoder.iou_prediction_head

        self.final_feature_proj = nn.Sequential(
            nn.Conv2d(in_channels, prompt_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, prompt_dim),
            nn.GELU(),
        )
        self.early_feature_proj = nn.Sequential(
            nn.Conv2d(in_channels, prompt_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, prompt_dim),
            nn.GELU(),
        )
        self.hq_token = nn.Embedding(1, prompt_dim)
        self.hq_gate = nn.Sequential(
            nn.Linear(prompt_dim, prompt_dim),
            nn.GELU(),
            nn.Linear(prompt_dim, prompt_dim),
            nn.Sigmoid(),
        )
        self.refine_upscale = nn.Sequential(
            nn.ConvTranspose2d(prompt_dim, prompt_dim // 2, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(prompt_dim // 2, prompt_dim // 4, kernel_size=2, stride=2),
            nn.GELU(),
        )
        self.hq_hypernet = nn.Sequential(
            nn.Linear(prompt_dim, prompt_dim),
            nn.GELU(),
            nn.Linear(prompt_dim, prompt_dim // 4),
        )
        self.dense_prompt_scale = nn.Parameter(torch.tensor(1.0))
        self.early_fusion_scale = nn.Parameter(torch.tensor(1.0))
        self.refine_feature_channels = 1 + (prompt_dim // 4) + 1

    def _predict_masks(
        self,
        *,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        batch_size, channels, height, width = src.shape

        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        src = src.transpose(1, 2).view(batch_size, channels, height, width)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in = torch.stack(
            [self.output_hypernetworks_mlps[index](mask_tokens_out[:, index, :]) for index in range(self.num_mask_tokens)],
            dim=1,
        )
        batch_size, upscaled_channels, upscaled_height, upscaled_width = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(batch_size, upscaled_channels, upscaled_height * upscaled_width)).view(
            batch_size,
            -1,
            upscaled_height,
            upscaled_width,
        )
        iou_pred = self.iou_prediction_head(iou_token_out)
        return masks, iou_pred

    def forward(
        self,
        *,
        final_features: torch.Tensor,
        early_features: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        dense_prompt_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if final_features.shape[0] != 1 or early_features.shape[0] != 1:
            raise ValueError("HQMaskDecoder currently expects per-image features with batch size 1.")

        prompt_count = int(sparse_prompt_embeddings.shape[0])
        if prompt_count == 0:
            output_size = tuple(int(value * 4) for value in final_features.shape[-2:])
            empty_logits = final_features.new_zeros((0, 1, *output_size))
            empty_scores = final_features.new_zeros((0,))
            empty_features = final_features.new_zeros((0, self.refine_feature_channels, *output_size))
            return {
                "coarse_mask_logits": empty_logits,
                "refined_mask_logits": empty_logits,
                "refine_features": empty_features,
                "decoder_scores": empty_scores,
            }

        final_embeddings = self.final_feature_proj(final_features)
        early_embeddings = self.early_feature_proj(early_features)
        if early_embeddings.shape[-2:] != final_embeddings.shape[-2:]:
            early_embeddings = F.interpolate(
                early_embeddings,
                size=final_embeddings.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        fused_embeddings = final_embeddings + self.early_fusion_scale * early_embeddings

        coarse_masks, decoder_scores = self._predict_masks(
            image_embeddings=fused_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )
        coarse_masks = coarse_masks[:, :1]
        decoder_scores = decoder_scores[:, 0].sigmoid()

        prompt_summary = (
            sparse_prompt_embeddings.mean(dim=1)
            if sparse_prompt_embeddings.shape[1] > 0
            else sparse_prompt_embeddings.new_zeros((prompt_count, self.transformer_dim))
        )
        hq_context = prompt_summary + self.hq_token.weight.expand(prompt_count, -1)
        gating = self.hq_gate(hq_context)
        upscaled_early = self.refine_upscale(early_embeddings).repeat(prompt_count, 1, 1, 1)
        gated_early = upscaled_early * gating.unsqueeze(-1).unsqueeze(-1)[:, : upscaled_early.shape[1]]

        hq_weights = self.hq_hypernet(hq_context)
        residual_masks = torch.einsum("bc,bchw->bhw", hq_weights, gated_early).unsqueeze(1)
        dense_residual = F.interpolate(
            dense_prompt_logits.float(),
            size=coarse_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        refined_mask_logits = coarse_masks + residual_masks + self.dense_prompt_scale * dense_residual
        refine_features = torch.cat((coarse_masks, gated_early, dense_residual), dim=1)

        return {
            "coarse_mask_logits": coarse_masks,
            "refined_mask_logits": refined_mask_logits,
            "refine_features": refine_features,
            "decoder_scores": decoder_scores,
        }
