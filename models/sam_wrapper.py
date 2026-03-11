"""
SAM wrapper with CRNet adapter integration.

This module wraps the Segment Anything Model (SAM) with a CRNet feature
enhancement adapter that processes the encoder output before passing to
the mask decoder.
"""

import torch
import torch.nn as nn

from .crnet_adapter import crnet_adapter

__all__ = ['SAMWithCRNetAdapter']


class SAMWithCRNetAdapter(nn.Module):
    """
    SAM with CRNet feature enhancement adapter.

    This wrapper intercepts SAM's ViT encoder output, processes it through
    a CRNet adapter for feature enhancement, and then passes the enhanced
    features to SAM's mask decoder.

    Args:
        sam_model: Pre-loaded SAM model (ViT-H, ViT-L, or ViT-B)
        adapter_config: Dict with adapter configuration
            - adapter_size: 'small', 'medium', or 'large'
            - compression_ratio: 4, 8, 16, 32, or 64
            - use_residual: bool, default True
            - simple: bool, use simplified adapter (no FC compression)
        freeze_encoder: Whether to freeze SAM encoder weights (default: True)
        freeze_decoder: Whether to freeze SAM decoder weights (default: False)
        freeze_prompt_encoder: Whether to freeze SAM prompt encoder (default: True)

    Example:
        >>> from segment_anything import sam_model_registry
        >>> sam = sam_model_registry['vit_h'](checkpoint='sam_vit_h_4b8939.pth')
        >>> model = SAMWithCRNetAdapter(
        ...     sam_model=sam,
        ...     adapter_config={'adapter_size': 'medium', 'compression_ratio': 8}
        ... )
    """

    def __init__(self, sam_model, adapter_config,
                 freeze_encoder=True, freeze_decoder=False, freeze_prompt_encoder=True):
        super(SAMWithCRNetAdapter, self).__init__()

        self.sam_model_type = getattr(sam_model, '__class__', None)

        # Extract SAM components
        self.image_encoder = sam_model.image_encoder
        self.mask_decoder = sam_model.mask_decoder
        self.prompt_encoder = sam_model.prompt_encoder

        # Get encoder output channels
        self.encoder_out_channels = self._get_encoder_output_channels()

        # Create CRNet adapter
        self.adapter = crnet_adapter(
            in_channels=self.encoder_out_channels,
            adapter_size=adapter_config.get('adapter_size', 'medium'),
            compression_ratio=adapter_config.get('compression_ratio', 4),
            use_residual=adapter_config.get('use_residual', True),
            simple=adapter_config.get('simple', False)
        )

        # Freeze components if specified
        if freeze_encoder:
            self._freeze_module(self.image_encoder)

        if freeze_decoder:
            self._freeze_module(self.mask_decoder)

        if freeze_prompt_encoder:
            self._freeze_module(self.prompt_encoder)

        # Store configuration
        self.adapter_config = adapter_config
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder
        self.freeze_prompt_encoder = freeze_prompt_encoder

    def _get_encoder_output_channels(self):
        """
        Get the number of output channels from SAM's image encoder.

        Returns:
            int: Number of output channels (256 for all SAM variants)
        """
        # All SAM variants (ViT-H, ViT-L, ViT-B) output 256 channels
        return 256

    def _freeze_module(self, module):
        """Freeze all parameters in a module."""
        for param in module.parameters():
            param.requires_grad = False

    def _unfreeze_module(self, module):
        """Unfreeze all parameters in a module."""
        for param in module.parameters():
            param.requires_grad = True

    def forward(self, batched_input, multimask_output=True):
        """
        Forward pass through SAM with CRNet adapter.

        This follows the same interface as the original SAM model,
        processing images through the encoder, enhancing features with
        the adapter, then generating masks with the decoder.

        Args:
            batched_input: List of dicts, each containing:
                - 'image': (3, H, W) tensor, the input image
                - 'original_size': (H, W) tuple, original image size
                - Optional prompts: 'point_coords', 'point_labels', 'boxes', 'mask_inputs'
            multimask_output: Whether to output multiple mask predictions

        Returns:
            List of dicts, each containing:
                - 'masks': (K, H, W) tensor, predicted masks
                - 'iou_predictions': (K,) tensor, IoU predictions
                - 'low_res_logits': (K, 256, 256) tensor, low-res masks
        """
        # Get input information
        input_images = torch.stack([self.preprocess(x['image']) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        # Apply CRNet adapter for feature enhancement
        enhanced_embeddings = self.adapter(image_embeddings)

        # Process each sample in the batch
        outputs = []
        for batch_record, embedding in zip(batched_input, enhanced_embeddings):
            # Handle prompts
            if 'point_coords' in batch_record:
                points = (batch_record['point_coords'].to(image_embeddings.device),
                          batch_record['point_labels'].to(image_embeddings.device))
            else:
                points = None

            if 'boxes' in batch_record:
                boxes = batch_record['boxes'].to(image_embeddings.device)
            else:
                boxes = None

            if 'mask_inputs' in batch_record:
                mask_inputs = batch_record['mask_inputs'].to(image_embeddings.device)
            else:
                mask_inputs = None

            # Encode prompts
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=boxes,
                masks=mask_inputs,
            )

            # Decode masks with enhanced embeddings
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),  # Add batch dim
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )

            # Post-process masks
            masks = self.postprocess_masks(
                low_res_masks,
                original_size=batch_record['original_size']
            )

            outputs.append({
                'masks': masks,
                'iou_predictions': iou_predictions,
                'low_res_logits': low_res_masks,
            })

        return outputs

    def preprocess(self, x):
        """
        Preprocess input image.

        Args:
            x: Input image tensor

        Returns:
            Preprocessed image tensor
        """
        # SAM preprocessing logic would go here
        # For now, return as-is (SAM handles preprocessing internally)
        return x

    def postprocess_masks(self, masks, original_size):
        """
        Post-process masks to original image size.

        Args:
            masks: Low-resolution mask predictions
            original_size: Original image size (H, W)

        Returns:
            Resized masks
        """
        # Simple resize to original size
        return torch.nn.functional.interpolate(
            masks,
            size=original_size,
            mode='bilinear',
            align_corners=False
        )

    def get_adapter_params(self):
        """Get parameters of the CRNet adapter only."""
        return list(self.adapter.parameters())

    def get_trainable_params(self):
        """Get all trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]

    def get_num_adapter_params(self):
        """Get the number of parameters in the adapter."""
        return sum(p.numel() for p in self.adapter.parameters())

    def get_num_total_params(self):
        """Get the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def get_num_frozen_params(self):
        """Get the number of frozen parameters."""
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)

    def print_model_info(self):
        """Print model information including adapter statistics."""
        total_params = self.get_num_total_params()
        adapter_params = self.get_num_adapter_params()
        frozen_params = self.get_num_frozen_params()

        print(f"\n{'='*60}")
        print(f"SAM with CRNet Adapter Model Info")
        print(f"{'='*60}")
        print(f"Adapter size: {self.adapter_config.get('adapter_size', 'medium')}")
        print(f"Compression ratio: 1/{self.adapter_config.get('compression_ratio', 4)}")
        print(f"Use residual: {self.adapter_config.get('use_residual', True)}")
        print(f"\nParameter counts:")
        print(f"  Total params: {total_params:,}")
        print(f"  Adapter params: {adapter_params:,} ({100*adapter_params/total_params:.2f}%)")
        print(f"  Frozen params: {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
        print(f"  Trainable params: {total_params - frozen_params:,}")
        print(f"{'='*60}\n")


def create_sam_with_adapter(sam_model, adapter_size='medium', compression_ratio=4,
                            use_residual=True, simple=False,
                            freeze_encoder=True, freeze_decoder=False):
    """
    Factory function to create SAM with CRNet adapter.

    Args:
        sam_model: Pre-loaded SAM model
        adapter_size: 'small', 'medium', or 'large'
        compression_ratio: 4, 8, 16, 32, or 64
        use_residual: Whether to use residual connection
        simple: Use simplified adapter
        freeze_encoder: Freeze SAM encoder
        freeze_decoder: Freeze SAM decoder

    Returns:
        SAMWithCRNetAdapter instance
    """
    adapter_config = {
        'adapter_size': adapter_size,
        'compression_ratio': compression_ratio,
        'use_residual': use_residual,
        'simple': simple,
    }

    return SAMWithCRNetAdapter(
        sam_model=sam_model,
        adapter_config=adapter_config,
        freeze_encoder=freeze_encoder,
        freeze_decoder=freeze_decoder,
    )
