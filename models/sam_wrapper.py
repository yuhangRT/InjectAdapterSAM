"""
WireCR-SAM wrapper with class-aware prompts for automatic semantic segmentation.
"""

from typing import Any, Dict, List

import torch
import torch.nn as nn

from .crnet_adapter import crnet_adapter

__all__ = ["SAMWithCRNetAdapter", "create_sam_with_adapter"]


class SAMWithCRNetAdapter(nn.Module):
    """
    SAM wrapper for wire / interface-hole semantic segmentation.

    The model keeps SAM's image encoder and mask decoder, but replaces manual
    interaction prompts with class-aware learnable prompts generated from
    enhanced image embeddings.
    """

    def __init__(
        self,
        sam_model,
        adapter_config,
        num_classes=3,
        class_names=None,
        disable_adapter=False,
        class_aware_prompts=True,
        freeze_encoder=True,
        freeze_decoder=False,
        freeze_prompt_encoder=True,
    ):
        super().__init__()

        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        self.sam_model = sam_model
        self.image_encoder = sam_model.image_encoder
        self.mask_decoder = sam_model.mask_decoder
        self.prompt_encoder = sam_model.prompt_encoder

        self.num_classes = num_classes
        self.class_names = list(class_names or self._default_class_names(num_classes))
        self.foreground_classes = num_classes - 1
        self.class_aware_prompts = class_aware_prompts
        self.disable_adapter = disable_adapter
        self.encoder_out_channels = 256
        self.prompt_dim = self.prompt_encoder.embed_dim

        if self.disable_adapter:
            self.adapter = nn.Identity()
        else:
            self.adapter = crnet_adapter(
                in_channels=self.encoder_out_channels,
                adapter_size=adapter_config.get("adapter_size", "medium"),
                compression_ratio=adapter_config.get("compression_ratio", 8),
                use_residual=adapter_config.get("use_residual", True),
                simple=adapter_config.get("simple", False),
            )

        self.class_prompt_embeddings = nn.Embedding(self.foreground_classes, self.prompt_dim)
        self.prompt_generator = nn.Sequential(
            nn.Linear(self.encoder_out_channels, self.encoder_out_channels),
            nn.GELU(),
            nn.Linear(self.encoder_out_channels, self.foreground_classes * self.prompt_dim),
        )

        self.adapter_config = adapter_config
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder
        self.freeze_prompt_encoder = freeze_prompt_encoder

        if freeze_encoder:
            self._freeze_module(self.image_encoder)
        if freeze_decoder:
            self._freeze_module(self.mask_decoder)
        if freeze_prompt_encoder:
            self._freeze_module(self.prompt_encoder)

    @staticmethod
    def _default_class_names(num_classes):
        if num_classes == 2:
            return ["background", "foreground"]
        if num_classes == 3:
            return ["background", "wire", "interface-hole"]
        return ["background"] + [f"class_{idx}" for idx in range(1, num_classes)]

    def _freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def _build_class_prompts(self, enhanced_embedding):
        pooled = torch.mean(enhanced_embedding, dim=(2, 3))
        prompt_offsets = self.prompt_generator(pooled).view(
            enhanced_embedding.size(0), self.foreground_classes, self.prompt_dim
        )
        base_prompts = self.class_prompt_embeddings.weight.unsqueeze(0).expand_as(prompt_offsets)
        if self.class_aware_prompts:
            return base_prompts + prompt_offsets
        return base_prompts

    def _build_semantic_logits(self, class_logits):
        background_logit = -torch.amax(class_logits, dim=1, keepdim=True)
        return torch.cat([background_logit, class_logits], dim=1)

    def forward(self, batched_input: List[Dict[str, Any]], multimask_output=False):
        if not batched_input:
            raise ValueError("batched_input must not be empty")

        device = self.sam_model.device
        input_images = torch.stack(
            [self.sam_model.preprocess(record["image"].to(device).float()) for record in batched_input],
            dim=0,
        )
        image_embeddings = self.image_encoder(input_images)
        enhanced_embeddings = self.adapter(image_embeddings)
        class_prompts = self._build_class_prompts(enhanced_embeddings)
        image_pe = self.prompt_encoder.get_dense_pe()

        class_logits_list = []
        low_res_logits_list = []
        iou_predictions_list = []

        for record, embedding, prompt_tokens in zip(
            batched_input, enhanced_embeddings, class_prompts
        ):
            dense_prompt = self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                1, -1, embedding.shape[-2], embedding.shape[-1]
            )

            sample_low_res_logits = []
            sample_iou_predictions = []
            for class_prompt in prompt_tokens:
                sparse_prompt = class_prompt.view(1, 1, -1)
                low_res_masks, iou_predictions = self.mask_decoder(
                    image_embeddings=embedding.unsqueeze(0),
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sparse_prompt,
                    dense_prompt_embeddings=dense_prompt,
                    multimask_output=False,
                )
                sample_low_res_logits.append(low_res_masks[:, 0])
                sample_iou_predictions.append(iou_predictions[:, 0])

            low_res_logits = torch.cat(sample_low_res_logits, dim=0).unsqueeze(0)
            class_logits = self.sam_model.postprocess_masks(
                low_res_logits,
                input_size=record["image"].shape[-2:],
                original_size=record.get("output_size", record["image"].shape[-2:]),
            )

            class_logits_list.append(class_logits.squeeze(0))
            low_res_logits_list.append(low_res_logits.squeeze(0))
            iou_predictions_list.append(torch.cat(sample_iou_predictions, dim=0))

        class_logits = torch.stack(class_logits_list, dim=0)
        low_res_logits = torch.stack(low_res_logits_list, dim=0)
        iou_predictions = torch.stack(iou_predictions_list, dim=0)
        semantic_logits = self._build_semantic_logits(class_logits)

        return {
            "class_logits": class_logits,
            "low_res_class_logits": low_res_logits,
            "semantic_logits": semantic_logits,
            "semantic_masks": semantic_logits.argmax(dim=1),
            "iou_predictions": iou_predictions,
        }

    def get_adapter_params(self):
        return list(self.adapter.parameters()) + list(self.class_prompt_embeddings.parameters()) + list(
            self.prompt_generator.parameters()
        )

    def get_trainable_params(self):
        return [param for param in self.parameters() if param.requires_grad]

    def get_num_adapter_params(self):
        return sum(param.numel() for param in self.get_adapter_params())

    def get_num_total_params(self):
        return sum(param.numel() for param in self.parameters())

    def get_num_frozen_params(self):
        return sum(param.numel() for param in self.parameters() if not param.requires_grad)

    def print_model_info(self):
        total_params = self.get_num_total_params()
        adapter_params = self.get_num_adapter_params()
        frozen_params = self.get_num_frozen_params()

        print(f"\n{'=' * 60}")
        print("WireCR-SAM Model Info")
        print(f"{'=' * 60}")
        print(f"Adapter enabled: {not self.disable_adapter}")
        print(f"Adapter size: {self.adapter_config.get('adapter_size', 'medium')}")
        print(f"Compression ratio: 1/{self.adapter_config.get('compression_ratio', 8)}")
        print(f"Use residual: {self.adapter_config.get('use_residual', True)}")
        print(f"Class-aware prompts: {self.class_aware_prompts}")
        print(f"Semantic classes: {self.num_classes}")
        print("\nParameter counts:")
        print(f"  Total params: {total_params:,}")
        print(f"  Adapter params: {adapter_params:,} ({100 * adapter_params / total_params:.2f}%)")
        print(f"  Frozen params: {frozen_params:,} ({100 * frozen_params / total_params:.2f}%)")
        print(f"  Trainable params: {total_params - frozen_params:,}")
        print(f"{'=' * 60}\n")


def create_sam_with_adapter(
    sam_model,
    adapter_size="medium",
    compression_ratio=8,
    use_residual=True,
    simple=False,
    num_classes=3,
    disable_adapter=False,
    class_aware_prompts=True,
    freeze_encoder=True,
    freeze_decoder=False,
):
    """Factory function for WireCR-SAM."""
    adapter_config = {
        "adapter_size": adapter_size,
        "compression_ratio": compression_ratio,
        "use_residual": use_residual,
        "simple": simple,
    }
    return SAMWithCRNetAdapter(
        sam_model=sam_model,
        adapter_config=adapter_config,
        num_classes=num_classes,
        disable_adapter=disable_adapter,
        class_aware_prompts=class_aware_prompts,
        freeze_encoder=freeze_encoder,
        freeze_decoder=freeze_decoder,
    )
