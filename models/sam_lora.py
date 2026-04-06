"""LoRA utilities for SAM image encoder attention blocks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

import torch
import torch.nn as nn

__all__ = [
    "LoRAInjectionSummary",
    "QKVLoRALinear",
    "collect_lora_state_dict",
    "get_lora_trainable_stats",
    "inject_sam_lora",
    "iter_lora_modules",
    "load_lora_state_dict",
    "load_lora_weights",
    "save_lora_weights",
]


@dataclass(frozen=True)
class LoRAInjectionSummary:
    """Metadata returned after LoRA injection."""

    target_blocks: List[int]
    target_module_names: List[str]
    rank: int
    alpha: int
    dropout: float


class QKVLoRALinear(nn.Module):
    """Wrap a SAM qkv projection and add LoRA updates only to q and v."""

    def __init__(
        self,
        base_linear: nn.Linear,
        *,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if base_linear.out_features % 3 != 0:
            raise ValueError("SAM qkv projection must have 3 * embed_dim outputs")

        self.base_linear = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)

        embed_dim = base_linear.out_features // 3
        self.embed_dim = embed_dim
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        self.lora_q_a = nn.Linear(self.in_features, rank, bias=False)
        self.lora_q_b = nn.Linear(rank, embed_dim, bias=False)
        self.lora_v_a = nn.Linear(self.in_features, rank, bias=False)
        self.lora_v_b = nn.Linear(rank, embed_dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_q_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_q_b.weight)
        nn.init.kaiming_uniform_(self.lora_v_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_v_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        dropped = self.dropout(x)

        q_delta = self.lora_q_b(self.lora_q_a(dropped)) * self.scaling
        v_delta = self.lora_v_b(self.lora_v_a(dropped)) * self.scaling

        delta = torch.zeros_like(base_out)
        delta[..., : self.embed_dim] = q_delta
        delta[..., 2 * self.embed_dim :] = v_delta
        return base_out + delta


def iter_lora_modules(module: nn.Module) -> Iterator[tuple[str, QKVLoRALinear]]:
    """Yield all injected SAM LoRA modules."""

    for module_name, child in module.named_modules():
        if isinstance(child, QKVLoRALinear):
            yield module_name, child


def _freeze_module_parameters(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def inject_sam_lora(
    image_encoder: nn.Module,
    *,
    num_last_blocks: int = 6,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
) -> LoRAInjectionSummary:
    """Inject LoRA wrappers into the last N SAM attention qkv projections."""

    blocks = getattr(image_encoder, "blocks", None)
    if blocks is None:
        raise ValueError("image_encoder must expose a 'blocks' attribute")

    total_blocks = len(blocks)
    if num_last_blocks <= 0 or num_last_blocks > total_blocks:
        raise ValueError("num_last_blocks must be within the encoder depth")

    target_blocks = list(range(total_blocks - num_last_blocks, total_blocks))
    _freeze_module_parameters(image_encoder)

    target_module_names: List[str] = []
    for block_index in target_blocks:
        attention = blocks[block_index].attn
        qkv = attention.qkv
        if isinstance(qkv, QKVLoRALinear):
            target_module_names.append(f"blocks.{block_index}.attn.qkv")
            continue
        if not isinstance(qkv, nn.Linear):
            raise TypeError(f"Unsupported qkv module type: {type(qkv)!r}")
        attention.qkv = QKVLoRALinear(qkv, rank=rank, alpha=alpha, dropout=dropout)
        target_module_names.append(f"blocks.{block_index}.attn.qkv")

    return LoRAInjectionSummary(
        target_blocks=target_blocks,
        target_module_names=target_module_names,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )


def collect_lora_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    """Collect only LoRA weights for checkpoint export."""

    lora_state: Dict[str, torch.Tensor] = {}
    for module_name, lora_module in iter_lora_modules(module):
        prefix = f"{module_name}."
        lora_state[f"{prefix}lora_q_a.weight"] = lora_module.lora_q_a.weight.detach().cpu()
        lora_state[f"{prefix}lora_q_b.weight"] = lora_module.lora_q_b.weight.detach().cpu()
        lora_state[f"{prefix}lora_v_a.weight"] = lora_module.lora_v_a.weight.detach().cpu()
        lora_state[f"{prefix}lora_v_b.weight"] = lora_module.lora_v_b.weight.detach().cpu()
    return lora_state


def load_lora_state_dict(
    module: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    *,
    strict: bool = True,
) -> None:
    """Load a LoRA-only state dict into injected modules."""

    current_state = module.state_dict()
    missing_keys: List[str] = []
    unexpected_keys = [key for key in state_dict if key not in current_state]

    for key, tensor in state_dict.items():
        if key in current_state:
            current_state[key].copy_(tensor)

    for module_name, _ in iter_lora_modules(module):
        for suffix in (
            "lora_q_a.weight",
            "lora_q_b.weight",
            "lora_v_a.weight",
            "lora_v_b.weight",
        ):
            key = f"{module_name}.{suffix}"
            if key not in state_dict:
                missing_keys.append(key)

    if strict and (missing_keys or unexpected_keys):
        raise KeyError(
            f"LoRA state dict mismatch. Missing keys: {missing_keys}; unexpected keys: {unexpected_keys}"
        )


def save_lora_weights(module: nn.Module, path: str | Path) -> None:
    """Persist LoRA weights only."""

    torch.save(collect_lora_state_dict(module), Path(path))


def load_lora_weights(
    module: nn.Module,
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> None:
    """Load LoRA weights from disk."""

    state_dict = torch.load(Path(path), map_location=map_location)
    load_lora_state_dict(module, state_dict, strict=strict)


def get_lora_trainable_stats(module: nn.Module) -> Dict[str, object]:
    """Summarize injected LoRA modules and trainable parameter counts."""

    module_names: List[str] = []
    trainable_params = 0
    total_params = 0
    for module_name, lora_module in iter_lora_modules(module):
        module_names.append(module_name)
        for parameter in lora_module.parameters():
            total_params += parameter.numel()
            if parameter.requires_grad:
                trainable_params += parameter.numel()

    return {
        "module_names": module_names,
        "num_modules": len(module_names),
        "trainable_params": trainable_params,
        "total_params": total_params,
    }
