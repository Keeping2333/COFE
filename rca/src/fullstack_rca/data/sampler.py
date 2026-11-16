"""
Stratified batch sampler for contrastive learning.

Contrastive loss needs TWO things in every batch:
  1. **Positive pairs**: ≥2 samples of the SAME fault_type (anchors)
  2. **Hard negatives**: samples of a DIFFERENT fault_type from the SAME layer

Strategy for batch_size=8:
  - Pick 2 layers at random
  - For each layer, pick 2 fault types
  - For each fault type, pick 2 samples
  → 2 layers × 2 types × 2 samples = 8
  → Guarantees: 4 positive pairs + 4 intra-layer hard negatives per batch
"""
import random
from collections import defaultdict
from typing import Iterator, List

from torch.utils.data import Sampler


class StratifiedFaultTypeSampler(Sampler[List[int]]):
    """
    Yields batches that guarantee both positive pairs and intra-layer
    hard negatives for FaultTypeContrastiveLoss.

    Args:
        fault_type_ids: fault_type_id per sample (0-11, -1 for unknown)
        fault_layers:   fault_layer per sample (0-3, -1 for normal)
        batch_size:     target batch size (default 8)
        drop_last:      drop the final incomplete batch
    """

    def __init__(
        self,
        fault_type_ids: List[int],
        fault_layers: List[int],
        batch_size: int = 8,
        drop_last: bool = False,
    ):
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Group sample indices by (layer, fault_type)
        self.layer_ft_indices: dict = defaultdict(lambda: defaultdict(list))

        for idx, (ft_id, layer) in enumerate(zip(fault_type_ids, fault_layers)):
            if layer < 0:
                continue
            self.layer_ft_indices[layer][ft_id].append(idx)

        # Only keep layers that have ≥2 fault types (can form hard negatives)
        self.valid_layers = [
            l for l in sorted(self.layer_ft_indices.keys())
            if len(self.layer_ft_indices[l]) >= 2
        ]
        # Fallback: if no layer has ≥2 types, use all layers
        if not self.valid_layers:
            self.valid_layers = sorted(self.layer_ft_indices.keys())

        self.n_samples = sum(
            len(indices)
            for layer in self.layer_ft_indices
            for indices in self.layer_ft_indices[layer].values()
        )

        # Precompute: for each layer, list of fault types with ≥2 samples
        self.layer_ft_with_pairs = {}
        for layer in self.valid_layers:
            fts = [ft for ft, idxs in self.layer_ft_indices[layer].items() if len(idxs) >= 2]
            self.layer_ft_with_pairs[layer] = fts if len(fts) >= 2 else list(self.layer_ft_indices[layer].keys())

    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle all pools at epoch start
        pools: dict = defaultdict(dict)
        for layer in self.valid_layers:
            for ft_id, indices in self.layer_ft_indices[layer].items():
                pool = indices.copy()
                random.shuffle(pool)
                pools[layer][ft_id] = pool

        pointers: dict = defaultdict(lambda: defaultdict(int))

        def _draw(layer: int, ft_id: int) -> int:
            pool = pools[layer][ft_id]
            ptr = pointers[layer][ft_id]
            if ptr >= len(pool):
                random.shuffle(pool)
                pointers[layer][ft_id] = 0
                ptr = 0
            pointers[layer][ft_id] = ptr + 1
            return pool[ptr]

        # How many fault types per batch half = batch_size // (layers_per_batch * samples_per_type)
        # Default: batch_size=8 → 2 layers × 2 types × 2 samples = 8
        samples_per_type = 2  # minimum for positive pairs
        layers_per_batch = 2
        types_per_layer = max(1, self.batch_size // (layers_per_batch * samples_per_type))

        num_batches = self.n_samples // self.batch_size
        for _ in range(num_batches):
            batch = []

            # Pick layers_per_batch layers (with replacement if needed)
            selected_layers = random.choices(self.valid_layers, k=layers_per_batch)

            for layer in selected_layers:
                available_fts = self.layer_ft_with_pairs.get(layer, [])
                if len(available_fts) < types_per_layer:
                    available_fts = list(self.layer_ft_indices[layer].keys())

                # Pick types_per_layer different fault types from this layer
                chosen_fts = random.sample(
                    available_fts, min(types_per_layer, len(available_fts))
                )

                for ft in chosen_fts:
                    for _ in range(samples_per_type):
                        batch.append(_draw(layer, ft))

            if len(batch) >= self.batch_size:
                random.shuffle(batch)
                yield batch[:self.batch_size]
            elif not self.drop_last and len(batch) > 0:
                yield batch

    def __len__(self) -> int:
        n = self.n_samples // self.batch_size
        if not self.drop_last and (self.n_samples % self.batch_size != 0):
            n += 1
        return n
