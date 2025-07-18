import torch
import torch.nn as nn
from monai.networks.layers import Norm
from monai.networks.blocks.convolutions import Convolution
from torchga.torchga import GeometricAlgebra
from torchga.layers import (
    GeometricProductDense,
    TensorToGeometric,
    GeometricToTensor,
    GeometricProductConv1D,
    GeometricProductElementwise,
)
from typing import List
from collections.abc import Callable
from enum import Enum, auto

class FusionMode(Enum):
    Concatenation = auto()
    Clifford = auto()


class RomanuccioEmbeddingBlock(nn.Module):
    def __init__(
        self,
        embedding_length: int,
        used_channels: List[int],
        fusion_mode: FusionMode = FusionMode.Clifford,
        algebra_metric: List[int] = None,
        embedding_indices: List[int] = None,
        weight_blade_indices: torch.Tensor = None,
        real_reduction: Callable = None,
        depth=1,
        giovanna_activation="RELU",
        norm=Norm.BATCH,
        *args,
        **kwargs
    ):
        """
        Block that works on embeddings either the Giovanna or Clifford way.

        keyword arguments:
        algebra_metric: diagonal of bilinear form, assuming diagonal bilinear form matrix
        embedding_indices: list of indices (0 to 2^len(algebra_metric)) for embedding blades e.g. 0 => scalar, 1 => e1, ..., N => eN, N+1 => e12, ...
        weight_blade_indices: list of indices as for embedding_indices
        embedding_length: length of 1 embedding in unet
        """
        super().__init__(*args, **kwargs)
        self.fusion_mode = fusion_mode
        if self.fusion_mode == FusionMode.Concatenation:
            print("Defining Giovanna model")
            self.to_ga_embeddings = None
            self.embedding_processing_block = Convolution(
                spatial_dims=1,
                in_channels=1,
                out_channels=1,
                strides=len(used_channels) if isinstance(used_channels, list) else 1,
                kernel_size=3,
                norm=norm,
                bias=False,
                act=giovanna_activation,
            )
        elif self.fusion_mode == FusionMode.Clifford:
            print("Defining Clifford model")
            self.ga = GeometricAlgebra(algebra_metric)
            self.to_ga_embeddings = nn.ModuleList()
            self.real_reduction = real_reduction

            # prepare embedding sublayers
            for index in embedding_indices:
                to_ith_blade = TensorToGeometric(self.ga, torch.tensor(index))
                self.to_ga_embeddings.append(to_ith_blade)

            self.embedding_processing_block = nn.Sequential()
            for _ in range(depth):
                self.embedding_processing_block.append(
                    GeometricProductElementwise(
                        self.ga,
                        embedding_length,
                        embedding_length,
                        weight_blade_indices,
                        activation="tanh",
                        use_bias=False,
                    )
                )
        else:
            raise Exception("Incorrect fusion mode.")

    def forward(self, embeddings: list):
        # embeddings is a list of individual embedding batchxN, 
        # where the length of the list is equal to number of modalities

        if self.fusion_mode == FusionMode.Concatenation:
            # Giovanna way
            emb = embeddings[0]

            if len(embeddings) > 1:
                for embedding in embeddings[1:]:
                    emb = torch.cat(
                        (emb, embedding), dim=1
                    )  # dim because of batched input BCWHD

            # do stuff with embedding
            emb = self.embedding_processing_block(emb[:, None, :])

            return emb
        elif self.fusion_mode == FusionMode.Clifford:
            # Clifford way

            # embed tensors in ga
            ga_embedding_vectors = [
                embedding_ith(emb[:, :, None])
                for embedding_ith, emb in zip(self.to_ga_embeddings, embeddings)
            ]

            # sum ga tensors: now the embedding has a multivector per channel, 1xN
            ga_embedding = torch.stack(ga_embedding_vectors, dim=0).sum(dim=0)

            # do operations on ga tensors
            ga_embedding = self.embedding_processing_block(ga_embedding)

            # go back to real numbers
            real_embedding = self.real_reduction(ga_embedding)

            # return result
            return real_embedding