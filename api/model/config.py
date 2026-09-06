# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/11 2:03
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/11 2:03

"""
Module defining the PredictionModel factory for instantiating various time-series prediction architectures.

This module provides a unified interface for creating and managing different time-series
prediction models. It supports a wide range of architectures including traditional
statistical models, neural networks, and modern transformer-based approaches.

Supported Models:
    - Traditional: LSTM, AR, ARIMA, LO (Last Observation)
    - Neural Networks: FCNN (Fully Connected Neural Network)
    - Modern Architectures: SegRNN, FreTS, ModernTCN, MultiPatchFormer, ConvTimeNet

Each model is initialized with appropriate parameters including sequence length,
feature dimensions, number of nodes, and prediction horizon.
"""

# Standard library imports
from typing import Any

# Third-party imports
import torch

# Local imports
from api.model.modules import Lstm, Lo, Ar, Arima, Fcnn, SegRNN, FreTS, ModernTCN, MultiPatchFormer, ConvTimeNet, SatCrossAttention, SatFreTS
from api.model.NbSatFusion import NbSatFusion
from api.model.specfrets import SpecFreTS
from api.model.latentfusion import LatentFusion
from api.model.gwnet import GWNet
from api.model.astgcn import ASTGCN
from api.model.ev_vlm import EV_VLM
from api.model.patchtst import PatchTST


class PredictionModel(object):
    """
    Factory and loader for prediction models.

    This class provides a unified interface for selecting and initializing various
    time-series prediction model architectures. It acts as a factory that creates
    the appropriate model based on the specified name and parameters.

    The class supports both traditional statistical models and modern deep learning
    architectures, providing a consistent interface for model creation, loading,
    and configuration management.

    Attributes:
        model_name (str): Identifier of the model architecture.
        model (torch.nn.Module): Instantiated model ready for training or inference.
    """

    def __init__(
            self,
            num_node: int,
            n_fea: int,
            model_name: str,
            seq_l: int,
            pre_len: int,
            sat_dim: int | None = None,
            vit_image_size: int = 224,
            vit_patch_size: int = 16,
            vit_embed_dim: int = 128,
            vit_depth: int = 4,
            vit_heads: int = 4,
            fusion_dim: int = 256,
            d_model: int = 128,
            gpt2_model_path: str = None,
            siglip_model_path: str = None,
            ablate_proto: bool = False,
            ablate_dyn: bool = False,
            spatial_early: bool = False,
            no_gat: bool = False,
            plain_gat: bool = False,
            gate_bias: float = 0.0,
            qformer_queries: int = 32,
            lora_rank: int = 8,
            lora_alpha: int = 16,
            patch_len: int = 6,
            patch_stride: int = 3,
            llm_chunk_size: int = 64,
            ablate_qformer: bool = False,
            ablate_text: bool = False,
    ) -> None:
        """
        Initialize a PredictionModel instance by selecting the appropriate architecture.

        This method creates the specified model architecture with the given parameters.
        It supports multiple model types ranging from simple statistical models to
        complex deep learning architectures including EV-VLM with custom model paths.

        Args:
            num_node (int): Number of nodes or output dimensions (for graph-based models).
            n_fea (int): Number of input feature channels.
            model_name (str): Name of the model to instantiate ('lstm', 'lo', 'ar', etc.).
            seq_l (int): Input sequence length for time-series prediction.
            pre_len (int): Prediction horizon length (number of future time steps).
            gpt2_model_path (str): Optional path to GPT-2 model weights for EV-VLM.
            siglip_model_path (str): Optional path to SigLIP model weights for EV-VLM.

        Raises:
            ValueError: If the specified model_name is unsupported.
        """
        self.model_name = model_name  # Store model name for reference

        # Initialize model based on specified architecture
        if model_name == 'patchtst':
            self.model = PatchTST(
                seq_len=seq_l,
                pred_len=pre_len,
                num_vars=num_node,
                patch_len=6,
                stride=3,
                d_model=d_model,
                nhead=8,
                n_layers=3,
                d_ff=4 * d_model,
                dropout=0.2,
            )
        elif model_name == 'ev_vlm':
            # EV-VLM: GPT-2 unified backbone + Q-former satellite compression + LoRA
            self.model = EV_VLM(
                seq_len=seq_l,
                pred_len=pre_len,
                num_nodes=num_node,
                gpt2_model_path=gpt2_model_path,
                patch_len=patch_len,
                patch_stride=patch_stride,
                qformer_queries=qformer_queries,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                llm_chunk_size=llm_chunk_size,
                ablate_qformer=ablate_qformer,
                ablate_dyn=ablate_dyn,
                ablate_text=ablate_text,
            )
        elif model_name == 'lstm':
            # Long Short-Term Memory network for sequence modeling
            self.model = Lstm(seq=seq_l, n_fea=n_fea, node=num_node)
        elif model_name == 'lo':
            # Last Observation: simple baseline using last value
            self.model = Lo(pred_len=pre_len)
        elif model_name == 'ar':
            # Autoregressive model with specified lag order
            self.model = Ar(pred_len=pre_len, lags=seq_l)
        elif model_name == 'arima':
            # ARIMA model for time series forecasting
            self.model = Arima(pred_len=pre_len)
        elif model_name == 'fcnn':
            # Fully Connected Neural Network
            self.model = Fcnn(n_fea, node=num_node, seq=seq_l)
        elif model_name == 'segrnn':
            # Segmented Recurrent Neural Network
            self.model = SegRNN(seq_len=seq_l, pred_len=pre_len, n_fea=n_fea)
        elif model_name == 'frets':
            # Frequency-based Time Series model
            self.model = FreTS(seq_len=seq_l, pred_len=pre_len, n_fea=n_fea)
        elif model_name == 'moderntcn':
            # Modern Temporal Convolutional Network
            self.model = ModernTCN(seq_len=seq_l, n_fea=n_fea, pred_len=pre_len)
        elif model_name == 'multipatchformer':
            # Multi-Patch Transformer for time series
            self.model = MultiPatchFormer(seq_len=seq_l, n_fea=n_fea, pred_len=pre_len)
        elif model_name == 'convtimenet':
            # Convolutional Time Network
            self.model = ConvTimeNet(seq_len=seq_l, c_in=n_fea, c_out=pre_len)
        elif model_name == 'satcrossattn':
            # Cross-attention between temporal token and satellite embedding token
            self.model = SatCrossAttention(seq_len=seq_l, sat_dim=n_fea - 1)
        elif model_name == 'satfrets':
            # FreTS backbone with satellite-aware cross-attention fusion head
            self.model = SatFreTS(seq_len=seq_l, pred_len=pre_len, sat_dim=n_fea - 1)
        elif model_name == 'specfrets':
            # Spectral-aligned fusion in frequency domain (temporal + satellite)
            self.model = SpecFreTS(seq_len=seq_l, pred_len=pre_len, sat_dim=n_fea - 1)
        elif model_name == 'latentfusion':
            # Latent space projection and cross-modal attention fusion
            self.model = LatentFusion(seq_len=seq_l, pred_len=pre_len, sat_dim=n_fea - 1)
        elif model_name == 'NbSatFusion':
            self.model = NbSatFusion(
                seq_len=seq_l,
                pred_len=pre_len,
                n_fea=n_fea,
                image_size=vit_image_size,
                patch_size=vit_patch_size,
                image_embed_dim=vit_embed_dim,
                vit_depth=vit_depth,
                vit_heads=vit_heads,
                fusion_dim=fusion_dim,
            )
        elif model_name == 'gwnet':
            # Graph WaveNet: Spatial-temporal graph neural network
            self.model = GWNet(
                num_nodes=num_node,
                seq_len=seq_l,
                pred_len=pre_len,
                c_in=1,
                c_out=1,
                nhid=32,
                layers=2,
                kernel_size=2,
                dropout=0.3,
                gdep=2,
                supports_len=2
            )
        elif model_name == 'astgcn':
            # ASTGCN: Attention-based Spatial-Temporal Graph Convolutional Network
            # Following original paper: "Attention Based Spatial-Temporal Graph 
            # Convolutional Networks for Traffic Flow Forecasting" (Guo et al., 2019)
            if seq_l != 48:
                raise ValueError("ASTGCN in this benchmark expects seq_l=48 for the two-branch sampler")
            self.model = ASTGCN(
                num_nodes=num_node,
                seq_len=seq_l,
                pred_len=pre_len,
                c_in=1,
                c_out=1,
                nhid=64,           # Original paper uses 64
                layers=3,          # Original paper uses 3 ST-Res blocks
                num_heads=8,       # Original paper uses 8 attention heads
                dropout=0.0        # Original paper doesn't use dropout
            )
        else:
            # Raise error for unsupported model names
            raise ValueError(f"Unsupported model_name: {model_name}")

        # Set default chunk size for processing large sequences in batches
        # This helps manage memory usage for long sequences
        self.model.chunk_size = 512

    def load_model(self, model_path: str) -> None:
        """
        Load pretrained model weights from a checkpoint file.

        This method loads previously saved model parameters from a file and
        applies them to the current model instance. It uses torch's load_state_dict
        for parameter restoration.

        Args:
            model_path (str): Filesystem path to the saved state dictionary file.
        """
        state_dict = torch.load(model_path, weights_only=True)  # Load state dictionary
        # strict=False: train.pth holds trainable params only (frozen GPT-2 base is
        # intentionally excluded), so missing keys are expected and skipped.
        self.model.load_state_dict(state_dict, strict=False)  # Apply loaded parameters to model

    def update_chunksize(self, chunk_size: int) -> None:
        """
        Update the model's internal chunk size parameter.

        This method allows dynamic adjustment of the chunk size used for processing
        long sequences. A larger chunk size may improve throughput but increase
        memory usage, while a smaller chunk size reduces memory usage but may
        decrease processing speed.

        Args:
            chunk_size (int): New chunk size for sequence processing.
        """
        self.model.chunk_size = chunk_size  # Update chunk size parameter
