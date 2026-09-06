# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/11 2:14
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/11 2:14

"""
Module implementing a variety of time-series forecasting algorithms and neural models.

This module contains implementations of both traditional statistical models and modern
deep learning architectures for time series forecasting, specifically designed for
electric vehicle charging demand prediction.

Classes:
    Lo: Last-observation baseline predictor - uses the most recent value as prediction.
    Ar: Autoregressive model predictor - traditional AR model with configurable lags.
    Arima: ARIMA model predictor with automated order selection using pmdarima.
    Fcnn: Fully-connected neural predictor - simple feedforward neural network.
    Lstm: LSTM-based sequence predictor - recurrent neural network for sequence modeling.
    SegRNN: Segmented RNN forecasting model - advanced RNN with segmentation strategy.
    FreTS: Frequency-domain temporal-spatial model - FFT-based neural architecture.
    ModernTCN: Temporal convolutional network with Reversible IN and re-parameterization.
    MultiPatchFormer: Multi-patch transformer encoder-based forecaster.
    ConvTimeNet: Convolutional time-series encoder with modern architectural features.

Each class provides a `predict` or `forward` method for inference.
"""

# Standard library imports
import math
import warnings

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from joblib import Parallel, delayed
from statsmodels.tsa.ar_model import AutoReg

try:
    from einops import rearrange
except ImportError:  # pragma: no cover - optional dependency for a subset of models
    def rearrange(*args, **kwargs):
        raise ImportError(
            "einops is required for models that use rearrange. Install it or choose a different model."
        )

try:
    from pmdarima import auto_arima
except ImportError:  # pragma: no cover - optional dependency for ARIMA baseline only
    auto_arima = None

# Local imports
from api.model.layers import (
    ModernTCN_RevIN,
    ModernTCN_Stage,
    ModernTCN_Flatten_Head,
    AttentionLayer,
    FullAttention,
    MultiPatchFormer_Encoder,
    ConvTimeNet_ConvEncoder,
    ConvTimeNet_get_activation_fn,
    CrossModalAttentionBlock,
)


class Lo:
    """
    Last-observation baseline predictor.
    
    This is a simple baseline model that uses the most recent observed value
    as the prediction for future time steps. It serves as a fundamental
    benchmark for evaluating more sophisticated forecasting models.
    
    Args:
        pre_len (int): Prediction horizon length.
    """
    
    def __init__(self, pre_len):
        """Initialize the last-observation predictor."""
        self.pred_len = pre_len

    def predict(self, train_valid_feat, test_feat):
        """
        Use the latest observed value as the prediction for the next time step.
        
        This method implements a simple strategy where predictions are based on
        the most recent available observations. For the first few predictions,
        it uses values from the training/validation set, then switches to using
        the test set values as they become available.
        
        Args:
            train_valid_feat (np.ndarray): Training and validation features [T, N].
            test_feat (np.ndarray): Test features [T, N].
            
        Returns:
            np.ndarray: Predictions of shape [T, N].
        """
        time_len, node = test_feat.shape  # Get dimensions
        preds = np.zeros((time_len, node))  # Initialize prediction array

        # Generate predictions for each node
        for j in range(node):
            for i in range(time_len):
                if i < self.pred_len:
                    # Use training/validation data for initial predictions
                    preds[i, j] = train_valid_feat[-self.pred_len + i, j]
                else:
                    # Use test data for subsequent predictions
                    preds[i, j] = test_feat[i - self.pred_len, j]

        return preds


class Ar:
    """
    Autoregressive model predictor.
    
    This class implements a traditional autoregressive model for time series
    forecasting. It uses past values of the time series to predict future values.
    
    Args:
        pred_len (int): Prediction horizon length.
        lags (int): Number of lagged observations to use (default: 1).
    """
    
    def __init__(self, pred_len, lags=1):
        """Initialize the autoregressive model."""
        self.pred_len = pred_len
        self.lags = lags

    def predict(self, train_valid_feat, test_feat):
        """
        Perform predictions using the AR model.
        
        This method fits an autoregressive model to the training data and
        generates predictions for the test period. It uses parallel processing
        to handle multiple nodes efficiently.
        
        Args:
            train_valid_feat (np.ndarray): Training and validation features [T, N].
            test_feat (np.ndarray): Test features [T, N].
            
        Returns:
            np.ndarray: Predictions of shape [T, N].
        """
        time_len, node = test_feat.shape  # Get dimensions
        train_valid_feat = train_valid_feat[:-self.pred_len, :]  # Remove prediction period from training

        def fit_predict(j):
            """Fit AR model and predict for a single node."""
            fit_series = train_valid_feat[:, j]  # Extract series for current node
            model = AutoReg(fit_series, lags=self.lags)  # Create AR model
            model_fitted = model.fit()  # Fit the model
            pred_node = []
            
            # Generate predictions for each time step
            for i in range(time_len):
                start = len(fit_series) + self.pred_len
                end = start
                pred = model_fitted.predict(start=start, end=end)  # Make prediction
                pred_node.append(pred[-1])  # Take the last predicted value
            return pred_node

        # Use parallel processing for efficiency
        preds = Parallel(n_jobs=10)(delayed(fit_predict)(j) for j in range(node))
        preds = np.array(preds).T  # Transpose to get [T, N] format
        return preds


class Arima:
    """
    ARIMA model predictor with automated order selection.
    
    This class implements an ARIMA (Autoregressive Integrated Moving Average)
    model with automatic parameter selection using the pmdarima library.
    
    Args:
        pred_len (int): Prediction horizon length.
    """
    
    def __init__(self, pred_len):
        """Initialize the ARIMA model parameters."""
        self.pred_len = pred_len

    def predict(self, train_valid_feat, test_feat):
        """
        Perform predictions using the ARIMA model with automatic parameter selection.

        This method automatically selects the optimal ARIMA parameters (p, d, q)
        and generates predictions for the test period. It uses parallel processing
        for efficiency and includes warning suppression for convergence issues.

        Args:
            train_valid_feat (np.ndarray): Training and validation features [T, N].
            test_feat (np.ndarray): Test features [T, N].

        Returns:
            np.ndarray: Predictions of shape [T, N].
        """
        time_len, node = test_feat.shape  # Get dimensions
        train_valid_feat = train_valid_feat[:-self.pred_len, :]  # Remove prediction period

        if auto_arima is None:
            raise ImportError(
                "pmdarima is required to use the Arima model. Install it or choose a different model."
            )

        # Suppress convergence warnings and constant series warnings
        warnings.filterwarnings("ignore", category=UserWarning,
                                message=".*Maximum Likelihood optimization failed to converge.*")
        warnings.filterwarnings("ignore", category=UserWarning,
                                message=".*Input time-series is completely constant.*")

        def fit_predict(j):
            """Fit ARIMA model and predict for a single node."""
            fit_series = train_valid_feat[:, j]  # Extract series for current node

            # Check if series is constant (all values are the same)
            if np.std(fit_series) < 1e-10:
                # For constant series, use the mean value as prediction
                pred_node = [np.mean(fit_series)] * time_len
            else:
                try:
                    # Automatically select ARIMA parameters
                    model = auto_arima(
                        fit_series,
                        start_p=0, max_p=2,  # AR order range
                        start_q=0, max_q=2,  # MA order range
                    )
                    pred_node = []

                    # Generate predictions for each time step
                    for i in range(time_len):
                        try:
                            pred = model.predict(n_periods=self.pred_len, return_conf_int=False)  # Predict multiple steps
                            # Check for NaN values and use fallback if found
                            if np.isnan(pred).any():
                                pred_node.append(np.mean(fit_series))
                            else:
                                pred_node.append(pred[-1])  # Take only the last value
                        except (ValueError, RuntimeError):
                            # If prediction fails, use the mean value as fallback
                            pred_node.append(np.mean(fit_series))
                except Exception:
                    # If ARIMA fitting fails, use mean as fallback for all predictions
                    pred_node = [np.mean(fit_series)] * time_len
            return pred_node

        # Use parallel processing for efficiency
        preds = Parallel(n_jobs=10)(delayed(fit_predict)(j) for j in range(node))
        preds = np.array(preds).T  # Transpose to get [T, N] format
        return preds


class Fcnn(nn.Module):
    """
    Fully-connected neural predictor.
    
    A simple feedforward neural network for time series forecasting.
    It flattens the input sequence and applies linear transformations.
    
    Args:
        n_fea (int): Number of input features.
        node (int): Number of nodes/sites.
        seq (int): Input sequence length (default: 12).
    """
    
    def __init__(self, n_fea, node, seq=12):  # input_dim = seq_length
        """Initialize the fully-connected neural network."""
        super(Fcnn, self).__init__()
        self.num_feat = n_fea
        self.seq = seq
        self.nodes = node
        self.linear = nn.Linear(seq * n_fea, 1)  # Single linear layer

    def forward(self, feat, extra_feat=None):
        """
        Forward pass through the fully-connected network.
        
        Args:
            feat (torch.Tensor): Input features of shape [batch, node, seq].
            extra_feat (torch.Tensor, optional): Auxiliary features.
            
        Returns:
            torch.Tensor: Predictions of shape [batch, node].
        """
        x = feat
        if extra_feat is not None:
            # Concatenate main features with auxiliary features
            x = torch.cat([feat.unsqueeze(-1), extra_feat], dim=-1)
            assert x.shape[-1] == self.num_feat, f"Number of features ({x.shape[-1]}) does not match n_fea ({self.num_feat})."
        
        # Flatten and apply linear transformation
        x = x.view(-1, self.nodes, self.seq * self.num_feat)
        x = self.linear(x)
        x = torch.squeeze(x)
        return x


class Lstm(nn.Module):
    """
    LSTM-based sequence predictor.
    
    A recurrent neural network using LSTM cells for time series forecasting.
    It processes the input sequence through LSTM layers and applies
    a final linear transformation for prediction.
    
    Args:
        seq (int): Input sequence length.
        n_fea (int): Number of input features.
        node (int): Number of nodes/sites.
    """
    
    def __init__(self, seq, n_fea, node):
        """Initialize the LSTM network."""
        super(Lstm, self).__init__()
        self.num_feat = n_fea
        self.nodes = node
        self.seq_len = seq
        self.encoder = nn.Linear(n_fea, 1)  # Input encoder
        self.lstm_hidden_dim = 16  # LSTM hidden dimension
        self.lstm = nn.LSTM(input_size=n_fea, hidden_size=self.lstm_hidden_dim, num_layers=2,
                            batch_first=True)  # LSTM layers
        self.linear = nn.Linear(seq * self.lstm_hidden_dim, 1)  # Output layer

    def forward(self, feat, extra_feat=None):  # feat.shape = [batch, node, seq]
        """
        Forward pass through the LSTM network.
        
        Args:
            feat (torch.Tensor): Input features of shape [batch, node, seq].
            extra_feat (torch.Tensor, optional): Auxiliary features.
            
        Returns:
            torch.Tensor: Predictions of shape [batch, node].
        """
        x = feat.unsqueeze(-1)  # Add channel dimension
        if extra_feat is not None:
            # Concatenate main features with auxiliary features
            x = torch.cat([feat.unsqueeze(-1), extra_feat], dim=-1)
        assert x.shape[-1] == self.num_feat, f"Number of features ({x.shape[-1]}) does not match n_fea ({self.num_feat})."

        bs = x.shape[0]  # Batch size
        # Reshape for LSTM processing: [batch*node, seq, features]
        x = x.view(bs * self.nodes, self.seq_len, self.num_feat)

        # Process through LSTM
        lstm_out, _ = self.lstm(x)  # lstm_out shape: [batch_size * node, seq_len, lstm_hidden_dim]
        lstm_out = lstm_out.reshape(bs, self.nodes, self.seq_len * self.lstm_hidden_dim)
        x = self.linear(lstm_out)  # Apply final linear layer
        x = torch.squeeze(x)
        return x  # [batch, node]


class SegRNN(nn.Module):
    """
    Segmented RNN forecasting model.
    
    An advanced RNN architecture that segments the input sequence and uses
    GRU cells with positional and channel embeddings for improved forecasting.
    
    Args:
        seq_len (int): Input sequence length.
        pred_len (int): Prediction horizon length.
        n_fea (int): Number of input features.
        seg_len (int): Segment length for segmentation (default: 1).
        d_model (int): Model dimension (default: 256).
        dropout (float): Dropout rate (default: 0.1).
    """

    def __init__(self, seq_len, pred_len, n_fea, seg_len=1, d_model=256, dropout=0.1):
        """Initialize the Segmented RNN model."""
        super(SegRNN, self).__init__()

        # Store parameters
        self.seq_len = seq_len
        self.enc_in = n_fea
        self.d_model = d_model
        self.dropout = dropout
        self.pred_len = pred_len

        # Segmentation parameters
        self.seg_len = seg_len
        self.seg_num_x = self.seq_len // self.seg_len  # Number of input segments
        self.seg_num_y = self.pred_len // self.seg_len  # Number of output segments

        # Build model components
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),  # Segment embedding
            nn.ReLU()
        )
        # GRU layer for sequence processing
        self.rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
                          batch_first=True, bidirectional=False)
        # Positional and channel embeddings
        self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model // 2))  # Positional embedding
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))  # Channel embedding

        # Prediction layers
        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.seg_len)  # Segment prediction
        )
        self.final_linear = nn.Linear(self.enc_in, 1)  # Final output layer

    def encoder(self, x):
        """
        Encode input sequence using segmentation and GRU processing.
        
        This method implements the core encoding logic of the SegRNN model:
        1. Normalizes the input sequence
        2. Segments the sequence into chunks
        3. Applies embeddings and GRU processing
        4. Uses positional and channel embeddings
        5. Generates predictions
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch, seq_len, channels].
            
        Returns:
            torch.Tensor: Encoded output of shape [batch, pred_len, channels].
        """
        # Variable naming convention:
        # b: batch_size, c: channel_size, s: seq_len
        # d: d_model, w: seg_len, n: seg_num_x, m: seg_num_y
        batch_size = x.size(0)

        # Normalization and permute: b,s,c -> b,c,s
        seq_last = x[:, -1:, :].detach()  # Get last sequence value for normalization
        x = (x - seq_last)  # Normalize by subtracting last value: b,c,s

        # Segment and embedding: b,c,s -> bc,n,w -> bc,n,d
        x = self.valueEmbedding(x.reshape(-1, self.seg_num_x, self.seg_len))

        # Encoding with GRU
        _, hn = self.rnn(x)  # bc,n,d -> 1,bc,d (hidden state)

        # Create positional and channel embeddings
        # m,d//2 -> 1,m,d//2 -> c,m,d//2
        # c,d//2 -> c,1,d//2 -> c,m,d//2
        # c,m,d -> cm,1,d -> bcm,1,d
        pos_emb = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),  # Repeat positional embedding
            self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)  # Repeat channel embedding
        ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size, 1, 1)

        # Process embeddings through GRU
        _, hy = self.rnn(pos_emb, hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model))  # bcm,1,d -> 1,bcm,d

        # Generate predictions: 1,bcm,d -> 1,bcm,w -> b,c,s
        y = self.predict(hy).view(-1, self.enc_in, self.pred_len)

        # Permute and denormalize
        y = y.permute(0, 2, 1) + seq_last  # Add back the normalization offset
        y = self.final_linear(y)  # Final linear transformation
        return y

    def forecast(self, x_enc):
        """
        Generate forecasts using the encoder.
        
        Args:
            x_enc (torch.Tensor): Input tensor for forecasting.
            
        Returns:
            torch.Tensor: Forecasted values.
        """
        return self.encoder(x_enc)

    def forward(self, feat, extra_feat=None):
        """
        Forward pass through the SegRNN model.
        
        This method handles the complete forward pass, including chunking for
        memory efficiency and error handling for CUDA memory issues.
        
        Args:
            feat (torch.Tensor): Input features of shape [batch, node, seq].
            extra_feat (torch.Tensor, optional): Auxiliary features.
            
        Returns:
            torch.Tensor: Predictions of shape [batch, node].
        """
        x = feat.unsqueeze(-1)  # Add channel dimension
        if extra_feat is not None:
            # Concatenate main features with auxiliary features
            x = torch.cat([feat.unsqueeze(-1), extra_feat], dim=-1)
        B_ori, node, seq_len, channel = x.shape  # Get tensor dimensions
        x_enc = x.view(B_ori * node, seq_len, channel)  # Reshape for processing
        
        outputs = []  # Store chunk outputs
        current_chunk_size = self.chunk_size  # Initialize chunk size
        start_idx = 0
        
        # Process data in chunks to handle memory constraints
        while start_idx < x_enc.shape[0]:
            try:
                x_chunk = x_enc[start_idx: start_idx + current_chunk_size]  # Extract chunk
                dec_out_chunk = self.forecast(x_chunk)  # Process chunk
                features_chunk = dec_out_chunk[:, -1, :]  # Extract last timestep
                outputs.append(features_chunk)  # Store result
                start_idx += current_chunk_size  # Move to next chunk
            except RuntimeError as e:
                if "CUDA" in str(e):
                    # Reduce chunk size on CUDA memory error
                    current_chunk_size = max(1, current_chunk_size // 2)
                    torch.cuda.empty_cache()  # Clear CUDA cache
                else:
                    raise e  # Re-raise non-CUDA errors
        
        # Combine all chunk outputs
        out = torch.cat(outputs, dim=0)  # Concatenate outputs
        out = out.view(B_ori, node, -1)  # Reshape to original batch format [B, N, pre_len]
        return out


class FreTS(nn.Module):
    """
    Frequency-domain temporal-spatial model.
    
    A neural architecture that operates in the frequency domain using FFT
    transformations for both temporal and spatial dimensions. It combines
    frequency-domain processing with MLPs for time series forecasting.
    
    Args:
        seq_len (int): Input sequence length.
        pred_len (int): Prediction horizon length.
        n_fea (int): Number of input features.
        embed_size (int): Embedding dimension (default: 128).
        hidden_size (int): Hidden layer dimension (default: 256).
        sparsity_threshold (float): Threshold for soft shrinkage (default: 0.01).
        scale (float): Scale factor for weight initialization (default: 0.02).
    """
    
    def __init__(self, seq_len, pred_len, n_fea, embed_size=128, hidden_size=256, sparsity_threshold=0.01, scale=0.02):
        """Initialize the FreTS model."""
        super(FreTS, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.pre_length = pred_len
        self.feature_size = n_fea
        self.seq_length = seq_len
        self.sparsity_threshold = sparsity_threshold
        self.scale = scale
        
        # Learnable embeddings and parameters
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))  # Token embeddings
        
        # Real and imaginary parts of weights for frequency-domain processing
        self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))  # Real weights for channel
        self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))  # Imaginary weights for channel
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))  # Real bias for channel
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))  # Imaginary bias for channel
        
        self.r2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))  # Real weights for temporal
        self.i2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))  # Imaginary weights for temporal
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))  # Real bias for temporal
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))  # Imaginary bias for temporal

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(self.seq_length * self.embed_size, self.hidden_size),  # Flatten and project
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length)  # Final prediction layer
        )
        self.final_linear = nn.Linear(self.feature_size, 1)  # Channel-wise output

    def tokenEmb(self, x):
        """
        Apply token embeddings to input.
        
        Args:
            x (torch.Tensor): Input tensor of shape [Batch, Input length, Channel].
            
        Returns:
            torch.Tensor: Embedded tensor of shape [Batch, Input length, Channel, Embed_size].
        """
        x = x.permute(0, 2, 1)  # [B, C, T]
        x = x.unsqueeze(3)  # [B, C, T, 1]
        # N*T*1 x 1*D = N*T*D (broadcast multiplication)
        y = self.embeddings  # [1, D]
        return x * y  # [B, C, T, D]

    def MLP_temporal(self, x, B, N, L):
        """
        Apply frequency-domain MLP along temporal dimension.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, N, T, D].
            B (int): Batch size.
            N (int): Number of nodes.
            L (int): Sequence length.
            
        Returns:
            torch.Tensor: Processed tensor of shape [B, N, T, D].
        """
        # [B, N, T, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on temporal dimension (L)
        y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)  # Apply frequency MLP
        x = torch.fft.irfft(y, n=self.seq_length, dim=2, norm="ortho")  # Inverse FFT
        return x

    def MLP_channel(self, x, B, N, L):
        """
        Apply frequency-domain MLP along channel dimension.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, N, T, D].
            B (int): Batch size.
            N (int): Number of nodes.
            L (int): Sequence length.
            
        Returns:
            torch.Tensor: Processed tensor of shape [B, N, T, D].
        """
        # [B, N, T, D]
        x = x.permute(0, 2, 1, 3)  # [B, T, N, D]
        # [B, T, N, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on channel dimension (N)
        y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)  # Apply frequency MLP
        x = torch.fft.irfft(y, n=self.feature_size, dim=2, norm="ortho")  # Inverse FFT
        x = x.permute(0, 2, 1, 3)  # [B, N, T, D]
        return x

    def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
        """
        Frequency-domain MLP with complex number operations.
        
        This method implements a frequency-domain MLP that operates on complex
        numbers, applying separate transformations to real and imaginary parts.
        
        Args:
            B (int): Batch size.
            nd (int): Number of dimensions.
            dimension (int): Feature dimension.
            x (torch.Tensor): Complex input tensor.
            r (torch.Tensor): Real part of weights.
            i (torch.Tensor): Imaginary part of weights.
            rb (torch.Tensor): Real part of bias.
            ib (torch.Tensor): Imaginary part of bias.
            
        Returns:
            torch.Tensor: Complex output tensor.
        """
        # Initialize output tensors for real and imaginary parts
        o1_real = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size], device=x.device)
        o1_imag = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size], device=x.device)

        # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        o1_real = F.relu(
            torch.einsum('bijd,dd->bijd', x.real, r) - 
            torch.einsum('bijd,dd->bijd', x.imag, i) + 
            rb  # Real bias
        )

        o1_imag = F.relu(
            torch.einsum('bijd,dd->bijd', x.imag, r) + 
            torch.einsum('bijd,dd->bijd', x.real, i) + 
            ib  # Imaginary bias
        )

        # Combine real and imaginary parts into complex tensor
        y = torch.stack([o1_real, o1_imag], dim=-1)  # Stack along last dimension
        y = F.softshrink(y, lambd=self.sparsity_threshold)  # Apply soft shrinkage
        y = torch.view_as_complex(y)  # Convert to complex tensor
        return y

    def forward(self, feat, extra_feat=None):
        """
        Forward pass through the FreTS model.
        
        This method implements the complete forward pass with chunking for
        memory efficiency and error handling for CUDA memory issues.
        
        Args:
            feat (torch.Tensor): Input features of shape [batch, node, seq].
            extra_feat (torch.Tensor, optional): Auxiliary features.
            
        Returns:
            torch.Tensor: Predictions of shape [batch, node].
        """
        x = feat.unsqueeze(-1)  # Add channel dimension
        if extra_feat is not None:
            # Concatenate main features with auxiliary features
            x = torch.cat([feat.unsqueeze(-1), extra_feat], dim=-1)
        B_ori, node, seq_len, channel = x.shape  # Get tensor dimensions
        x_enc = x.view(B_ori * node, seq_len, channel)  # Reshape for processing
        
        outputs = []  # Store chunk outputs
        current_chunk_size = self.chunk_size  # Initialize chunk size
        start_idx = 0
        
        # Process data in chunks to handle memory constraints
        while start_idx < x_enc.shape[0]:
            try:
                x_chunk = x_enc[start_idx: start_idx + current_chunk_size]  # Extract chunk
                B, T, N = x_chunk.shape  # Get chunk dimensions
                x_chunk = self.tokenEmb(x_chunk)  # Apply token embeddings
                bias = x_chunk  # Store original for residual connection
                x_chunk = self.MLP_channel(x_chunk, B, N, T)  # Process channel dimension
                x_chunk = self.MLP_temporal(x_chunk, B, N, T)  # Process temporal dimension
                x_chunk = x_chunk + bias  # Add residual connection
                x_chunk = self.fc(x_chunk.reshape(B, N, -1)).permute(0, 2, 1)  # Flatten and project
                x_chunk = self.final_linear(x_chunk)  # Final linear transformation
                x_chunk = x_chunk.squeeze(-1)  # Remove last dimension
                outputs.append(x_chunk)  # Store result
                start_idx += current_chunk_size  # Move to next chunk
            except RuntimeError as e:
                if "CUDA" in str(e):
                    # Reduce chunk size on CUDA memory error
                    current_chunk_size = max(1, current_chunk_size // 2)
                    torch.cuda.empty_cache()  # Clear CUDA cache
                else:
                    raise e  # Re-raise non-CUDA errors
        
        # Combine all chunk outputs
        out = torch.cat(outputs, dim=0)  # Concatenate outputs
        out = out.view(B_ori, node, -1)  # Reshape to original batch format [B, N, pre_len]
        return out


class ModernTCN(nn.Module):
    def __init__(self, n_fea, seq_len, pred_len, patch_size=16, patch_stride=8, downsample_ratio=2, ffn_ratio=2,
                 num_blocks=[1, 1, 1, 1],
                 large_size=[31, 29, 27, 13], small_size=[5, 5, 5, 5], dims=[256, 256, 256, 256],
                 dw_dims=[256, 256, 256, 256],
                 small_kernel_merged=False, backbone_dropout=0.1, head_dropout=0.1, use_multi_scale=True, revin=True,
                 affine=True,
                 subtract_last=False, individual=False, ):

        super(ModernTCN, self).__init__()
        c_in = n_fea
        target_window = pred_len
        # RevIN
        self.revin = revin
        if self.revin: self.revin_layer = ModernTCN_RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # stem layer & down sampling layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv1d(1, dims[0], kernel_size=patch_size, stride=patch_stride),
            nn.BatchNorm1d(dims[0])
        )
        self.downsample_layers.append(stem)

        self.num_stage = len(num_blocks)
        if self.num_stage > 1:
            for i in range(self.num_stage - 1):
                downsample_layer = nn.Sequential(
                    nn.BatchNorm1d(dims[i]),
                    nn.Conv1d(dims[i], dims[i + 1], kernel_size=downsample_ratio, stride=downsample_ratio),
                )
                self.downsample_layers.append(downsample_layer)

        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.downsample_ratio = downsample_ratio

        # backbone
        self.num_stage = len(num_blocks)
        self.stages = nn.ModuleList()
        for stage_idx in range(self.num_stage):
            layer = ModernTCN_Stage(ffn_ratio, num_blocks[stage_idx], large_size[stage_idx], small_size[stage_idx],
                                    dmodel=dims[stage_idx],
                                    dw_model=dw_dims[stage_idx], nvars=n_fea, small_kernel_merged=small_kernel_merged,
                                    drop=backbone_dropout)
            self.stages.append(layer)

        # head
        patch_num = seq_len // patch_stride
        self.n_vars = c_in
        self.individual = individual
        d_model = dims[self.num_stage - 1]

        if use_multi_scale:
            self.head_nf = d_model * patch_num
            self.head = ModernTCN_Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window,
                                               head_dropout=head_dropout)
        else:

            if patch_num % pow(downsample_ratio, (self.num_stage - 1)) == 0:
                self.head_nf = d_model * patch_num // pow(downsample_ratio, (self.num_stage - 1))
            else:
                self.head_nf = d_model * (patch_num // pow(downsample_ratio, (self.num_stage - 1)) + 1)
            self.head = ModernTCN_Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window,
                                               head_dropout=head_dropout)

        self.final_linear = nn.Linear(n_fea, 1)

    def forward_feature(self, x):

        B, M, L = x.shape
        x = x.unsqueeze(-2)

        for i in range(self.num_stage):
            B, M, D, N = x.shape
            x = x.reshape(B * M, D, N)
            if i == 0:
                if self.patch_size != self.patch_stride:
                    pad_len = self.patch_size - self.patch_stride
                    pad = x[:, :, -1:].repeat(1, 1, pad_len)
                    x = torch.cat([x, pad], dim=-1)
            else:
                if N % self.downsample_ratio != 0:
                    pad_len = self.downsample_ratio - (N % self.downsample_ratio)
                    x = torch.cat([x, x[:, :, -pad_len:]], dim=-1)
            x = self.downsample_layers[i](x)
            _, D_, N_ = x.shape
            x = x.reshape(B, M, D_, N_)
            x = self.stages[i](x)
        return x

    def forward(self, feat, extra_feat=None):
        x = feat.unsqueeze(-1)
        if extra_feat is not None:
            x = torch.cat([feat.unsqueeze(-1), extra_feat], dim=-1)
        B_ori, node, seq_len, channel = x.shape
        x_enc = x.view(B_ori * node, seq_len, channel)
        outputs = []
        current_chunk_size = self.chunk_size
        start_idx = 0
        while start_idx < x_enc.shape[0]:
            try:
                x_chunk = x_enc[start_idx: start_idx + current_chunk_size]
                if self.revin:
                    x_chunk = self.revin_layer(x_chunk, 'norm')
                    x_chunk = x_chunk.permute(0, 2, 1)
                x_chunk = self.forward_feature(x_chunk)
                x_chunk = self.head(x_chunk)
                # de-instance norm
                if self.revin:
                    x_chunk = x_chunk.permute(0, 2, 1)
                    x_chunk = self.revin_layer(x_chunk, 'denorm')
                x_chunk = self.final_linear(x_chunk)
                x_chunk = x_chunk.squeeze(-1)
                outputs.append(x_chunk)
                start_idx += current_chunk_size
            except RuntimeError as e:
                if "CUDA" in str(e):
                    current_chunk_size = max(1, current_chunk_size // 2)
                    torch.cuda.empty_cache()
                else:
                    raise e
        out = torch.cat(outputs, dim=0)
        out = out.view(B_ori, node, -1)  # fix multi-step shape [B, N, pre_len]
        return out

    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()


class MultiPatchFormer(nn.Module):
    def __init__(self, seq_len, pred_len, n_fea, e_layers=2, d_model=256, d_ff=1024, n_heads=8, dropout=0.1):
        super(MultiPatchFormer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_channel = n_fea
        self.N = e_layers
        # Embedding
        self.d_model = d_model
        self.d_hidden = d_ff
        self.n_heads = n_heads
        self.mask = True
        self.dropout = dropout

        self.stride1 = 3
        self.patch_len1 = 4
        self.stride2 = 4
        self.patch_len2 = 6
        self.stride3 = 3
        self.patch_len3 = 8
        self.stride4 = 2
        self.patch_len4 = 10
        self.patch_num1 = int((self.seq_len - self.patch_len2) // self.stride2) + 2
        self.padding_patch_layer1 = nn.ReplicationPad1d((0, self.stride1))
        self.padding_patch_layer2 = nn.ReplicationPad1d((0, self.stride2))
        self.padding_patch_layer3 = nn.ReplicationPad1d((0, self.stride3))
        self.padding_patch_layer4 = nn.ReplicationPad1d((0, self.stride4))

        self.shared_MHA = nn.ModuleList(
            [
                AttentionLayer(
                    FullAttention(mask_flag=self.mask),
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                )
                for _ in range(self.N)
            ]
        )

        self.shared_MHA_ch = nn.ModuleList(
            [
                AttentionLayer(
                    FullAttention(mask_flag=self.mask),
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                )
                for _ in range(self.N)
            ]
        )

        self.encoder_list = nn.ModuleList(
            [
                MultiPatchFormer_Encoder(
                    d_model=self.d_model,
                    mha=self.shared_MHA[ll],
                    d_hidden=self.d_hidden,
                    dropout=self.dropout,
                    channel_wise=False,
                )
                for ll in range(self.N)
            ]
        )

        self.encoder_list_ch = nn.ModuleList(
            [
                MultiPatchFormer_Encoder(
                    d_model=self.d_model,
                    mha=self.shared_MHA_ch[0],
                    d_hidden=self.d_hidden,
                    dropout=self.dropout,
                    channel_wise=True,
                )
                for ll in range(self.N)
            ]
        )

        pe = torch.zeros(self.patch_num1, self.d_model)
        for pos in range(self.patch_num1):
            for i in range(0, self.d_model, 2):
                wavelength = 10000 ** ((2 * i) / self.d_model)
                pe[pos, i] = math.sin(pos / wavelength)
                pe[pos, i + 1] = math.cos(pos / wavelength)
        pe = pe.unsqueeze(0)  # add a batch dimention to your pe matrix
        self.register_buffer("pe", pe)

        self.embedding_channel = nn.Conv1d(
            in_channels=self.d_model * self.patch_num1,
            out_channels=self.d_model,
            kernel_size=1,
        )

        self.embedding_patch_1 = torch.nn.Conv1d(
            in_channels=1,
            out_channels=self.d_model // 4,
            kernel_size=self.patch_len1,
            stride=self.stride1,
        )
        self.embedding_patch_2 = torch.nn.Conv1d(
            in_channels=1,
            out_channels=self.d_model // 4,
            kernel_size=self.patch_len2,
            stride=self.stride2,
        )
        self.embedding_patch_3 = torch.nn.Conv1d(
            in_channels=1,
            out_channels=self.d_model // 4,
            kernel_size=self.patch_len3,
            stride=self.stride3,
        )
        self.embedding_patch_4 = torch.nn.Conv1d(
            in_channels=1,
            out_channels=self.d_model // 4,
            kernel_size=self.patch_len4,
            stride=self.stride4,
        )

        self.out_linear_1 = torch.nn.Linear(self.d_model, self.pred_len // 8)
        self.out_linear_2 = torch.nn.Linear(
            self.d_model + self.pred_len // 8, self.pred_len // 8
        )
        self.out_linear_3 = torch.nn.Linear(
            self.d_model + 2 * self.pred_len // 8, self.pred_len // 8
        )
        self.out_linear_4 = torch.nn.Linear(
            self.d_model + 3 * self.pred_len // 8, self.pred_len // 8
        )
        self.out_linear_5 = torch.nn.Linear(
            self.d_model + self.pred_len // 2, self.pred_len // 8
        )
        self.out_linear_6 = torch.nn.Linear(
            self.d_model + 5 * self.pred_len // 8, self.pred_len // 8
        )
        self.out_linear_7 = torch.nn.Linear(
            self.d_model + 6 * self.pred_len // 8, self.pred_len // 8
        )
        self.out_linear_8 = torch.nn.Linear(
            self.d_model + 7 * self.pred_len // 8,
            self.pred_len - 7 * (self.pred_len // 8),
        )

        self.remap = torch.nn.Linear(self.d_model, self.seq_len)
        self.final_linear = nn.Linear(n_fea, 1)

    def forecast(self, x_enc):
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Multi-scale embedding
        x_i = x_enc.permute(0, 2, 1)

        x_i_p1 = x_i
        x_i_p2 = self.padding_patch_layer2(x_i)
        x_i_p3 = self.padding_patch_layer3(x_i)
        x_i_p4 = self.padding_patch_layer4(x_i)
        encoding_patch1 = self.embedding_patch_1(
            rearrange(x_i_p1, "b c l -> (b c) l").unsqueeze(-1).permute(0, 2, 1)
        ).permute(0, 2, 1)
        encoding_patch2 = self.embedding_patch_2(
            rearrange(x_i_p2, "b c l -> (b c) l").unsqueeze(-1).permute(0, 2, 1)
        ).permute(0, 2, 1)
        encoding_patch3 = self.embedding_patch_3(
            rearrange(x_i_p3, "b c l -> (b c) l").unsqueeze(-1).permute(0, 2, 1)
        ).permute(0, 2, 1)
        encoding_patch4 = self.embedding_patch_4(
            rearrange(x_i_p4, "b c l -> (b c) l").unsqueeze(-1).permute(0, 2, 1)
        ).permute(0, 2, 1)

        encoding_patch = (
                torch.cat(
                    (encoding_patch1, encoding_patch2, encoding_patch3, encoding_patch4),
                    dim=-1,
                )
                + self.pe
        )
        # Temporal encoding
        for i in range(self.N):
            encoding_patch = self.encoder_list[i](encoding_patch)[0]

        # Channel-wise encoding
        x_patch_c = rearrange(
            encoding_patch, "(b c) p d -> b c (p d)", b=x_enc.shape[0], c=self.d_channel
        )
        x_ch = self.embedding_channel(x_patch_c.permute(0, 2, 1)).transpose(
            1, 2
        )  # [b c d]

        encoding_1_ch = self.encoder_list_ch[0](x_ch)[0]

        # Semi Auto-regressive
        forecast_ch1 = self.out_linear_1(encoding_1_ch)
        forecast_ch2 = self.out_linear_2(
            torch.cat((encoding_1_ch, forecast_ch1), dim=-1)
        )
        forecast_ch3 = self.out_linear_3(
            torch.cat((encoding_1_ch, forecast_ch1, forecast_ch2), dim=-1)
        )
        forecast_ch4 = self.out_linear_4(
            torch.cat((encoding_1_ch, forecast_ch1, forecast_ch2, forecast_ch3), dim=-1)
        )
        forecast_ch5 = self.out_linear_5(
            torch.cat(
                (encoding_1_ch, forecast_ch1, forecast_ch2, forecast_ch3, forecast_ch4),
                dim=-1,
            )
        )
        forecast_ch6 = self.out_linear_6(
            torch.cat(
                (
                    encoding_1_ch,
                    forecast_ch1,
                    forecast_ch2,
                    forecast_ch3,
                    forecast_ch4,
                    forecast_ch5,
                ),
                dim=-1,
            )
        )
        forecast_ch7 = self.out_linear_7(
            torch.cat(
                (
                    encoding_1_ch,
                    forecast_ch1,
                    forecast_ch2,
                    forecast_ch3,
                    forecast_ch4,
                    forecast_ch5,
                    forecast_ch6,
                ),
                dim=-1,
            )
        )
        forecast_ch8 = self.out_linear_8(
            torch.cat(
                (
                    encoding_1_ch,
                    forecast_ch1,
                    forecast_ch2,
                    forecast_ch3,
                    forecast_ch4,
                    forecast_ch5,
                    forecast_ch6,
                    forecast_ch7,
                ),
                dim=-1,
            )
        )

        final_forecast = torch.cat(
            (
                forecast_ch1,
                forecast_ch2,
                forecast_ch3,
                forecast_ch4,
                forecast_ch5,
                forecast_ch6,
                forecast_ch7,
                forecast_ch8,
            ),
            dim=-1,
        ).permute(0, 2, 1)

        # De-Normalization
        dec_out = final_forecast * (
            stdev[:, 0].unsqueeze(1).repeat(1, self.pred_len, 1)
        )
        dec_out = dec_out + (means[:, 0].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, feat, extra_feat=None):
        x = feat.unsqueeze(-1)
        if extra_feat is not None:
            x = torch.cat([feat.unsqueeze(-1), extra_feat], dim=-1)
        B_ori, node, seq_len, channel = x.shape
        x_enc = x.view(B_ori * node, seq_len, channel)
        outputs = []
        current_chunk_size = self.chunk_size
        start_idx = 0
        while start_idx < x_enc.shape[0]:
            try:
                x_chunk = x_enc[start_idx: start_idx + current_chunk_size]
                dec_out = self.forecast(x_chunk)
                dec_out = self.final_linear(dec_out)
                dec_out = dec_out.squeeze(-1)
                outputs.append(dec_out)
                start_idx += current_chunk_size
            except RuntimeError as e:
                if "CUDA" in str(e):
                    current_chunk_size = max(1, current_chunk_size // 2)
                    torch.cuda.empty_cache()
                else:
                    raise e
        out = torch.cat(outputs, dim=0)
        out = out.view(B_ori, node, -1)  # fix multi-step shape [B, N, pre_len]
        return out


class ConvTimeNet(nn.Module):
    def __init__(self, c_in: int, c_out: int, seq_len: int, n_layers: int = 3, d_model: int = 128,
                 d_ff: int = 256, dropout=0.1, act: str = "relu", pooling_tp='max', fc_dropout: float = 0.,
                 enable_res_param=False, dw_ks=[7, 13, 19], norm='batch', use_embed=True, re_param=False,
                 device: str = 'cuda:0'):

        super(ConvTimeNet, self).__init__()
        assert n_layers == len(dw_ks), "dw_ks should match the n_layers!"

        self.c_out, self.seq_len = c_out, seq_len

        # Input Embedding
        self.use_embed = use_embed
        self.W_P = nn.Linear(c_in, d_model)

        # Positional encoding
        # W_pos = torch.empty((seq_len, d_model), device=device)
        # nn.init.uniform_(W_pos, -0.02, 0.02)
        # self.W_pos = nn.Parameter(W_pos, requires_grad=True)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = ConvTimeNet_ConvEncoder(d_model, d_ff, kernel_size=dw_ks, dropout=dropout, activation=act, \
                                               n_layers=n_layers, enable_res_param=enable_res_param, norm=norm,
                                               re_param=re_param,
                                               device=device)

        self.flatten = nn.Flatten()

        # Head
        self.head_nf = seq_len * d_model if pooling_tp == 'cat' else d_model
        self.head = self.create_head(self.head_nf, c_out, act=act, pooling_tp=pooling_tp, fc_dropout=fc_dropout)

    def create_head(self, nf, c_out, act="gelu", pooling_tp='max', fc_dropout=0., **kwargs):
        layers = []
        if pooling_tp == 'cat':
            layers = [ConvTimeNet_get_activation_fn(act), self.flatten]
            if fc_dropout: layers += [nn.Dropout(fc_dropout)]
        elif pooling_tp == 'mean':
            layers = [nn.AdaptiveAvgPool1d(1), self.flatten]
        elif pooling_tp == 'max':
            layers = [nn.AdaptiveMaxPool1d(1), self.flatten]

        layers += [nn.Linear(nf, c_out)]

        # could just be used in classifying task
        return nn.Sequential(*layers)

    def forward(self, feat, extra_feat=None) -> torch.Tensor:
        x = feat.unsqueeze(-1)
        if extra_feat is not None:
            x = torch.cat([feat.unsqueeze(-1), extra_feat], dim=-1)
        B_ori, node, seq_len, channel = x.shape
        x_enc = x.view(B_ori * node, seq_len, channel)
        outputs = []
        current_chunk_size = self.chunk_size
        start_idx = 0
        while start_idx < x_enc.shape[0]:
            try:
                x_chunk = x_enc[start_idx: start_idx + current_chunk_size]
                x_chunk = x_chunk.permute(0, 2, 1)
                if self.use_embed:
                    u = self.W_P(x_chunk.transpose(2, 1))
                z = self.encoder(u.transpose(2, 1).contiguous())
                out = self.head(z)
                outputs.append(out)
                start_idx += current_chunk_size
            except RuntimeError as e:
                if "CUDA" in str(e):
                    current_chunk_size = max(1, current_chunk_size // 2)
                    torch.cuda.empty_cache()
                else:
                    raise e
        out = torch.cat(outputs, dim=0)
        out = out.view(B_ori, node, -1)  # fix multi-step shape [B, N, pre_len]
        return out


class SatCrossAttention(nn.Module):
    """Forecasting model that fuses temporal tokens with satellite embeddings."""

    def __init__(
            self,
            seq_len: int,
            sat_dim: int,
            d_model: int = 128,
            n_heads: int = 4,
            dropout: float = 0.1,
    ):
        super(SatCrossAttention, self).__init__()
        if sat_dim <= 0:
            raise ValueError("sat_dim must be > 0 for SatCrossAttention")

        self.seq_len = seq_len
        self.sat_dim = sat_dim
        self.temporal_proj = nn.Linear(seq_len, d_model)
        self.sat_proj = nn.Linear(sat_dim, d_model)
        self.cross_block = CrossModalAttentionBlock(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.out = nn.Linear(d_model, 1)

    def _extract_sat_embedding(self, extra_feat: torch.Tensor) -> torch.Tensor:
        """Support [B, N, C] and [B, N, S, C] auxiliary tensors."""
        if extra_feat.dim() == 4:
            # Keep the latest-step embedding when auxiliary features are time-indexed.
            sat = extra_feat[:, :, -1, :]
        elif extra_feat.dim() == 3:
            sat = extra_feat
        else:
            raise ValueError(
                "extra_feat for SatCrossAttention must be [B, N, C] or [B, N, S, C]"
            )

        if sat.shape[-1] != self.sat_dim:
            raise ValueError(
                f"Expected satellite embedding dim={self.sat_dim}, got {sat.shape[-1]}"
            )
        return sat

    def forward(self, feat, extra_feat=None):
        if extra_feat is None:
            raise ValueError("SatCrossAttention requires satellite embeddings via extra_feat")

        # feat: [B, N, S] -> one temporal token per node.
        temporal_token = self.temporal_proj(feat)
        sat_embed = self._extract_sat_embedding(extra_feat)
        sat_token = self.sat_proj(sat_embed)

        bsz, num_nodes, _ = temporal_token.shape
        # MultiheadAttention expects 3D tensors: [batch, seq_len, d_model].
        query = temporal_token.reshape(bsz * num_nodes, 1, -1)
        context = sat_token.reshape(bsz * num_nodes, 1, -1)

        fused = self.cross_block(query, context).squeeze(1)
        fused = fused.reshape(bsz, num_nodes, -1)
        pred = self.out(fused).squeeze(-1)
        return pred


class SatFreTS(FreTS):
    """FreTS backbone with satellite-aware gated fusion head."""

    def __init__(
            self,
            seq_len: int,
            pred_len: int,
            sat_dim: int,
            embed_size: int = 128,
            hidden_size: int = 256,
            sparsity_threshold: float = 0.01,
            scale: float = 0.02,
            d_model: int = 128,
    ):
        # Keep FreTS backbone for temporal modeling, use only target series channel.
        super(SatFreTS, self).__init__(
            seq_len=seq_len,
            pred_len=pred_len,
            n_fea=1,
            embed_size=embed_size,
            hidden_size=hidden_size,
            sparsity_threshold=sparsity_threshold,
            scale=scale,
        )
        if sat_dim <= 0:
            raise ValueError("sat_dim must be > 0 for SatFreTS")

        self.sat_dim = sat_dim
        self.temporal_proj = nn.Linear(pred_len, d_model)
        self.sat_proj = nn.Linear(sat_dim, d_model)
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.out = nn.Linear(d_model, 1)

    def _extract_sat_embedding(self, extra_feat: torch.Tensor) -> torch.Tensor:
        """Support [B, N, C] and [B, N, S, C] auxiliary tensors."""
        if extra_feat.dim() == 4:
            sat = extra_feat[:, :, -1, :]
        elif extra_feat.dim() == 3:
            sat = extra_feat
        else:
            raise ValueError(
                "extra_feat for SatFreTS must be [B, N, C] or [B, N, S, C]"
            )

        if sat.shape[-1] < self.sat_dim:
            raise ValueError(
                f"Expected satellite embedding dim >= {self.sat_dim}, got {sat.shape[-1]}"
            )

        # Use the trailing channels as satellite embedding when extra channels exist.
        return sat[..., -self.sat_dim:]

    def _encode_temporal(self, feat: torch.Tensor) -> torch.Tensor:
        """Encode temporal series with FreTS backbone and return [B, N, pred_len]."""
        x = feat.unsqueeze(-1)
        bsz, num_nodes, _, _ = x.shape
        x_enc = x.view(bsz * num_nodes, self.seq_length, 1)

        outputs = []
        current_chunk_size = self.chunk_size
        start_idx = 0
        while start_idx < x_enc.shape[0]:
            try:
                x_chunk = x_enc[start_idx: start_idx + current_chunk_size]
                batch_chunk, time_chunk, channel_chunk = x_chunk.shape
                x_chunk = self.tokenEmb(x_chunk)
                bias = x_chunk
                x_chunk = self.MLP_channel(x_chunk, batch_chunk, channel_chunk, time_chunk)
                x_chunk = self.MLP_temporal(x_chunk, batch_chunk, channel_chunk, time_chunk)
                x_chunk = x_chunk + bias
                x_chunk = self.fc(x_chunk.reshape(batch_chunk, channel_chunk, -1)).squeeze(1)
                outputs.append(x_chunk)
                start_idx += current_chunk_size
            except RuntimeError as e:
                if "CUDA" in str(e):
                    current_chunk_size = max(1, current_chunk_size // 2)
                    torch.cuda.empty_cache()
                else:
                    raise e

        temporal = torch.cat(outputs, dim=0)
        return temporal.view(bsz, num_nodes, self.pre_length)

    def forward(self, feat, extra_feat=None):
        if extra_feat is None:
            raise ValueError("SatFreTS requires satellite embeddings via extra_feat")

        temporal_repr = self._encode_temporal(feat)
        sat_embed = self._extract_sat_embedding(extra_feat)

        temporal_token = self.temporal_proj(temporal_repr)
        sat_token = self.sat_proj(sat_embed)

        gate = self.fusion_gate(torch.cat([temporal_token, sat_token], dim=-1))
        fused = gate * sat_token + (1.0 - gate) * temporal_token
        return self.out(fused).squeeze(-1)
