"""PatchTST wrapper: delegates to HuggingFace transformers.PatchTSTForPrediction.

Paper: A Time Series is Worth 64 Words (Nie et al., ICLR 2023).
"""
import torch
import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTForPrediction


class PatchTST(nn.Module):
    """PatchTST using the official HuggingFace implementation.

    Input:  [B, N, T]  — our internal convention (N=stations, T=timesteps)
    Output: [B, N, pred_len]

    Architecture (handled by HF):
        RevIN → Patching → Transformer Encoder → Flatten + Linear → RevIN denorm
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_vars: int,
        patch_len: int = 12,
        stride: int = 6,
        d_model: int = 128,
        nhead: int = 8,
        n_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()

        config = PatchTSTConfig(
            num_input_channels=num_vars,
            context_length=seq_len,
            prediction_length=pred_len,
            patch_length=patch_len,
            patch_stride=stride,
            d_model=d_model,
            num_attention_heads=nhead,
            num_hidden_layers=n_layers,
            ffn_dim=d_ff,
            attention_dropout=dropout,
            ff_dropout=dropout,
            head_dropout=dropout,
            positional_dropout=dropout,
            do_mask_input=False,
        )
        self.model = PatchTSTForPrediction(config)

    def forward(self, x: torch.Tensor, extra: torch.Tensor = None) -> torch.Tensor:
        """x: [B, N, T] → [B, N, pred_len]"""
        # HF expects past_values [B, T, N]
        out = self.model(past_values=x.permute(0, 2, 1))
        # HF returns [B, pred_len, N] → transpose back
        return out.prediction_outputs.permute(0, 2, 1)
