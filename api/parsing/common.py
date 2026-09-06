# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/10 16:48
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/10 16:48
"""
Module for parsing command-line arguments for EV charging demand prediction across multiple cities.

Provides a single function `parse_args` that returns an `argparse.Namespace` of all configuration parameters.
"""
import argparse


def parse_args():
    """
    Parse and return command-line arguments for the EV prediction script.

    Returns:
        argparse.Namespace: Parsed arguments with the following attributes:
            city (str): City abbreviation (default: 'SPO').
            device (int): CUDA device index (default: 0).
            seed (int): Random seed for reproducibility (default: 2025).
            feature (str): Feature type to predict ('volume' or 'duration').
            auxiliary (str): Auxiliary variable(s) to include (default: 'None').
            data_path (str): Path to input data directory (default: '../data/').
            output_path (str): Directory to save results (default: './result/univariate/').
            model (str): Model name to use for prediction (default: 'convtimenet').
            seq_l (int): Input sequence length (default: 12).
            pre_len (int): Prediction horizon length (default: 1).
            fold (int): Current fold index for cross-validation (default: 1).
            total_fold (int): Total number of folds (default: 6).
            pred_type (str): Granularity of prediction ('site' or 'city').
            batch_size (int): Batch size for training (default: 32).
            epoch (int): Maximum number of training epochs (default: 50).
            is_train (bool): Flag indicating training mode (default: True).
    """
    parser = argparse.ArgumentParser(
        description="EV Charging Demand Prediction across Multiple Cities Worldwide!"
    )

    parser.add_argument(
        '--city', type=str, default='SPO', help="City abbreviation."
    )
    parser.add_argument(
        '--device', type=int, default=0, help="CUDA device index."
    )
    parser.add_argument(
        '--seed', type=int, default=2025, help="Random seed."
    )
    parser.add_argument(
        '--feature', type=str, default='volume', help="Which feature to use for prediction ('volume' or 'duration')."
    )
    parser.add_argument(
        '--auxiliary', type=str, default='None',
        help="Auxiliary variable(s) to include ('None', 'all', or combination)."
    )
    parser.add_argument(
        '--sat_embed_root', type=str, default='',
        help="Root folder of cached satellite embeddings, e.g. 'tiles_dataset'."
    )
    parser.add_argument(
        '--tiles_path', type=str, default='tiles_dataset/',
        help="Root folder of satellite tile images."
    )
    parser.add_argument(
        '--sat_embed_zoom', type=str, default='z16',
        help="Zoom folder name used for cached satellite embeddings, e.g. z16."
    )
    parser.add_argument(
        '--sat_embed_dim', type=int, default=256,
        help="Satellite embedding dimension used in sat_embed files."
    )
    parser.add_argument(
        '--sat_embed_strict', action='store_true', default=False,
        help="Fail when satellite embedding files or site mappings are incomplete."
    )
    parser.add_argument(
        '--vit_image_size', type=int, default=384,
        help="Resize size for the ViT image branch."
    )
    parser.add_argument(
        '--vit_patch_size', type=int, default=16,
        help="Patch size for the ViT image branch."
    )
    parser.add_argument(
        '--vit_embed_dim', type=int, default=128,
        help="Embedding width for the ViT image branch."
    )
    parser.add_argument(
        '--vit_depth', type=int, default=4,
        help="Number of transformer blocks in the ViT image branch."
    )
    parser.add_argument(
        '--vit_heads', type=int, default=4,
        help="Number of attention heads in the ViT image branch."
    )
    parser.add_argument(
        '--fusion_dim', type=int, default=256,
        help="Shared latent dimension used before tri-branch fusion."
    )
    parser.add_argument(
        '--poi_topk', type=int, default=32,
        help="Maximum number of POI types kept when auxiliary includes 'poi'."
    )
    parser.add_argument(
        '--poi_min_count', type=int, default=50,
        help="Minimum global POI count for a POI type to be included."
    )
    parser.add_argument(
        '--poi_radius_m', type=str, default='300,800,1500',
        help="Comma-separated POI radius layers in meters, e.g. 300,800,1500."
    )
    parser.add_argument(
        '--poi_radius_weights', type=str, default='1.0,0.5,0.2',
        help="Comma-separated layer weights aligned with poi_radius_m."
    )
    parser.add_argument(
        '--data_path', type=str, default='../data/', help="Path to data directory."
    )
    parser.add_argument(
        '--output_path', type=str, default='../result/univariate/', help="Path to save results."
    )
    parser.add_argument(
        '--model', type=str, default='convtimenet', help="Model name to use."
    )
    parser.add_argument(
        '--seq_l', type=int, default=12, help="Input sequence length."
    )
    parser.add_argument(
        '--pre_len', type=int, default=1, help="Prediction horizon length."
    )
    parser.add_argument(
        '--fold', type=int, default=1, help="Current fold number for cross-validation."
    )
    parser.add_argument(
        '--total_fold', type=int, default=6, help="Total number of folds for cross-validation."
    )
    parser.add_argument(
        '--pred_type', type=str, default='site', help="Prediction granularity ('site' or 'city')."
    )
    parser.add_argument(
        '--batch_size', type=int, default=32, help="Batch size for training."
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3, help="Learning rate for neural model optimizer."
    )
    parser.add_argument(
        '--weight_decay', type=float, default=1e-5, help="Weight decay for neural model optimizer."
    )
    parser.add_argument(
        '--epoch', type=int, default=50, help="Maximum training epochs."
    )
    parser.add_argument(
        '--is_train', action='store_true', default=True,
        help="Flag indicating whether to run in training mode."
    )
    parser.add_argument(
        '--gpt2_model_path', type=str, default=None,
        help="Local path to GPT-2 model files (e.g. api/model/weights/gpt2_model/)."
    )
    parser.add_argument(
        '--llm_backbone', type=str, default='distilgpt2', choices=['gpt2', 'distilgpt2'],
        help="EV-VLM LLM backbone: 'distilgpt2' (6 layers, faster) or 'gpt2' (12 layers)."
    )
    parser.add_argument(
        '--siglip_model_path', type=str, default=None,
        help="Local path to SigLIP model files (e.g. api/model/weights/siglip2/)."
    )
    parser.add_argument(
        '--ev_vlm_mode', type=str, default='staged', choices=['staged', 'end_to_end'],
        help="EV-VLM training mode: 'staged' (3-stage) or 'end_to_end' (single stage 3)."
    )
    parser.add_argument(
        '--ablate_proto', action='store_true', default=False,
        help="[deprecated] Ablation: skip PrototypeSemanticLayer."
    )
    parser.add_argument(
        '--ablate_dyn', action='store_true', default=False,
        help="Ablation: mask out the dynamic time/stats text tokens (time + stats)."
    )
    parser.add_argument(
        '--ablate_qformer', action='store_true', default=False,
        help="Ablation: mask out the Q-former satellite tokens."
    )
    parser.add_argument(
        '--ablate_text', action='store_true', default=False,
        help="Ablation: mask out the entire text prompt (shared + time + stats + static)."
    )
    parser.add_argument(
        '--qformer_queries', type=int, default=32,
        help="Number of Q-former learnable queries compressing satellite tokens."
    )
    parser.add_argument(
        '--lora_rank', type=int, default=8,
        help="LoRA rank applied to GPT-2 c_attn."
    )
    parser.add_argument(
        '--lora_alpha', type=int, default=16,
        help="LoRA alpha scaling (scaling = alpha / rank)."
    )
    parser.add_argument(
        '--patch_len', type=int, default=6,
        help="Patch length for temporal patch embedding."
    )
    parser.add_argument(
        '--patch_stride', type=int, default=3,
        help="Patch stride for temporal patch embedding."
    )
    parser.add_argument(
        '--llm_chunk_size', type=int, default=64,
        help="Number of B*N sequences fed to GPT-2 per forward chunk (bounds memory)."
    )
    parser.add_argument(
        '--spatial_early', action='store_true', default=False,
        help="[deprecated] Place SpatialGCN right after PatchTST encoder."
    )
    parser.add_argument(
        '--gate_bias', type=float, default=0.0,
        help="[deprecated] Gate bias for CrossModalGateFusion init."
    )
    parser.add_argument(
        '--no_gat', action='store_true', default=False,
        help="[deprecated] Ablation: skip RelationGAT entirely."
    )
    parser.add_argument(
        '--plain_gat', action='store_true', default=False,
        help="[deprecated] Use plain GAT (no satellite conditioning) for spatial relation."
    )

    return parser.parse_args()
