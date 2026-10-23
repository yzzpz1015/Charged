# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/10 16:16
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/10 16:16


"""
Main entry point for single-city EV charging demand prediction.

Parses command-line arguments, initializes dataset and model,
performs cross-validation split, trains (if enabled), and tests.
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import torch

# Ensure parent directory is in path for package imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from api.parsing.common import parse_args
from api.dataset.common import EVDataset
from api.model.config import PredictionModel
from api.trainer.common import PredictionTrainer
from api.utils import random_seed, get_n_feature, Logger


def main():
    """
    Execute single-city EV demand prediction workflow.

    Steps:
    1. Parse CLI arguments.
    2. Configure output directory and logging.
    3. Seed randomness and select compute device.
    4. Load EVDataset and initialize model.
    5. Perform cross-validation split and DataLoader creation.
    6. Train model if enabled and evaluate on test set.
    """
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1
    args = parse_args()
    if args.model == 'astgcn' and args.seq_l != 48:
        args.seq_l = 48

    # Resolve the EV-VLM LLM backbone to local weights unless explicitly overridden
    if args.model == 'ev_vlm' and args.gpt2_model_path is None:
        args.gpt2_model_path = os.path.join('api', 'model', 'weights', f'{args.llm_backbone}_model')

    # Build unique output path (relative to script directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, args.output_path)
    base_path = f'{output_path}{args.city}/{args.model}/{args.feature}-{args.auxiliary}-{args.pred_type}-{args.seq_l}-{args.pre_len}-{args.fold}'
    new_path = base_path
    counter = 0
    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base_path}#{counter}/"
    os.makedirs(new_path)
    sys.stdout=Logger(os.path.join(new_path, 'logging.txt'))

    # Device and reproducibility
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    random_seed(args.seed)

    # Load dataset. 用 _remove_const（已剔恒值站）——与基线 script/prob_baseline.py
    # 同口径，否则站数不同、数字不能放一张表（SZH: 858 vs 1379）。
    data_dir = f"{args.data_path}{args.city}_remove_const/"
    ev_dataset = EVDataset(
        feature=args.feature,
        auxiliary=args.auxiliary,
        data_path=data_dir,
        sat_embed_root=args.sat_embed_root or None,
        sat_embed_zoom=args.sat_embed_zoom,
        sat_embed_dim=args.sat_embed_dim,
        sat_embed_strict=args.sat_embed_strict,
        poi_topk=args.poi_topk,
        poi_min_count=args.poi_min_count,
        poi_radius_m=args.poi_radius_m,
        poi_radius_weights=args.poi_radius_weights,
    )
    print(
        f"Running {args.model} on {args.city} with feature={args.feature}, "
        f"pre_len={args.pre_len}, fold={args.fold}, auxiliary={args.auxiliary}, pred_type={args.pred_type}"
    )

    # Cross-validation split. **必须早于读 num_node**：split 会按"训练期均值为 0"
    # 剔站（与基线 script/prob_baseline.py 同口径），站数在那之后才定下来。
    ev_dataset.split_cross_validation(
        fold=args.fold,
        total_fold=args.total_fold,
        train_ratio=TRAIN_RATIO,
        valid_ratio=VAL_RATIO,
    )
    print(
        f"Split outcome - Training set: {len(ev_dataset.train_feat)}, Validation set: {len(ev_dataset.valid_feat)}, Test set: {len(ev_dataset.test_feat)}"
    )

    # Determine number of nodes and features
    num_node = ev_dataset.feat.shape[1]
    n_fea = get_n_feature(ev_dataset.extra_feat)

    # Initialize prediction model
    ev_model = PredictionModel(
        num_node=num_node,
        n_fea=n_fea,
        model_name=args.model,
        seq_l=args.seq_l,
        pre_len=args.pre_len,
        sat_dim=args.sat_embed_dim,
        vit_image_size=args.vit_image_size,
        vit_patch_size=args.vit_patch_size,
        vit_embed_dim=args.vit_embed_dim,
        vit_depth=args.vit_depth,
        vit_heads=args.vit_heads,
        fusion_dim=args.fusion_dim,
        gpt2_model_path=args.gpt2_model_path,
        siglip_model_path=args.siglip_model_path,
        ablate_proto=args.ablate_proto,
        ablate_dyn=args.ablate_dyn,
        spatial_early=args.spatial_early,
        no_gat=args.no_gat,
        plain_gat=args.plain_gat,
        gate_bias=args.gate_bias,
        qformer_queries=args.qformer_queries,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
        llm_chunk_size=args.llm_chunk_size,
        ablate_qformer=args.ablate_qformer,
        ablate_text=args.ablate_text,
    )
    if args.model not in ['lo','ar','arima']:
        ev_model.model=ev_model.model.to(device)

    # Create DataLoaders
    skip_images = False

    ev_dataset.create_loaders(
        seq_l=args.seq_l,
        pre_len=args.pre_len,
        batch_size=args.batch_size,
        device=device,
        model_name=args.model,
        tiles_path=args.tiles_path,
        city=args.city,
        image_size=args.vit_image_size,
        skip_images=skip_images,
        gpt2_model_path=args.gpt2_model_path,
        siglip_model_path=args.siglip_model_path,
    )

    # Initialize trainer
    ev_trainer = PredictionTrainer(
        dataset=ev_dataset,
        model=ev_model,
        seq_l=args.seq_l,
        pre_len=args.pre_len,
        is_train=args.is_train,
        save_path=new_path,
        lr=args.lr,
        weight_decay=args.weight_decay,
        early_stop_patience=20,
        ev_vlm_mode=args.ev_vlm_mode,
    )

    # Train and test
    if ev_trainer.is_train:
        ev_trainer.training(epoch=args.epoch)
        # Evaluate on the best-validation checkpoint (train.pth stores trainable params only)
        ev_trainer.test(os.path.join(new_path, 'train.pth'))
    else:
        ev_trainer.test()


if __name__ == '__main__':
    main()
