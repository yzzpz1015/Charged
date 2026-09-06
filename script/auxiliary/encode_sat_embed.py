# -*- coding: utf-8 -*-
"""
Encode satellite mosaic images into fixed-dimensional site embeddings.

Example:
    python script/auxiliary/encode_sat_embed.py --cities MEL SZH --zoom z16 --embed_dim 256
"""

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image

# Ensure project root is importable when running script directly.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from api.model.layers import SatelliteImageEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode city mosaic images into fixed-dimensional site embeddings."
    )
    parser.add_argument(
        "--cities",
        type=str,
        nargs="+",
        required=True,
        help="City codes, e.g. MEL SZH AMS",
    )
    parser.add_argument(
        "--zoom",
        type=str,
        default="z16",
        help="Zoom folder under mosaic directory, e.g. z16",
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=256,
        help="Output embedding dimension.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Root folder that contains <CITY>_remove_zero/sites.csv",
    )
    parser.add_argument(
        "--tiles_root",
        type=str,
        default="tiles_dataset",
        help="Root folder that contains <CITY>/mosaic/<zoom>",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for encoding.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any site mosaic image is missing.",
    )
    return parser.parse_args()


def resolve_site_column(df: pd.DataFrame) -> str:
    if "site_id" in df.columns:
        return "site_id"
    if "site" in df.columns:
        return "site"
    raise KeyError("sites.csv must contain 'site_id' or 'site'")


def choose_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return device_arg


def load_sites(city: str, data_root: str) -> List[str]:
    sites_path = os.path.join(data_root, f"{city}_remove_zero", "sites.csv")
    if not os.path.exists(sites_path):
        raise FileNotFoundError(f"Missing sites file: {sites_path}")

    df = pd.read_csv(sites_path)
    site_col = resolve_site_column(df)
    return [str(x) for x in df[site_col].tolist()]


def candidate_site_names(site_id: str) -> List[str]:
    """Generate candidate filename stems for a site id (e.g. 0 <-> 0.0)."""
    candidates = [site_id]
    try:
        value = float(site_id)
        if value.is_integer():
            int_name = str(int(value))
            float_name = f"{int(value)}.0"
            if int_name not in candidates:
                candidates.append(int_name)
            if float_name not in candidates:
                candidates.append(float_name)
    except ValueError:
        pass
    return candidates


def encode_city(
    city: str,
    zoom: str,
    embed_dim: int,
    data_root: str,
    tiles_root: str,
    encoder: SatelliteImageEncoder,
    device: str,
    strict: bool,
) -> Tuple[np.ndarray, List[str], int]:
    site_ids = load_sites(city, data_root)
    mosaic_dir = os.path.join(tiles_root, city, "mosaic", zoom)
    if not os.path.exists(mosaic_dir):
        raise FileNotFoundError(f"Missing mosaic directory: {mosaic_dir}")

    embeddings: List[np.ndarray] = []
    kept_sites: List[str] = []
    missing = 0

    with torch.no_grad():
        for sid in site_ids:
            img_path = None
            for name in candidate_site_names(sid):
                cur_path = os.path.join(mosaic_dir, f"{name}.png")
                if os.path.exists(cur_path):
                    img_path = cur_path
                    break

            if img_path is None:
                missing += 1
                if strict:
                    raise FileNotFoundError(f"Missing mosaic image for site_id={sid}")
                continue

            image = Image.open(img_path).convert("RGB")
            array = np.asarray(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)

            emb = encoder(tensor).squeeze(0).cpu().numpy().astype(np.float32)
            embeddings.append(emb)
            kept_sites.append(sid)

    if embeddings:
        embed_array = np.stack(embeddings, axis=0)
    else:
        embed_array = np.zeros((0, embed_dim), dtype=np.float32)

    return embed_array, kept_sites, missing


def save_city_outputs(
    city: str,
    zoom: str,
    embed_dim: int,
    tiles_root: str,
    embed_array: np.ndarray,
    site_ids: List[str],
) -> Tuple[str, str]:
    out_dir = os.path.join(tiles_root, city, "embedding", zoom)
    os.makedirs(out_dir, exist_ok=True)

    embed_path = os.path.join(out_dir, f"sat_embed_{embed_dim}.npy")
    sites_path = os.path.join(out_dir, f"sat_embed_{embed_dim}_sites.csv")

    np.save(embed_path, embed_array)
    pd.DataFrame({"site_id": site_ids}).to_csv(sites_path, index=False)
    return embed_path, sites_path


def main() -> None:
    args = parse_args()
    if args.embed_dim <= 0:
        raise ValueError("embed_dim must be > 0")

    device = choose_device(args.device)
    encoder = SatelliteImageEncoder(out_dim=args.embed_dim).to(device).eval()

    print(f"Using device: {device}")
    print(f"Cities: {args.cities}")

    for city in args.cities:
        print(f"\nEncoding city: {city}")
        embed_array, kept_sites, missing = encode_city(
            city=city,
            zoom=args.zoom,
            embed_dim=args.embed_dim,
            data_root=args.data_root,
            tiles_root=args.tiles_root,
            encoder=encoder,
            device=device,
            strict=args.strict,
        )
        embed_path, sites_path = save_city_outputs(
            city=city,
            zoom=args.zoom,
            embed_dim=args.embed_dim,
            tiles_root=args.tiles_root,
            embed_array=embed_array,
            site_ids=kept_sites,
        )

        print(f"Embedding shape: {embed_array.shape}")
        print(f"Missing mosaics: {missing}")
        print(f"Embedding file: {embed_path}")
        print(f"Site index file: {sites_path}")


if __name__ == "__main__":
    main()
