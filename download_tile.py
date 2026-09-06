import argparse
import csv
import io
import math
import os
from dataclasses import dataclass

import pandas as pd
import requests
from PIL import Image


DEFAULT_CITIES = ["AMS", "JHB", "LOA", "MEL", "SPO", "SZH"]
TILE_SIZE = 256


@dataclass
class TileWindow:
    x_min: int
    x_max: int
    y_min: int
    y_max: int

    @property
    def count_x(self) -> int:
        return self.x_max - self.x_min + 1

    @property
    def count_y(self) -> int:
        return self.y_max - self.y_min + 1

    @property
    def tile_count(self) -> int:
        return self.count_x * self.count_y


def lonlat_to_tile_float(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    """Convert WGS84 lon/lat to slippy-map tile coordinates (float)."""
    lat = max(min(lat, 85.05112878), -85.05112878)
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def bbox_km_around_point(lon: float, lat: float, half_km: float) -> tuple[float, float, float, float]:
    """Return (min_lon, max_lon, min_lat, max_lat) for a square window in km."""
    lat_deg_per_km = 1.0 / 110.574
    lon_deg_per_km = 1.0 / (111.320 * max(math.cos(math.radians(lat)), 1e-8))
    d_lat = half_km * lat_deg_per_km
    d_lon = half_km * lon_deg_per_km
    return lon - d_lon, lon + d_lon, lat - d_lat, lat + d_lat


def tile_window_from_bbox(min_lon: float, max_lon: float, min_lat: float, max_lat: float, zoom: int) -> TileWindow:
    """Compute inclusive tile index window for a lon/lat bbox."""
    x1, y1 = lonlat_to_tile_float(min_lon, max_lat, zoom)  # northwest
    x2, y2 = lonlat_to_tile_float(max_lon, min_lat, zoom)  # southeast
    x_min = math.floor(min(x1, x2))
    x_max = math.floor(max(x1, x2))
    y_min = math.floor(min(y1, y2))
    y_max = math.floor(max(y1, y2))
    return TileWindow(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)


def fixed_square_window(lon: float, lat: float, zoom: int, side_tiles: int) -> TileWindow:
    """Compute a fixed-size square tile window centered at lon/lat."""
    if side_tiles <= 0:
        raise ValueError("side_tiles must be > 0")

    n = 2 ** zoom
    if side_tiles > n:
        raise ValueError(f"side_tiles ({side_tiles}) cannot exceed tile span at zoom {zoom} ({n})")

    center_x, center_y = lonlat_to_tile_float(lon, lat, zoom)
    half = side_tiles / 2.0

    x_min = math.floor(center_x - half)
    y_min = math.floor(center_y - half)
    x_max = x_min + side_tiles - 1
    y_max = y_min + side_tiles - 1

    max_idx = n - 1

    if x_min < 0:
        x_max += -x_min
        x_min = 0
    if x_max > max_idx:
        x_min -= x_max - max_idx
        x_max = max_idx

    if y_min < 0:
        y_max += -y_min
        y_min = 0
    if y_max > max_idx:
        y_min -= y_max - max_idx
        y_max = max_idx

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(max_idx, x_max)
    y_max = min(max_idx, y_max)

    return TileWindow(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)


def tile_url(x: int, y: int, z: int, provider: str, style: int = 6) -> str:
    if provider == "esri":
        return f"https://server.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}.png"
    if provider == "amap":
        return f"https://webst01.is.autonavi.com/appmaptile?style={style}&x={x}&y={y}&z={z}"
    raise ValueError(f"Unknown provider: {provider}")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_site_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    if "site_id" in df.columns:
        site_col = "site_id"
    elif "site" in df.columns:
        site_col = "site"
    else:
        raise KeyError("sites.csv must contain 'site_id' or 'site'")

    if "longitude" not in df.columns or "latitude" not in df.columns:
        raise KeyError("sites.csv must contain 'longitude' and 'latitude'")

    return site_col, "longitude", "latitude"


def download_one_tile(
    session: requests.Session,
    x: int,
    y: int,
    z: int,
    save_path: str,
    provider: str,
    style: int,
    timeout: int,
) -> bool:
    if os.path.exists(save_path):
        return True

    url = tile_url(x, y, z, provider=provider, style=style)
    try:
        resp = session.get(url, timeout=timeout)
        if resp.status_code == 200 and resp.content:
            content_type = resp.headers.get("Content-Type", "").lower()
            if content_type and "image" not in content_type:
                return False

            # Validate bytes as an image before writing to disk.
            try:
                with Image.open(io.BytesIO(resp.content)) as img:
                    img.verify()
            except (OSError, ValueError):
                return False

            tmp_path = f"{save_path}.tmp"
            with open(tmp_path, "wb") as f:
                f.write(resp.content)
            os.replace(tmp_path, save_path)
            return True
        return False
    except requests.RequestException:
        return False


def stitch_tiles(tile_dir: str, window: TileWindow, out_path: str) -> bool:
    width = window.count_x * TILE_SIZE
    height = window.count_y * TILE_SIZE
    canvas = Image.new("RGB", (width, height))

    for y in range(window.y_min, window.y_max + 1):
        for x in range(window.x_min, window.x_max + 1):
            tile_path = os.path.join(tile_dir, f"{x}_{y}.png")
            if not os.path.exists(tile_path):
                return False
            try:
                with Image.open(tile_path) as tile_img:
                    tile = tile_img.convert("RGB")
            except (OSError, ValueError):
                return False
            px = (x - window.x_min) * TILE_SIZE
            py = (y - window.y_min) * TILE_SIZE
            canvas.paste(tile, (px, py))

    try:
        canvas.save(out_path)
        return True
    except OSError:
        return False


def process_city(
    city: str,
    data_root: str,
    out_root: str,
    zoom: int,
    region_km: float,
    provider: str,
    style: int,
    timeout: int,
    retries: int,
    skip_existing_mosaic: bool,
    fixed_tile_side: int,
) -> list[dict]:
    city_data_dir = os.path.join(data_root, f"{city}_remove_zero")
    sites_path = os.path.join(city_data_dir, "sites.csv")
    if not os.path.exists(sites_path):
        raise FileNotFoundError(f"Missing sites.csv: {sites_path}")

    df = pd.read_csv(sites_path)
    site_col, lon_col, lat_col = resolve_site_columns(df)

    city_out_dir = os.path.join(out_root, city)
    tiles_dir = os.path.join(city_out_dir, "tiles", f"z{zoom}")
    mosaic_dir = os.path.join(city_out_dir, "mosaic", f"z{zoom}")
    ensure_dir(tiles_dir)
    ensure_dir(mosaic_dir)

    session = requests.Session()
    session.headers.update({"User-Agent": "charged-tile-downloader/1.0"})

    half_km = region_km / 2.0
    rows: list[dict] = []

    for idx, row in df.iterrows():
        site_id = str(row[site_col])
        try:
            lon = float(row[lon_col])
            lat = float(row[lat_col])
        except (TypeError, ValueError):
            rows.append(
                {
                    "city": city,
                    "site_id": site_id,
                    "center_lon": row.get(lon_col),
                    "center_lat": row.get(lat_col),
                    "region_km": region_km,
                    "zoom": zoom,
                    "provider": provider,
                    "x_min": "",
                    "x_max": "",
                    "y_min": "",
                    "y_max": "",
                    "tile_count": 0,
                    "download_success": 0,
                    "download_failed": 0,
                    "mosaic_path": "",
                    "status": "invalid_coord",
                    "bbox_min_lon": "",
                    "bbox_max_lon": "",
                    "bbox_min_lat": "",
                    "bbox_max_lat": "",
                }
            )
            continue

        min_lon, max_lon, min_lat, max_lat = bbox_km_around_point(lon, lat, half_km)
        if fixed_tile_side > 0:
            window = fixed_square_window(lon, lat, zoom, fixed_tile_side)
        else:
            window = tile_window_from_bbox(min_lon, max_lon, min_lat, max_lat, zoom)

        site_tile_dir = os.path.join(tiles_dir, site_id)
        ensure_dir(site_tile_dir)
        mosaic_path = os.path.join(mosaic_dir, f"{site_id}.png")

        success_count = 0
        fail_count = 0

        if not (skip_existing_mosaic and os.path.exists(mosaic_path)):
            for y in range(window.y_min, window.y_max + 1):
                for x in range(window.x_min, window.x_max + 1):
                    tile_path = os.path.join(site_tile_dir, f"{x}_{y}.png")

                    ok = False
                    for _ in range(retries + 1):
                        ok = download_one_tile(
                            session=session,
                            x=x,
                            y=y,
                            z=zoom,
                            save_path=tile_path,
                            provider=provider,
                            style=style,
                            timeout=timeout,
                        )
                        if ok:
                            break

                    if ok:
                        success_count += 1
                    else:
                        fail_count += 1

            mosaic_ok = stitch_tiles(site_tile_dir, window, mosaic_path)
            status = "ok" if mosaic_ok and fail_count == 0 else "partial"
        else:
            status = "cached"

        rows.append(
            {
                "city": city,
                "site_id": site_id,
                "center_lon": lon,
                "center_lat": lat,
                "region_km": region_km,
                "zoom": zoom,
                "provider": provider,
                "fixed_tile_side": fixed_tile_side,
                "x_min": window.x_min,
                "x_max": window.x_max,
                "y_min": window.y_min,
                "y_max": window.y_max,
                "tile_count": window.tile_count,
                "download_success": success_count,
                "download_failed": fail_count,
                "mosaic_path": mosaic_path,
                "status": status,
                "bbox_min_lon": min_lon,
                "bbox_max_lon": max_lon,
                "bbox_min_lat": min_lat,
                "bbox_max_lat": max_lat,
            }
        )

        if (idx + 1) % 100 == 0:
            print(f"[{city}] processed {idx + 1}/{len(df)} sites")

    return rows


def write_index(rows: list[dict], out_csv: str) -> None:
    ensure_dir(os.path.dirname(out_csv))
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and stitch site-centered 2kmx2km satellite tiles for *_remove_zero datasets."
    )
    parser.add_argument("--data_root", type=str, default="data", help="Root path containing city_remove_zero folders.")
    parser.add_argument("--out_root", type=str, default="tiles_dataset", help="Root output folder.")
    parser.add_argument("--cities", type=str, nargs="+", default=DEFAULT_CITIES, help="City codes, e.g. AMS JHB LOA MEL SPO SZH")
    parser.add_argument("--zoom", type=int, default=16, help="Tile zoom level.")
    parser.add_argument("--region_km", type=float, default=2.0, help="Square region size in km.")
    parser.add_argument("--provider", type=str, default="esri", choices=["esri", "amap"], help="Tile provider.")
    parser.add_argument("--style", type=int, default=6, help="Map tile style code.")
    parser.add_argument("--timeout", type=int, default=10, help="HTTP timeout seconds.")
    parser.add_argument("--retries", type=int, default=2, help="Retry count per tile.")
    parser.add_argument(
        "--fixed_tile_side",
        type=int,
        default=0,
        help="If > 0, force every site to use a fixed side_tiles x side_tiles square tile window.",
    )
    parser.add_argument("--skip_existing_mosaic", action="store_true", help="Skip download/stitch if mosaic image already exists.")
    parser.add_argument("--index_name", type=str, default="site_tile_index.csv", help="Index CSV filename.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    all_rows: list[dict] = []
    for city in args.cities:
        print(f"Processing city: {city}")
        city_rows = process_city(
            city=city,
            data_root=args.data_root,
            out_root=args.out_root,
            zoom=args.zoom,
            region_km=args.region_km,
            provider=args.provider,
            style=args.style,
            timeout=args.timeout,
            retries=args.retries,
            skip_existing_mosaic=args.skip_existing_mosaic,
            fixed_tile_side=args.fixed_tile_side,
        )
        all_rows.extend(city_rows)

    index_path = os.path.join(args.out_root, args.index_name)
    write_index(all_rows, index_path)
    print(f"Done. Index saved to: {index_path}")
    print(f"Total sites processed: {len(all_rows)}")


if __name__ == "__main__":
    main()