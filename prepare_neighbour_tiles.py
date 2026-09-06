"""
为每个站点获取邻域卫星图，并记录邻点在图像中的位置。
"""

import argparse
import csv
import io
import json
import math
import os
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from PIL import Image


TILE_SIZE = 256
ZOOM = 16
PROVIDER = "esri"
STYLE = 6
TIMEOUT = 8
RETRIES = 1
MAX_WORKERS = 8


_thread_local = threading.local()


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


def haversine(lat1, lon1, lat2, lon2):
    """计算两点间的大圆距离（km）"""
    R = 6371  # 地球半径（km）
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c


def find_neighbors_dynamic(site_idx, sites_df):
    """
    以3个邻站点为基准，找出该站点的邻点。
    返回：(邻点索引列表, 邻点距离列表, 所用半径km)
    """
    target_lon = sites_df.loc[site_idx, 'longitude']
    target_lat = sites_df.loc[site_idx, 'latitude']
    
    radii = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    
    for radius in radii:
        neighbors = []
        distances = []
        
        for other_idx in range(len(sites_df)):
            if other_idx == site_idx:
                continue
            
            other_lon = sites_df.loc[other_idx, 'longitude']
            other_lat = sites_df.loc[other_idx, 'latitude']
            dist = haversine(target_lat, target_lon, other_lat, other_lon)
            
            if dist <= radius:
                neighbors.append(other_idx)
                distances.append(dist)
        
        # 如果达到3个邻点，立即返回
        if len(neighbors) >= 3:
            return neighbors, distances, radius
    
    # 如果最大半径也达不到3个，返回10.0km时的邻点
    neighbors = []
    distances = []
    for other_idx in range(len(sites_df)):
        if other_idx == site_idx:
            continue
        
        other_lon = sites_df.loc[other_idx, 'longitude']
        other_lat = sites_df.loc[other_idx, 'latitude']
        dist = haversine(target_lat, target_lon, other_lat, other_lon)
        
        if dist <= 10.0:
            neighbors.append(other_idx)
            distances.append(dist)
    
    return neighbors, distances, 10.0


def calculate_required_tiles(max_distance_km):
    """
    根据最远邻点距离，计算所需的瓦片数。
    距离 + 470m（一个瓦片的边长）作为边界。
    在MEL纬度（-37.68°），一个瓦片≈470m。
    """
    tile_size_km = 0.470  # 一个瓦片的大小（km）
    
    # 边界距离 = 最远邻点距离 + 一个瓦片大小
    boundary_distance_km = max_distance_km + tile_size_km
    
    # 所需的半径（从中心到边界）
    required_half_km = boundary_distance_km
    
    # 从中心出发需要多少个瓦片
    tiles_from_center = required_half_km / tile_size_km
    
    # 总边长（双向）
    side_tiles = math.ceil(tiles_from_center * 2)
    
    # 确保是偶数（保持中心对称）
    if side_tiles % 2 == 1:
        side_tiles += 1
    
    return side_tiles


def tile_url(x: int, y: int, z: int, provider: str = "esri", style: int = 6) -> str:
    if provider == "esri":
        return f"https://server.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}.png"
    if provider == "amap":
        return f"https://webst01.is.autonavi.com/appmaptile?style={style}&x={x}&y={y}&z={z}"
    raise ValueError(f"Unknown provider: {provider}")


def get_thread_session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": "charged-neighbour-tiles/1.0"})
        _thread_local.session = session
    return session


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


def download_one_tile(session, x: int, y: int, z: int, save_path: str, provider: str) -> bool:
    """Download a single tile."""
    if os.path.exists(save_path):
        return True

    url = tile_url(x, y, z, provider=provider, style=STYLE)
    try:
        resp = session.get(url, timeout=TIMEOUT)
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


def download_one_tile_with_retry(x: int, y: int, z: int, save_path: str, retries: int, provider: str) -> bool:
    for _ in range(retries + 1):
        session = get_thread_session()
        if download_one_tile(session, x, y, z, save_path, provider):
            return True
    return False


def stitch_tiles(tile_dir: str, window: TileWindow, out_path: str) -> bool:
    """Stitch tiles together into a single image."""
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


def lonlat_to_pixel_in_window(lon: float, lat: float, zoom: int, window: TileWindow) -> tuple[float, float]:
    """
    将经纬度坐标转换为瓦片窗口中的像素坐标。
    返回 (pixel_x, pixel_y)，其中原点 (0, 0) 在左上角。
    """
    # 转换为瓦片浮点坐标
    tile_x, tile_y = lonlat_to_tile_float(lon, lat, zoom)
    
    # 相对于窗口左上角的瓦片坐标
    relative_tile_x = tile_x - window.x_min
    relative_tile_y = tile_y - window.y_min
    
    # 转换为像素坐标
    pixel_x = relative_tile_x * TILE_SIZE
    pixel_y = relative_tile_y * TILE_SIZE
    
    return pixel_x, pixel_y


def process_site(
    site_idx: int,
    sites_df: pd.DataFrame,
    output_dir: str,
    tiles_cache_dir: str,
    session: requests.Session,
    provider: str,
    max_workers: int = MAX_WORKERS,
) -> dict:
    """
    为一个站点处理邻域瓦片。
    返回：处理结果的字典
    """
    site_id = int(sites_df.loc[site_idx, 'site'])
    target_lon = sites_df.loc[site_idx, 'longitude']
    target_lat = sites_df.loc[site_idx, 'latitude']
    
    # 找邻点
    neighbor_indices, neighbor_distances, used_radius = find_neighbors_dynamic(site_idx, sites_df)
    
    result = {
        'site_id': site_id,
        'center_lon': target_lon,
        'center_lat': target_lat,
        'neighbor_count': len(neighbor_indices),
        'used_radius_km': used_radius,
        'status': 'success',
        'error': None,
        'image_path': None,
        'neighbors_file': None
    }
    
    # 计算所需瓦片数
    if len(neighbor_indices) == 0:
        max_distance_km = 0.0
    else:
        max_distance_km = max(neighbor_distances)
    
    side_tiles = calculate_required_tiles(max_distance_km)
    result['side_tiles'] = side_tiles
    result['max_neighbor_distance_km'] = max_distance_km
    
    # 获取瓦片窗口
    window = fixed_square_window(target_lon, target_lat, ZOOM, side_tiles)
    result['tile_window'] = {
        'x_min': window.x_min,
        'x_max': window.x_max,
        'y_min': window.y_min,
        'y_max': window.y_max,
        'count_x': window.count_x,
        'count_y': window.count_y,
    }
    
    # 计算经纬度范围（瓦片索引转经纬度）
    n = 2.0 ** ZOOM
    
    # 瓦片索引的经纬度转换
    # 左边界（x_min）和右边界（x_max + 1）
    lon_min = (window.x_min / n) * 360.0 - 180.0
    lon_max = ((window.x_max + 1) / n) * 360.0 - 180.0
    
    # 上边界（y_min）和下边界（y_max + 1）
    lat_max = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * window.y_min / n))))
    lat_min = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * (window.y_max + 1) / n))))
    
    result['bbox'] = {
        'lon_min': lon_min,
        'lon_max': lon_max,
        'lat_min': lat_min,
        'lat_max': lat_max,
    }
    
    # 创建站点的瓦片缓存目录
    site_tiles_dir = os.path.join(tiles_cache_dir, f"site_{site_id}")
    os.makedirs(site_tiles_dir, exist_ok=True)
    
    # 下载瓦片
    print(f"  Site {site_id}: 下载 {window.tile_count} 个瓦片 ({window.count_x}x{window.count_y})...", end='', flush=True)

    tile_jobs = []
    for y in range(window.y_min, window.y_max + 1):
        for x in range(window.x_min, window.x_max + 1):
            tile_path = os.path.join(site_tiles_dir, f"{x}_{y}.png")
            tile_jobs.append((x, y, ZOOM, tile_path))

    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(download_one_tile_with_retry, x, y, z, tile_path, RETRIES, provider)
            for x, y, z, tile_path in tile_jobs
        ]

        try:
            for index, future in enumerate(as_completed(futures), start=1):
                ok = future.result()
                if ok:
                    success_count += 1
                else:
                    fail_count += 1

                if index % 32 == 0 or index == len(futures):
                    print(f"\r  Site {site_id}: 下载 {index}/{window.tile_count} 完成", end='', flush=True)
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            print(f"\n  Site {site_id}: 下载已中断")
            raise

    print()
    
    print(f" {success_count}/{window.tile_count} 成功")
    
    if success_count < window.tile_count:
        result['status'] = 'partial'
        result['error'] = f'Only {success_count}/{window.tile_count} tiles downloaded'
    
    # 拼接瓦片
    print(f"  Site {site_id}: 拼接瓦片...", end='', flush=True)
    
    mosaic_path = os.path.join(output_dir, f"site_{site_id}_mosaic.png")
    if stitch_tiles(site_tiles_dir, window, mosaic_path):
        result['image_path'] = mosaic_path
        print(f" 完成")
    else:
        result['status'] = 'stitch_failed'
        result['error'] = 'Failed to stitch tiles'
        print(f" 失败")
        return result
    
    # 记录邻点信息
    print(f"  Site {site_id}: 记录邻点位置...", end='', flush=True)
    
    neighbors_data = {
        'site_id': site_id,
        'center_lon': float(target_lon),
        'center_lat': float(target_lat),
        'image_size_pixels': {
            'width': window.count_x * TILE_SIZE,
            'height': window.count_y * TILE_SIZE,
        },
        'tile_window': {
            'x_min': window.x_min,
            'x_max': window.x_max,
            'y_min': window.y_min,
            'y_max': window.y_max,
        },
        'neighbors': []
    }
    
    # 计算每个邻点的像素位置
    for neighbor_idx, neighbor_distance in zip(neighbor_indices, neighbor_distances):
        neighbor_lon = sites_df.loc[neighbor_idx, 'longitude']
        neighbor_lat = sites_df.loc[neighbor_idx, 'latitude']
        
        pixel_x, pixel_y = lonlat_to_pixel_in_window(
            neighbor_lon, neighbor_lat, ZOOM, window
        )
        
        neighbors_data['neighbors'].append({
            'site_id': int(sites_df.loc[neighbor_idx, 'site']),
            'longitude': float(neighbor_lon),
            'latitude': float(neighbor_lat),
            'distance_km': float(neighbor_distance),
            'pixel_x': float(pixel_x),
            'pixel_y': float(pixel_y),
        })
    
    # 保存邻点信息
    neighbors_file = os.path.join(output_dir, f"site_{site_id}_neighbors.json")
    with open(neighbors_file, 'w') as f:
        json.dump(neighbors_data, f, indent=2)
    
    result['neighbors_file'] = neighbors_file
    print(f" 完成 ({len(neighbor_indices)} 个邻点)")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Prepare neighbour tiles for MEL charging stations')
    parser.add_argument('--data-root', default='data', help='Root directory of data')
    parser.add_argument('--output-root', default='tiles_dataset', help='Output root directory')
    parser.add_argument('--city', default='MEL', help='City name')
    parser.add_argument('--provider', default='esri', choices=['esri', 'amap'], help='Tile provider')
    parser.add_argument('--max-workers', type=int, default=MAX_WORKERS, help='Concurrent download workers per site')
    args = parser.parse_args()
    
    # 加载站点数据
    sites_path = os.path.join(args.data_root, f"{args.city}_remove_zero", "sites.csv")
    if not os.path.exists(sites_path):
        print(f"Error: {sites_path} not found")
        return
    
    sites_df = pd.read_csv(sites_path)
    print(f"Loaded {len(sites_df)} sites from {args.city}")
    
    # 创建输出目录
    output_dir = os.path.join(args.output_root, f"{args.city}_neighbour")
    tiles_cache_dir = os.path.join(output_dir, "tiles_cache")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tiles_cache_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # 创建HTTP session
    session = requests.Session()
    session.headers.update({"User-Agent": "charged-neighbour-tiles/1.0"})
    
    # 处理每个站点
    results = []
    
    for site_idx in range(len(sites_df)):
        site_id = int(sites_df.loc[site_idx, 'site'])
        print(f"\nProcessing Site {site_id} ({site_idx + 1}/{len(sites_df)})...")
        
        try:
            result = process_site(
                site_idx,
                sites_df,
                output_dir,
                tiles_cache_dir,
                session,
                provider=args.provider,
                max_workers=args.max_workers,
            )
            results.append(result)
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'site_id': site_id,
                'status': 'error',
                'error': str(e)
            })
    
    # 保存汇总结果
    summary_file = os.path.join(output_dir, "processing_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nProcessing complete. Summary saved to {summary_file}")
    
    # 打印统计
    success = sum(1 for r in results if r['status'] == 'success')
    partial = sum(1 for r in results if r['status'] == 'partial')
    failed = sum(1 for r in results if r['status'] not in ['success', 'partial'])
    
    print(f"\nSummary:")
    print(f"  Success: {success}")
    print(f"  Partial: {partial}")
    print(f"  Failed: {failed}")


if __name__ == '__main__':
    main()
