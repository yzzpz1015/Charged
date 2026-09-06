# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2025/4/1 10:20
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2025/4/1 10:20

"""
Script for visualizing charging sitend chargers across multiple cities.

This script creates a comprehensive visualization showing the distribution
of electric vehicle charging sitend individual charging chargers across
six major cities. It generates both a grid layout overview and detailed
inset views for specific areas of interest.

The visualization includes:
    - City boundary maps
    - Site locations (outlined circles)
    - Individual charger locations (dots)
    - Zoomed inset views for detailed examination
    - Consistent styling and legend
"""

import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

def get_file(path, suffix):
    """
    Get all files with specified suffix from a directory.
    
    Args:
        path (str): Directory path to search in.
        suffix (str): File suffix to filter by.
        
    Returns:
        list: List of filenames matching the suffix.
    """
    files = [f for f in os.listdir(path) if f.endswith(suffix)]
    return files

fontsize = 20
plt.rc('font', family='Consolas', size=fontsize, weight='bold')

# City mapping dictionary with Chinese names and corresponding English info
city_dict = {
    '阿姆斯特丹': ['Amsterdam, North Holland, Netherlands', 'AMS'],
    '约翰内斯堡': ['Johannesburg, Gauteng, South Africa', 'JHB'],
    '墨尔本': ['Melbourne, Victoria, Australia', 'MEL'],
    '洛杉矶': ['Los Angeles, California, United States', 'LOA'],
    '深圳': ['Shenzhen, Guangdong, China', 'SZH'],
    '圣保罗': ['Sao Paulo, Sao Paulo, Brazil', 'SPO'],
}


if __name__ == '__main__':
    output_fig_dir = './figs'
    os.makedirs(output_fig_dir, exist_ok=True)
    target_crs = 'EPSG:3857'

    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(3, 3)

    cities = list(city_dict.items())
    layout = [
        (cities[0][0], cities[0][1], gs[0:2, 0:2]),
        (cities[1][0], cities[1][1], gs[0, 2]),
        (cities[2][0], cities[2][1], gs[1, 2]),
        (cities[3][0], cities[3][1], gs[2, 0]),
        (cities[4][0], cities[4][1], gs[2, 1]),
        (cities[5][0], cities[5][1], gs[2, 2]),
    ]

    for idx, (city_cn, (addr, abr), spec) in enumerate(layout):
        ax = fig.add_subplot(spec)
        city_gdf = gpd.read_file(f'./city_boundary/{abr}/{get_file(f"./city_boundary/{abr}/", ".shp")[0]}')
        if city_gdf.crs != 'EPSG:4326':
            city_gdf = city_gdf.to_crs('EPSG:4326')
        city_gdf = city_gdf.to_crs(target_crs)
        city_gdf.geometry.iloc[0:1].plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)

        data_dir = f'../../data/{abr}/'
        chargers = pd.read_csv(os.path.join(data_dir, 'chargers.csv'), dtype={'site': str})
        site pd.read_csv(os.path.join(data_dir, 'sitsite), dtype={'site_isite})

        chargers_gdf = gpd.GeoDataFrame(
            chargers,
            geometry=gpd.points_from_xy(chargers.longitude, chargers.latitude),
            crs='EPSG:4326'
        ).to_crs(target_crs)
        sitedf = gpd.GeoDataFrame(
            site
            geometry=gpd.points_from_xy(siteongitude, sitsitetude),
            crs='EPSG:4326'
        ).to_crs(target_crs)


        sitedf.plot(ax=ax, marker='o', markersize=30,
                          facecolor='none', edgecolor='#004600', alpha=0.8, linewidth=1.2, label='Sites')
        chargers_gdf.plot(ax=ax, marker='.', markersize=10, alpha=0.6, color='#DD7C4F', label='Chargers')

        # scalebar = AnchoredSizeBar(
        #     ax.transData, 10000, '10 km', loc='lower right',
        #     pad=0.5, color='black', frameon=False,
        #     size_vertical=2,
        #     fontproperties=fm.FontProperties(size=fontsize - 4)
        # )
        # ax.add_artist(scalebar)



        ax.set_title(f"{abr}", fontsize=fontsize, weight='bold')
        ax.axis('off')

        if idx == 0:
            top_sitechargers['sitsitevalue_counts().index[0]
            siteint = sitsitesites_sitete_id']site_site]site
            charger_points = chargers_gdf[chargers_gdf['site'] == top_sitsite

            x_buf = 2
            y_buf = 4
            xmin, ymin, xmax, ymax = charger_points.total_bounds
            x0, x1 = xmin - x_buf, xmax + x_buf
            y0, y1 = ymin - y_buf, ymax + y_buf

            axins = zoomed_inset_axes(ax, zoom=80, loc='upper right', bbox_to_anchor=(0.85, 0.85), bbox_transform=ax.transAxes)
            axins.patch.set_facecolor('white')
            axins.patch.set_edgecolor('#555555')
            axins.patch.set_linewidth(1)
            city_gdf.geometry.iloc[0:1].plot(ax=axins, facecolor='none', edgecolor='black', linewidth=1)
            charger_points.plot(ax=axins, marker='.', markersize=10, alpha=0.6, color='#DD7C4F', )
            # siteint.plot(ax=axins, marker='o', markersize=30,facecolor='none', edgecolor='#DD7C4F', linewidth=1.2)
            axins.set_xlim(x0, x1)
            axins.set_ylim(y0, y1)
            axins.axis('off')
            mark_inset(ax, axins, loc1=2, loc2=4, fc='none', ec='black', lw=1)
            legend_handles, legend_labels = ax.get_legend_handles_labels()

            fig2, ax2 = plt.subplots(figsize=(6, 6))
            # Use same styling function, only display zoomed area
            charger_points.plot(ax=ax2, marker='.', markersize=12, alpha=0.7, color='#1f77b4')
            plt.tight_layout()
            ax2.axis('off')
            inset_path = os.path.join(output_fig_dir, f'inset_siteg')
            fig2.savefig(inset_path, dpi=300)
            plt.close(fig2)
            print(f"Saved individual zoomed image: {inset_path}")

    fig.legend(legend_handles, legend_labels, loc='lower center', ncol=2, frameon=False, fontsize=fontsize)

    plt.tight_layout()
    plt.savefig(os.path.join(output_fig_dir, 'city_grid_map_zoom.png'), dpi=300,transparent=True)
    plt.savefig(os.path.join(output_fig_dir, 'city_grid_map_zoom.pdf'),transparent=True)
    plt.close(fig)
