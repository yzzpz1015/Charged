# -*- coding: utf-8 -*-
"""
api.utils 包 - 包含所有工具函数和空间处理模块
"""

import sys
import os

# 获取原有 utils.py 的路径（在 api 目录中）
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
utils_py_path = os.path.join(parent_dir, 'utils.py')

# 动态加载原有 utils.py 中的所有内容
if os.path.exists(utils_py_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("utils_legacy", utils_py_path)
    utils_legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils_legacy)
    
    # 导入所有公共API
    Logger = utils_legacy.Logger
    random_seed = utils_legacy.random_seed
    get_n_feature = utils_legacy.get_n_feature
    CreateDataset = utils_legacy.CreateDataset
    EV_VLM_Dataset = utils_legacy.EV_VLM_Dataset
    ASTGCNDataset = utils_legacy.ASTGCNDataset
    create_rnn_data = utils_legacy.create_rnn_data
    split_window_starts = utils_legacy.split_window_starts
    usable_run_length = utils_legacy.usable_run_length
    straight_stretch_hours = utils_legacy.straight_stretch_hours
    straight_window_keep = utils_legacy.straight_window_keep
    calculate_regression_metrics = utils_legacy.calculate_regression_metrics
    convert_numpy = utils_legacy.convert_numpy
    get_data_paths = utils_legacy.get_data_paths
    ASTGCN_RECENT_LEN = utils_legacy.ASTGCN_RECENT_LEN
    ASTGCN_DAILY_LEN = utils_legacy.ASTGCN_DAILY_LEN
    ASTGCN_DAILY_INTERVAL = utils_legacy.ASTGCN_DAILY_INTERVAL

# 导入spatial模块
from . import spatial

__all__ = [
    'Logger',
    'random_seed',
    'get_n_feature',
    'CreateDataset',
    'EV_VLM_Dataset',
    'ASTGCNDataset',
    'create_rnn_data',
    'split_window_starts',
    'usable_run_length',
    'straight_stretch_hours',
    'straight_window_keep',
    'calculate_regression_metrics',
    'convert_numpy',
    'get_data_paths',
    'ASTGCN_RECENT_LEN',
    'ASTGCN_DAILY_LEN',
    'ASTGCN_DAILY_INTERVAL',
    'spatial',
]

