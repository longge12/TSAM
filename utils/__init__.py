# Utils package
# 注意：这个包与根目录的 utils.py 冲突
# 为了解决导入问题，我们从根目录的 utils.py 导入函数

import sys
import os
import importlib.util

# 获取根目录路径（父目录的父目录）
_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_utils_path = os.path.join(_root_dir, 'utils.py')

if os.path.exists(_utils_path):
    # 直接加载根目录的 utils.py 模块
    spec = importlib.util.spec_from_file_location("utils_root", _utils_path)
    utils_root = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils_root)
    
    # 导出函数
    calculate_rank = utils_root.calculate_rank
    metrics = utils_root.metrics
else:
    raise ImportError(f"Cannot find utils.py at {_utils_path}")

__all__ = ['calculate_rank', 'metrics']

