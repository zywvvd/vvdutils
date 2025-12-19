# ./lib/data/__init__.py
"""
数据管理模块。
"""

from .class_mapper import DataManagerClassManipulator as DataManager
from .builder import DatamanagerBuilder

# 明确导出
__all__ = ['DataManager', 'DatamanagerBuilder']

# 对于 VSCode 类型提示
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # 类型检查时，导入所有内容
    from .class_mapper import DataManagerClassManipulator as DataManager
    from .builder import DatamanagerBuilder
    from . import *