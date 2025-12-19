# ./lib/__init__.py
"""
vvdutils 主模块。
采用懒加载机制，仅在首次访问时导入子模块。
"""

from .loader import LazyLoader

# 定义懒加载映射表 - 包含所有主要模块
LAZY_IMPORTS = {
    # 数据管理类
    'DataManager': ('.data', 'DataManager'),
    'DatamanagerBuilder': ('.data', 'DatamanagerBuilder'),
    
    # 工具模块
    'utils': ('.utils', None),
    
    # 数据处理模块
    'data': ('.data', None),
    
    # 标签处理模块
    'labelme': ('.labelme', None),
    
    # 处理模块
    'processing': ('.processing', None),
    
    # 数据库模块
    'database': ('.database', None),
    
    # 曲线生成器
    'curves_generator': ('.curves_generator', None),
    
    # 工具模块
    'tools': ('.tools', None),

    # 通信模块
    'communication': ('.communication', None),
    
    # 评估器模块
    'evaluator': ('.evaluator', None)
}

# 立即导入的内容（轻量级）
IMMEDIATE_IMPORTS = {}

# 创建懒加载器实例
lazy_loader = LazyLoader(
    module_name=__name__,
    lazy_imports=LAZY_IMPORTS,
    immediate_imports=IMMEDIATE_IMPORTS
)

# 设置模块的特殊方法
__getattr__ = lazy_loader.getattr
__dir__ = lazy_loader.dir

# 定义公开的接口
__all__ = list(LAZY_IMPORTS.keys())
