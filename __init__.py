# ./__init__.py
"""
vvdutils 主包。
采用智能懒加载机制，根据模块映射表动态导入。
"""

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    # 类型检查时，直接导入所有内容
    from .lib import *
    from .lib.data import *
    from .lib.labelme import *
    from .lib.processing import *
    from .lib.utils import *
    from .lib.communication import *
    from .lib.curves_generator import *
    from .lib.database import *
    from .lib.evaluator import *
    from .lib.tools import *

    from .lib.data import DataManager, DatamanagerBuilder
    from .lib import (
        utils,
        data,
        labelme,
        processing,
        database,
        curves_generator,
        tools,
        communication,
        evaluator
    )
    # 为顶层模块定义类型
    data = data
    utils = utils
    labelme = labelme
    processing = processing
    database = database
    curves_generator = curves_generator
    tools = tools
    communication = communication
    evaluator = evaluator
    DataManager = DataManager
    DatamanagerBuilder = DatamanagerBuilder   


# 导入 lib 模块
from . import lib

# 首先，设置我们自己的属性
__all__ = ['lib']
__version__ = '0.1.0'

# 缓存
_cache = {
    'lib': lib,
    '__version__': __version__,
}

# 模块映射表缓存
_module_map_cache = None
_attribute_to_module = {}  # 属性名 -> 模块路径映射

def _load_module_map():
    """加载模块映射表"""
    global _module_map_cache, _attribute_to_module
    
    if _module_map_cache is not None:
        return _module_map_cache
    
    # 尝试加载映射表
    map_files = [
        Path(__file__).parent / "_module_map.pkl",
        Path(__file__).parent / "_module_map.json",
    ]
    
    for map_file in map_files:
        if map_file.exists():
            try:
                if map_file.suffix == '.pkl':
                    with open(map_file, 'rb') as f:
                        _module_map_cache = pickle.load(f)
                else:  # .json
                    import json
                    with open(map_file, 'r', encoding='utf-8') as f:
                        _module_map_cache = json.load(f)
                
                print(f"✅ 已加载模块映射表: {map_file}")
                
                # 构建属性到模块的映射
                if 'attributes' in _module_map_cache:
                    # JSON 格式
                    for attr_name, modules in _module_map_cache['attributes'].items():
                        if modules:  # 可能有多个模块定义相同属性，取第一个
                            module_info = modules[0]
                            if isinstance(module_info, dict):
                                _attribute_to_module[attr_name] = module_info.get('module')
                            else:
                                _attribute_to_module[attr_name] = module_info
                elif 'module_map' in _module_map_cache:
                    # Pickle 格式
                    for attr_name, items in _module_map_cache['module_map'].items():
                        if items:
                            _attribute_to_module[attr_name] = items[0].get('module_path')
                
                # 打印统计信息
                if 'stats' in _module_map_cache:
                    stats = _module_map_cache['stats']
                    print(f"📊 映射表统计:")
                    print(f"   - 总属性数: {stats.get('total_attributes', 'N/A')}")
                    print(f"   - 总模块数: {stats.get('total_modules', 'N/A')}")
                
                return _module_map_cache
            except Exception as e:
                print(f"⚠ 加载模块映射表失败 {map_file}: {e}")
    
    print("⚠ 未找到模块映射表，将使用传统懒加载机制")
    _module_map_cache = {}
    return _module_map_cache

# 初始化时加载映射表
_load_module_map()

def _import_from_module(module_path: str, attr_name: Optional[str] = None) -> Any:
    """从指定模块导入属性"""
    try:
        # 动态导入模块
        import importlib
        
        # 如果是相对路径，转换为绝对路径
        if module_path.startswith('.'):
            # 计算相对于当前包的路径
            parts = module_path.split('.')
            if parts[0] == '':
                parts = parts[1:]
            # 从当前包开始构建路径
            full_path = __name__
            for part in parts:
                if part:
                    full_path += '.' + part
            module = importlib.import_module(full_path)
        else:
            module = importlib.import_module(module_path)
        
        # 如果指定了属性名，返回属性，否则返回模块
        if attr_name:
            return getattr(module, attr_name)
        return module
    except Exception as e:
        print(f"⚠ 导入失败 {module_path}.{attr_name if attr_name else ''}: {e}")
        return None

def _smart_getattr(name: str) -> Any:
    """智能获取属性：先查缓存，再查映射表，最后回退到lib"""
    # 1. 检查缓存
    if name in _cache:
        return _cache[name]
    
    # 2. 检查模块映射表
    if name in _attribute_to_module:
        module_path = _attribute_to_module[name]
        if module_path:
            value = _import_from_module(module_path, name)
            if value is not None:
                _cache[name] = value
                return value
    
    # 3. 尝试从 lib 模块获取
    try:
        # 先尝试从 lib 直接获取
        if hasattr(lib, name):
            value = getattr(lib, name)
            _cache[name] = value
            return value
        
        # 尝试从 lib 的子模块获取
        # 比如 MongoGridFSConnection 在 lib.database.mongofs
        # 我们可以尝试解析名称
        if name.endswith('Connection'):
            # 可能是数据库连接类
            if name.startswith('Mongo'):
                try:
                    from .lib.database.mongofs import MongoGridFSConnection
                    _cache[name] = MongoGridFSConnection
                    return MongoGridFSConnection
                except ImportError:
                    pass
            elif name.startswith('Mysql'):
                try:
                    from .lib.database.mysql import MysqlConnection
                    _cache[name] = MysqlConnection
                    return MysqlConnection
                except ImportError:
                    pass
        
        # 尝试常见的模块模式
        common_prefixes = [
            ('DataManager', '.lib.data'),
            ('ClassifierEval', '.lib.utils.classifier_eval'),
            ('Registry', '.lib.utils'),
        ]
        
        for prefix, module_base in common_prefixes:
            if name.startswith(prefix):
                try:
                    value = _import_from_module(module_base, name)
                    if value is not None:
                        _cache[name] = value
                        return value
                except:
                    pass
    except Exception as e:
        print(f"⚠ 从lib获取属性失败 {name}: {e}")
    
    # 4. 最后尝试从 utils 模块获取（原有逻辑）
    try:
        if not hasattr(lib, 'utils'):
            # 触发 utils 模块的懒加载
            lib.utils
        
        utils_module = lib.utils
        if hasattr(utils_module, name):
            func = getattr(utils_module, name)
            _cache[name] = func
            return func
    except:
        pass
    
    # 5. 抛出 AttributeError
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def __getattr__(name: str) -> Any:
    """模块级别的 __getattr__"""
    return _smart_getattr(name)

def __dir__():
    """返回所有可用的属性"""
    attrs = set(_cache.keys())
    
    # 添加映射表中的属性
    attrs.update(_attribute_to_module.keys())
    
    # 添加 lib 模块的所有属性
    attrs.update(dir(lib))
    
    # 添加标准模块属性
    attrs.update([
        '__all__', '__doc__', '__getattr__', '__dir__', 
        '__name__', '__package__', '__file__', '__path__',
        '__version__'
    ])
    
    return sorted(attrs)

# 预加载一些常用属性，提高访问速度
def _preload_common_attributes():
    """预加载常用属性"""
    common_attrs = [
        'DataManager',
        'DatamanagerBuilder',
        'MongoGridFSConnection',
        'MysqlConnection',
        'ClassifierEvalBinary',
        'ClassifierEvalMulticlass',
        'ClassifierEvalMultilabel',
        'Registry',
    ]
    
    for attr in common_attrs:
        if attr in _attribute_to_module and attr not in _cache:
            try:
                _smart_getattr(attr)
            except:
                pass

# 可选：在模块导入时预加载常用属性（会增加启动时间）
# _preload_common_attributes()