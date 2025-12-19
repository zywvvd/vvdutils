# ./lib/loader.py
"""
懒加载工具模块，可在多个包中复用。
"""

import sys
import importlib
from typing import Dict, Tuple, Optional, Any


def lazy_import(module_name):
    # print(f'Lazy import: {module_name}')
    return importlib.import_module(module_name)

def try_to_import(module_name, tutorial=None):
    """尝试导入模块，失败则返回 None"""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        print(f'Failed to import module: {module_name}')
        print(f"Please make sure the module is installed and available in your PYTHONPATH. ")
        if tutorial:
            print(f"For installation instructions, please refer to: {tutorial}")

        raise ImportError(f"Module '{module_name}' could not be imported.")

class LazyLoader:
    """
    通用的懒加载管理器
    
    示例:
        # 在 __init__.py 中
        from .loader import LazyLoader
        
        loader = LazyLoader(
            module_name=__name__,
            lazy_imports={
                'MyClass': ('mymodule', 'MyClass'),
                'my_module': ('subpackage', None),
            },
            immediate_imports={'my_function': some_function}
        )
        
        __getattr__ = loader.getattr
        __dir__ = loader.dir
    """
    
    def __init__(
        self,
        module_name: str,
        lazy_imports: Dict[str, Tuple[str, Optional[str]]],
        immediate_imports: Optional[Dict[str, Any]] = None
    ):
        """
        初始化懒加载器
        
        Args:
            module_name: 当前模块名称（通常是 __name__）
            lazy_imports: 懒加载映射表
            immediate_imports: 立即导入的映射表
        """
        self.module_name = module_name
        self.lazy_imports = lazy_imports
        self.immediate_imports = immediate_imports or {}
        
        # 初始化全局变量字典
        self._globals = {}
        
        # 尝试获取模块的全局字典，如果模块还没有完全初始化，则使用空字典
        try:
            if module_name in sys.modules:
                self._globals = sys.modules[module_name].__dict__
        except (KeyError, AttributeError):
            pass
        
        # 添加立即导入的内容
        for name, value in self.immediate_imports.items():
            self._globals[name] = value
        
        # 在模块完全初始化后更新全局字典
        self._update_globals_after_init()
    
    def _update_globals_after_init(self):
        """在模块初始化完成后更新全局字典引用"""
        # 这个方法应该在模块初始化完成后调用
        # 但为了简单起见，我们在这里直接尝试更新
        try:
            if self.module_name in sys.modules:
                module_dict = sys.modules[self.module_name].__dict__
                # 将我们已经设置的值复制到模块字典中
                for key, value in self._globals.items():
                    module_dict[key] = value
                # 更新我们的引用
                self._globals = module_dict
        except (KeyError, AttributeError):
            pass
    def getattr(self, name: str) -> Any:
        """实现 __getattr__ 逻辑"""
        # 首先检查是否已经加载
        if name in self._globals:
            return self._globals[name]
        
        if name in self.lazy_imports:
            module_path, class_name = self.lazy_imports[name]
            
            try:
                # 计算包名
                if '.' in self.module_name:
                    package_name = self.module_name
                else:
                    package_name = self.module_name
                
                # 构建完整的模块路径
                if module_path.startswith('.'):
                    # 相对导入
                    if package_name:
                        # 处理多个点的情况
                        if module_path == '.':
                            full_module_path = package_name
                        elif module_path.startswith('..'):
                            # 处理上级目录
                            up_level = len(module_path.split('..')) - 1
                            package_parts = package_name.split('.')
                            if len(package_parts) > up_level:
                                base_package = '.'.join(package_parts[:-up_level])
                                full_module_path = f"{base_package}{module_path[up_level*2:]}"
                            else:
                                full_module_path = module_path[up_level*2:] or module_path[2:]
                        else:
                            # 处理单点相对导入
                            full_module_path = f"{package_name}{module_path}"
                    else:
                        full_module_path = module_path[1:]  # 去掉开头的点
                else:
                    # 绝对导入
                    full_module_path = module_path
                
                # 导入模块
                module = importlib.import_module(full_module_path)
                
                # 如果模块路径有点号，需要获取子模块
                if '.' in full_module_path:
                    parts = full_module_path.split('.')
                    current_module = sys.modules.get(parts[0])
                    if current_module:
                        for i in range(1, len(parts)):
                            if hasattr(current_module, parts[i]):
                                current_module = getattr(current_module, parts[i])
                            else:
                                # 尝试导入子模块
                                sub_module_name = '.'.join(parts[:i+1])
                                current_module = importlib.import_module(sub_module_name)
                        module = current_module
                
                # 获取目标值
                if class_name:
                    value = getattr(module, class_name)
                else:
                    value = module
                
                # 缓存到全局变量
                self._globals[name] = value
                
                # 同时更新 sys.modules 中的模块字典
                if self.module_name in sys.modules:
                    sys.modules[self.module_name].__dict__[name] = value
                
                return value
                
            except (ImportError, AttributeError) as e:
                raise AttributeError(
                    f"module '{self.module_name}' cannot load attribute '{name}': {e}"
                ) from e
        
        raise AttributeError(f"module '{self.module_name}' has no attribute '{name}'")
    
    def dir(self) -> list:
        """实现 __dir__ 逻辑"""
        attrs = set(self._globals.keys())
        attrs.update(self.lazy_imports.keys())
        # 添加标准模块属性
        attrs.update([
            '__all__', '__doc__', '__getattr__', '__dir__', 
            '__name__', '__package__', '__file__', '__path__'
        ])
        return sorted(attrs)
    
    def add_lazy_import(self, name: str, module_path: str, class_name: Optional[str] = None):
        """动态添加懒加载项"""
        self.lazy_imports[name] = (module_path, class_name)
    
    def remove_lazy_import(self, name: str):
        """移除懒加载项"""
        if name in self.lazy_imports:
            del self.lazy_imports[name]