#!/usr/bin/env python3
"""
模块映射生成器 - 扫描所有模块，构建属性到模块的映射表。
运行: python generate_module_map.py
"""

import ast
import os
import sys
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
import json

class ModuleScanner:
    """模块扫描器，分析所有模块的结构"""
    
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.lib_dir = root_dir / "lib"
        self.module_map: Dict[str, List[Tuple[str, str]]] = {}  # attr_name -> [(module_path, attr_name), ...]
        self.processed_modules: Set[str] = set()
        
    def scan_module(self, module_path: str) -> Dict[str, Any]:
        """扫描单个模块，返回其所有公开属性"""
        try:
            # 动态导入模块
            module = importlib.import_module(module_path)
            
            # 获取模块的所有属性
            attrs = {}
            
            # 检查 __all__ 定义的导出
            if hasattr(module, '__all__'):
                for attr_name in module.__all__:
                    if hasattr(module, attr_name):
                        attrs[attr_name] = getattr(module, attr_name)
            
            # 如果没有 __all__，获取所有非私有的属性
            else:
                for attr_name in dir(module):
                    if not attr_name.startswith('_'):
                        attrs[attr_name] = getattr(module, attr_name)
            
            # 记录属性类型
            result = {}
            for attr_name, attr_value in attrs.items():
                # 获取属性的类型
                if inspect.ismodule(attr_value):
                    attr_type = 'module'
                elif inspect.isclass(attr_value):
                    attr_type = 'class'
                elif inspect.isfunction(attr_value) or inspect.ismethod(attr_value):
                    attr_type = 'function'
                elif inspect.isbuiltin(attr_value):
                    attr_type = 'builtin'
                else:
                    attr_type = 'variable'
                
                # 获取定义模块
                if hasattr(attr_value, '__module__'):
                    defined_in = attr_value.__module__
                else:
                    defined_in = module_path
                
                result[attr_name] = {
                    'type': attr_type,
                    'defined_in': defined_in,
                    'module_path': module_path,
                }
            
            return result
            
        except Exception as e:
            print(f"⚠ 警告: 扫描模块 {module_path} 失败: {e}")
            return {}
    
    def scan_directory(self, directory: Path, base_module: str = "") -> None:
        """递归扫描目录中的所有模块"""
        for item in directory.iterdir():
            # 跳过隐藏文件和特殊目录
            if item.name.startswith('.') or item.name == '__pycache__':
                continue
            
            # 如果是目录，递归扫描
            if item.is_dir():
                sub_module = f"{base_module}.{item.name}" if base_module else item.name
                self.scan_directory(item, sub_module)
            
            # 如果是Python文件
            elif item.suffix == '.py' and item.name != '__init__.py':
                # 转换为模块路径
                rel_path = item.relative_to(self.root_dir)
                module_path = str(rel_path).replace('/', '.').replace('.py', '')
                
                # 扫描模块
                if module_path not in self.processed_modules:
                    print(f"📦 扫描模块: {module_path}")
                    attrs = self.scan_module(module_path)
                    
                    # 更新映射表
                    for attr_name, attr_info in attrs.items():
                        if attr_name not in self.module_map:
                            self.module_map[attr_name] = []
                        
                        self.module_map[attr_name].append({
                            'module_path': attr_info['module_path'],
                            'defined_in': attr_info['defined_in'],
                            'type': attr_info['type']
                        })
                    
                    self.processed_modules.add(module_path)
    
    def scan_init_files(self) -> None:
        """扫描所有 __init__.py 文件中的导出"""
        for init_file in self.lib_dir.rglob("__init__.py"):
            # 跳过根目录的 __init__.py
            if init_file.parent == self.lib_dir:
                continue
            
            # 计算模块路径
            rel_path = init_file.relative_to(self.root_dir).parent
            module_path = str(rel_path).replace('/', '.')
            
            if module_path not in self.processed_modules:
                print(f"📦 扫描包: {module_path}")
                attrs = self.scan_module(module_path)
                
                # 更新映射表
                for attr_name, attr_info in attrs.items():
                    if attr_name not in self.module_map:
                        self.module_map[attr_name] = []
                    
                    self.module_map[attr_name].append({
                        'module_path': attr_info['module_path'],
                        'defined_in': attr_info['defined_in'],
                        'type': attr_info['type']
                    })
                
                self.processed_modules.add(module_path)
    
    def generate_module_map(self) -> Dict[str, Any]:
        """生成完整的模块映射表"""
        print("🚀 开始扫描模块...")
        
        # 1. 扫描所有包
        self.scan_init_files()
        
        # 2. 扫描所有独立模块
        self.scan_directory(self.lib_dir)
        
        # 3. 去重和排序
        for attr_name in self.module_map:
            # 去重：相同模块路径的只保留一个
            unique_items = {}
            for item in self.module_map[attr_name]:
                key = item['module_path']
                if key not in unique_items:
                    unique_items[key] = item
            
            self.module_map[attr_name] = list(unique_items.values())
            
            # 排序：类 > 函数 > 模块 > 变量
            type_priority = {'class': 0, 'function': 1, 'module': 2, 'builtin': 3, 'variable': 4}
            self.module_map[attr_name].sort(key=lambda x: type_priority.get(x['type'], 99))
        
        # 4. 生成统计数据
        stats = {
            'total_attributes': len(self.module_map),
            'total_modules': len(self.processed_modules),
            'attribute_types': {},
        }
        
        for attr_name, items in self.module_map.items():
            for item in items:
                attr_type = item['type']
                stats['attribute_types'][attr_type] = stats['attribute_types'].get(attr_type, 0) + 1
        
        return {
            'module_map': self.module_map,
            'stats': stats,
            'version': '1.0.0',
            'generated_at': importlib.import_module('datetime').datetime.now().isoformat()
        }
    
    def save_to_pickle(self, output_path: Path) -> None:
        """保存为 pickle 文件"""
        import pickle
        
        data = self.generate_module_map()
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✅ 模块映射已保存到: {output_path}")
        print(f"📊 统计信息:")
        print(f"   - 总属性数: {data['stats']['total_attributes']}")
        print(f"   - 总模块数: {data['stats']['total_modules']}")
        for attr_type, count in data['stats']['attribute_types'].items():
            print(f"   - {attr_type}: {count}")


    def generate_simple_map(self) -> Dict[str, str]:
        """生成简化的属性->模块映射"""
        simple_map = {}
        for attr_name, items in self.module_map.items():
            if items:
                # 取第一个项目（已经按优先级排序）
                item = items[0]
                # 记录模块路径和属性名
                simple_map[attr_name] = {
                    'module': item['module_path'],
                    'type': item['type'],
                    'defined_in': item.get('defined_in', '')
                }
        return simple_map
    
    def save_all_formats(self, output_dir: Path):
        """保存所有格式的映射表"""
        import pickle
        import json
        
        output_dir.mkdir(exist_ok=True)
        
        # 1. 保存完整版（pickle）
        full_data = self.generate_module_map()
        pickle_file = output_dir / "_module_map_full.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(full_data, f)
        
        # 2. 保存简化版（JSON）
        simple_map = self.generate_simple_map()
        json_file = output_dir / "_module_map.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'attributes': simple_map,
                'version': '1.0.0',
                'generated_at': importlib.import_module('datetime').datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)
        
        # 3. 保存快速查找版（pickle）
        quick_lookup = {k: v['module'] for k, v in simple_map.items()}
        quick_file = output_dir / "_module_map_quick.pkl"
        with open(quick_file, 'wb') as f:
            pickle.dump(quick_lookup, f)
        
        return pickle_file, json_file, quick_file


def main():
    # 获取项目根目录
    root_dir = Path(__file__).parent
    
    # 创建扫描器
    scanner = ModuleScanner(root_dir)
    scanner.save_all_formats(Path('.'))
    

if __name__ == '__main__':
    main()