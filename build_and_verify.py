#!/usr/bin/env python3
"""
构建并验证模块映射表
运行: python build_and_verify.py
"""

import subprocess
import sys
import importlib
from pathlib import Path

def build_module_map():
    """构建模块映射表"""
    print("🔨 构建模块映射表...")
    result = subprocess.run([sys.executable, "generate_module_map.py"], 
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ 构建失败: {result.stderr}")
        return False
    
    print(result.stdout)
    return True

def test_imports():
    """测试导入关键模块"""
    test_cases = [
        ("MongoGridFSConnection", "lib.database.mongofs.connect"),
        ("MysqlConnection", "lib.database.mysql.connect"),
        ("DataManager", "lib.data.base"),
        ("ClassifierEvalBinary", "lib.utils.classifier_eval.eval_metrics"),
        ("Registry", "lib.utils.register"),
    ]
    
    print("🧪 测试导入...")
    all_passed = True
    
    for attr_name, expected_module in test_cases:
        try:
            # 测试直接导入
            module = importlib.import_module(f".{expected_module}", package="vvdutils")
            if hasattr(module, attr_name):
                print(f"✅ {attr_name} -> {expected_module}")
            else:
                print(f"❌ {attr_name} 不在 {expected_module} 中")
                all_passed = False
        except Exception as e:
            print(f"❌ 导入失败 {attr_name}: {e}")
            all_passed = False
    
    return all_passed

def test_lazy_loading():
    """测试懒加载"""
    print("🚀 测试懒加载...")
    
    # 临时修改 sys.path 以便导入
    import sys
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    try:
        # 测试顶级导入
        import vvdutils as vv
        
        # 测试各种属性访问
        test_attrs = [
            "MongoGridFSConnection",
            "DataManager",
            "ClassifierEvalBinary",
            "Registry",
            "json_save",  # 来自 utils
            "dir_check",  # 来自 utils
        ]
        
        for attr_name in test_attrs:
            try:
                value = getattr(vv, attr_name)
                print(f"✅ 成功懒加载: {attr_name} ({type(value).__name__})")
            except AttributeError as e:
                print(f"❌ 懒加载失败: {attr_name} - {e}")
        
        # 测试实际使用
        print("\n🔧 测试实际使用...")
        try:
            # 测试创建 DataManager
            from vvdutils import DataManager
            print("✅ DataManager 导入成功")
            
            # 测试常用工具函数
            from vvdutils import dir_check
            print("✅ dir_check 导入成功")
            
        except Exception as e:
            print(f"❌ 使用测试失败: {e}")
            
    finally:
        # 清理
        sys.path.pop(0)

if __name__ == '__main__':
    # 1. 构建映射表
    if not build_module_map():
        sys.exit(1)
    
    # 2. 测试直接导入
    if not test_imports():
        print("⚠ 直接导入测试有失败，但懒加载可能仍能工作")
    
    # 3. 测试懒加载
    test_lazy_loading()
    
    print("\n🎉 所有测试完成！")