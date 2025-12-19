import os

def merge_project_files(project_path=".", output_file="merged_project.txt", extensions=None, ignore_dirs=None):
    """
    将项目文件合并成一个文档
    
    Args:
        project_path: 项目根目录路径
        output_file: 输出文件路径
        extensions: 要包含的文件扩展名列表
        ignore_dirs: 要忽略的目录列表
    """
    if extensions is None:
        extensions = ['.py', '.txt', '.md', '.yaml', '.yml', '.json', '.cfg', '.ini', '.md', '.hpp', '.h', '.c', '.cpp', '.java', '.js', '.html', '.css', '.sdf']
    
    if ignore_dirs is None:
        ignore_dirs = ['__pycache__', '.git', 'venv', 'env', '.idea', '.vscode', 'build', 'dist', 'install', 'log']
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # 写入项目结构
        outfile.write("=" * 80 + "\n")
        outfile.write("PROJECT STRUCTURE\n")
        outfile.write("=" * 80 + "\n")
        
        for root, dirs, files in os.walk(project_path):
            # 移除要忽略的目录
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            level = root.replace(project_path, '').count(os.sep)
            indent = ' ' * 2 * level
            outfile.write(f"{indent}{os.path.basename(root)}/\n")
            
            sub_indent = ' ' * 2 * (level + 1)
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    outfile.write(f"{sub_indent}{file}\n")
        
        outfile.write("\n" + "=" * 80 + "\n")
        outfile.write("FILE CONTENTS\n")
        outfile.write("=" * 80 + "\n\n")
        
        # 写入文件内容
        for root, dirs, files in os.walk(project_path):
            # 移除要忽略的目录
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            for file in files:
                if file == output_file:
                    continue

                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            outfile.write(f"\n{'='*60}\n")
                            outfile.write(f"FILE: {file_path}\n")
                            outfile.write(f"{'='*60}\n\n")
                            outfile.write(infile.read())
                            outfile.write("\n\n")
                    except Exception as e:
                        outfile.write(f"Error reading {file_path}: {str(e)}\n\n")

merge_project_files()