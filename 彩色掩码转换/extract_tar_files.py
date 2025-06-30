#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量解压缩.tar文件脚本
遍历指定路径下所有子文件夹中的.tar文件，并解压到当前位置
"""

import os
import tarfile
import sys
from pathlib import Path
import time


def extract_tar_files(base_path):
    """
    批量解压缩指定路径下的所有.tar文件
    
    Args:
        base_path (str): 要搜索的基础路径
    """
    print("="*60)
    print("🔧 批量.tar文件解压缩工具")
    print("="*60)
    print(f"📁 目标路径: {base_path}")
    print()
    
    # 检查路径是否存在
    if not os.path.exists(base_path):
        print(f"❌ 错误：目标路径不存在！")
        print(f"路径: {base_path}")
        return False
    
    # 计数器
    total_files = 0
    success_count = 0
    error_count = 0
    start_time = time.time()
    
    # 收集所有.tar文件
    tar_files = []
    print("🔍 正在搜索.tar文件...")
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith('.tar'):
                tar_files.append(os.path.join(root, file))
    
    total_files = len(tar_files)
    
    if total_files == 0:
        print("⚠️  未找到.tar文件！")
        print("请检查路径是否正确。")
        return True
    
    print(f"📦 找到 {total_files} 个.tar文件")
    print()
    
    # 逐个处理.tar文件
    for i, tar_path in enumerate(tar_files, 1):
        print(f"[{i}/{total_files}] 正在处理: {os.path.basename(tar_path)}")
        print(f"📂 位置: {os.path.dirname(tar_path)}")
        
        try:
            # 获取文件所在目录
            extract_dir = os.path.dirname(tar_path)
            
            # 打开并解压tar文件
            with tarfile.open(tar_path, 'r') as tar:
                # 获取文件列表
                members = tar.getmembers()
                print(f"📄 包含 {len(members)} 个文件/文件夹")
                
                # 解压到当前目录
                tar.extractall(path=extract_dir)
                
            print(f"✅ 解压成功！")
            success_count += 1
            
        except tarfile.TarError as e:
            print(f"❌ Tar文件错误: {str(e)}")
            error_count += 1
        except PermissionError as e:
            print(f"❌ 权限错误: {str(e)}")
            error_count += 1
        except Exception as e:
            print(f"❌ 未知错误: {str(e)}")
            error_count += 1
        
        print("-" * 50)
    
    # 显示最终统计
    end_time = time.time()
    duration = end_time - start_time
    
    print()
    print("="*60)
    print("📊 解压缩完成统计")
    print("="*60)
    print(f"⏱️  总耗时: {duration:.2f} 秒")
    print(f"📦 总文件数: {total_files}")
    print(f"✅ 成功解压: {success_count}")
    print(f"❌ 解压失败: {error_count}")
    
    if error_count == 0:
        print("🎉 所有文件解压成功！")
    else:
        print(f"⚠️  有 {error_count} 个文件解压失败，请检查错误信息")
    
    return error_count == 0


def main():
    """主函数"""
    # 目标路径
    base_path = r"C:\Users\root\OneDrive\Desktop\像素级标注-2-标签修正后\分割汇总-after"
    
    try:
        success = extract_tar_files(base_path)
        
        if success:
            print("\n✨ 程序执行完成！")
        else:
            print("\n💥 程序执行过程中遇到问题！")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 程序异常退出: {str(e)}")
        sys.exit(1)
    
    input("\n按Enter键退出...")


if __name__ == "__main__":
    main() 