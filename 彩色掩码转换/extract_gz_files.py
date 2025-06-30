#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量解压缩.gz文件脚本
遍历指定路径下所有子文件夹中的.gz文件，并解压到当前位置
"""

import os
import gzip
import shutil
import sys
from pathlib import Path
import time


def extract_gz_files(base_path):
    """
    批量解压缩指定路径下的所有.gz文件
    
    Args:
        base_path (str): 要搜索的基础路径
    """
    print("="*60)
    print("🔧 批量.gz文件解压缩工具")
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
    
    # 收集所有.gz文件
    gz_files = []
    print("🔍 正在搜索.gz文件...")
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith('.gz'):
                gz_files.append(os.path.join(root, file))
    
    total_files = len(gz_files)
    
    if total_files == 0:
        print("⚠️  未找到.gz文件！")
        print("请检查路径是否正确。")
        return True
    
    print(f"📦 找到 {total_files} 个.gz文件")
    print()
    
    # 逐个处理.gz文件
    for i, gz_path in enumerate(gz_files, 1):
        print(f"[{i}/{total_files}] 正在处理: {os.path.basename(gz_path)}")
        print(f"📂 位置: {os.path.dirname(gz_path)}")
        
        try:
            # 获取文件所在目录
            extract_dir = os.path.dirname(gz_path)
            
            # 确定解压后的文件名（去掉.gz后缀）
            if gz_path.lower().endswith('.gz'):
                output_filename = os.path.basename(gz_path)[:-3]  # 去掉.gz后缀
            else:
                output_filename = os.path.basename(gz_path) + '.decompressed'
            
            output_path = os.path.join(extract_dir, output_filename)
            
            # 检查输出文件是否已存在
            if os.path.exists(output_path):
                print(f"⚠️  输出文件已存在: {output_filename}")
                
                # 生成新的文件名
                base_name, ext = os.path.splitext(output_filename)
                counter = 1
                while os.path.exists(output_path):
                    new_name = f"{base_name}_{counter}{ext}"
                    output_path = os.path.join(extract_dir, new_name)
                    counter += 1
                
                print(f"🔄 重命名为: {os.path.basename(output_path)}")
            
            # 获取原文件大小
            original_size = os.path.getsize(gz_path)
            print(f"📏 原文件大小: {format_size(original_size)}")
            
            # 解压缩文件
            with gzip.open(gz_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # 获取解压后文件大小
            decompressed_size = os.path.getsize(output_path)
            compression_ratio = (1 - original_size / decompressed_size) * 100 if decompressed_size > 0 else 0
            
            print(f"📄 解压后大小: {format_size(decompressed_size)}")
            print(f"📊 压缩率: {compression_ratio:.1f}%")
            print(f"✅ 解压成功！输出文件: {output_filename}")
            success_count += 1
            
        except gzip.BadGzipFile as e:
            print(f"❌ 不是有效的gzip文件: {str(e)}")
            error_count += 1
        except PermissionError as e:
            print(f"❌ 权限错误: {str(e)}")
            error_count += 1
        except OSError as e:
            print(f"❌ 系统错误: {str(e)}")
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


def format_size(size_bytes):
    """
    格式化文件大小显示
    
    Args:
        size_bytes (int): 字节数
        
    Returns:
        str: 格式化后的大小字符串
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.2f} {size_names[i]}"


def main():
    """主函数"""
    # 目标路径
    base_path = r"C:\Users\root\OneDrive\Desktop\像素级标注-2-标签修正后\分割汇总-after"
    
    try:
        success = extract_gz_files(base_path)
        
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