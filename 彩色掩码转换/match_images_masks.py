#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像与掩码匹配验证脚本
验证images和masks文件夹中的文件是否一一对应匹配
"""

import os
import sys
import shutil
from pathlib import Path
import time
from collections import defaultdict


def scan_directory(directory, extensions):
    """
    扫描指定目录，查找指定扩展名的文件
    
    Args:
        directory (str): 目录路径
        extensions (list): 文件扩展名列表
        
    Returns:
        dict: {基础文件名: 完整文件路径}
    """
    files_dict = {}
    
    if not os.path.exists(directory):
        print(f"⚠️  目录不存在: {directory}")
        return files_dict
    
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext.lower()) for ext in extensions):
            # 获取基础文件名（去掉扩展名）
            base_name = os.path.splitext(file)[0]
            full_path = os.path.join(directory, file)
            files_dict[base_name] = full_path
    
    return files_dict


def analyze_matching(images_dict, masks_dict):
    """
    分析图像和掩码的匹配情况
    
    Args:
        images_dict (dict): 图像文件字典
        masks_dict (dict): 掩码文件字典
        
    Returns:
        tuple: (matched_pairs, orphan_images, orphan_masks)
    """
    # 获取所有文件的基础名称
    image_names = set(images_dict.keys())
    mask_names = set(masks_dict.keys())
    
    # 找出匹配的对
    matched_names = image_names & mask_names
    matched_pairs = [(images_dict[name], masks_dict[name]) for name in matched_names]
    
    # 找出孤儿文件（没有对应的文件）
    orphan_images = {name: images_dict[name] for name in image_names - mask_names}
    orphan_masks = {name: masks_dict[name] for name in mask_names - image_names}
    
    return matched_pairs, orphan_images, orphan_masks


def display_matching_report(matched_pairs, orphan_images, orphan_masks, images_dir, masks_dir):
    """
    显示匹配分析报告
    
    Args:
        matched_pairs (list): 匹配的文件对列表
        orphan_images (dict): 孤儿图像文件
        orphan_masks (dict): 孤儿掩码文件
        images_dir (str): 图像目录
        masks_dir (str): 掩码目录
    """
    print("="*80)
    print("📊 图像与掩码匹配分析报告")
    print("="*80)
    print(f"📁 图像目录: {images_dir}")
    print(f"📁 掩码目录: {masks_dir}")
    print()
    
    # 基本统计
    total_images = len(matched_pairs) + len(orphan_images)
    total_masks = len(matched_pairs) + len(orphan_masks)
    
    print("📈 基本统计:")
    print(f"   📸 图像文件总数: {total_images}")
    print(f"   🎨 掩码文件总数: {total_masks}")
    print(f"   ✅ 成功匹配对数: {len(matched_pairs)}")
    print(f"   🔍 匹配率: {len(matched_pairs)/max(total_images, total_masks)*100:.1f}%")
    print()
    
    # 匹配成功的文件
    if matched_pairs:
        print(f"✅ 成功匹配的文件对 ({len(matched_pairs)} 对):")
        for i, (img_path, mask_path) in enumerate(matched_pairs[:10], 1):
            img_name = os.path.basename(img_path)
            mask_name = os.path.basename(mask_path)
            img_size = format_size(os.path.getsize(img_path))
            mask_size = format_size(os.path.getsize(mask_path))
            print(f"   [{i:3d}] {img_name} ({img_size}) ↔ {mask_name} ({mask_size})")
        
        if len(matched_pairs) > 10:
            print(f"   ... 还有 {len(matched_pairs) - 10} 对匹配的文件")
        print()
    
    # 孤儿图像文件（有图像无掩码）
    if orphan_images:
        print(f"🖼️  孤儿图像文件 - 有图像但无对应掩码 ({len(orphan_images)} 个):")
        for i, (base_name, img_path) in enumerate(orphan_images.items(), 1):
            img_name = os.path.basename(img_path)
            img_size = format_size(os.path.getsize(img_path))
            print(f"   [{i:3d}] {img_name} ({img_size})")
            if i >= 10:
                print(f"   ... 还有 {len(orphan_images) - 10} 个孤儿图像")
                break
        print()
    
    # 孤儿掩码文件（有掩码无图像）
    if orphan_masks:
        print(f"🎭 孤儿掩码文件 - 有掩码但无对应图像 ({len(orphan_masks)} 个):")
        for i, (base_name, mask_path) in enumerate(orphan_masks.items(), 1):
            mask_name = os.path.basename(mask_path)
            mask_size = format_size(os.path.getsize(mask_path))
            print(f"   [{i:3d}] {mask_name} ({mask_size})")
            if i >= 10:
                print(f"   ... 还有 {len(orphan_masks) - 10} 个孤儿掩码")
                break
        print()
    
    # 匹配质量评估
    print("📋 匹配质量评估:")
    if not orphan_images and not orphan_masks:
        print("   🎉 完美匹配！所有图像和掩码都成功配对")
    elif len(orphan_images) == 0:
        print("   ✅ 所有图像都有对应掩码")
        print(f"   ⚠️  但有 {len(orphan_masks)} 个多余的掩码文件")
    elif len(orphan_masks) == 0:
        print("   ✅ 所有掩码都有对应图像")
        print(f"   ⚠️  但有 {len(orphan_images)} 个缺少掩码的图像")
    else:
        print(f"   ⚠️  部分匹配：{len(orphan_images)} 个图像缺少掩码，{len(orphan_masks)} 个掩码缺少图像")


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


def clean_orphan_files(orphan_images, orphan_masks, action='move'):
    """
    清理孤儿文件
    
    Args:
        orphan_images (dict): 孤儿图像文件
        orphan_masks (dict): 孤儿掩码文件
        action (str): 操作类型 'move' 或 'delete'
    """
    if not orphan_images and not orphan_masks:
        print("✅ 没有孤儿文件需要清理")
        return
    
    print("\n🧹 孤儿文件清理选项:")
    print("1. 移动到单独文件夹 (推荐)")
    print("2. 删除文件 (⚠️  危险操作)")
    print("3. 跳过清理")
    
    while True:
        choice = input("请选择操作 (1-3): ").strip()
        if choice in ['1', '2', '3']:
            break
        print("请输入有效选项 (1-3)")
    
    if choice == '3':
        print("🚫 跳过文件清理")
        return
    
    # 创建孤儿文件夹
    if choice == '1':
        base_dir = os.path.dirname(list(orphan_images.values())[0]) if orphan_images else os.path.dirname(list(orphan_masks.values())[0])
        parent_dir = os.path.dirname(base_dir)
        orphan_dir = os.path.join(parent_dir, "orphan_files")
        orphan_images_dir = os.path.join(orphan_dir, "images")
        orphan_masks_dir = os.path.join(orphan_dir, "masks")
        
        os.makedirs(orphan_images_dir, exist_ok=True)
        os.makedirs(orphan_masks_dir, exist_ok=True)
    
    moved_count = 0
    deleted_count = 0
    
    # 处理孤儿图像
    for base_name, img_path in orphan_images.items():
        try:
            if choice == '1':  # 移动
                dest_path = os.path.join(orphan_images_dir, os.path.basename(img_path))
                shutil.move(img_path, dest_path)
                print(f"📦 移动图像: {os.path.basename(img_path)} -> orphan_files/images/")
                moved_count += 1
            elif choice == '2':  # 删除
                os.remove(img_path)
                print(f"🗑️  删除图像: {os.path.basename(img_path)}")
                deleted_count += 1
        except Exception as e:
            print(f"❌ 处理失败 {os.path.basename(img_path)}: {str(e)}")
    
    # 处理孤儿掩码
    for base_name, mask_path in orphan_masks.items():
        try:
            if choice == '1':  # 移动
                dest_path = os.path.join(orphan_masks_dir, os.path.basename(mask_path))
                shutil.move(mask_path, dest_path)
                print(f"📦 移动掩码: {os.path.basename(mask_path)} -> orphan_files/masks/")
                moved_count += 1
            elif choice == '2':  # 删除
                os.remove(mask_path)
                print(f"🗑️  删除掩码: {os.path.basename(mask_path)}")
                deleted_count += 1
        except Exception as e:
            print(f"❌ 处理失败 {os.path.basename(mask_path)}: {str(e)}")
    
    # 显示结果
    if choice == '1':
        print(f"\n✅ 成功移动 {moved_count} 个孤儿文件到 orphan_files 文件夹")
    elif choice == '2':
        print(f"\n✅ 成功删除 {deleted_count} 个孤儿文件")


def export_matched_list(matched_pairs, output_file):
    """
    导出匹配的文件对列表
    
    Args:
        matched_pairs (list): 匹配的文件对
        output_file (str): 输出文件路径
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 图像与掩码匹配对列表\n")
            f.write("# 格式: 图像文件路径,掩码文件路径\n")
            f.write(f"# 总计: {len(matched_pairs)} 对\n")
            f.write("# 生成时间: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
            
            for img_path, mask_path in matched_pairs:
                f.write(f"{img_path},{mask_path}\n")
        
        print(f"📝 匹配列表已导出到: {output_file}")
        
    except Exception as e:
        print(f"❌ 导出失败: {str(e)}")


def main():
    """主函数"""
    # 配置路径
    images_dir = r"C:\Users\root\OneDrive\Desktop\9classes_lesion\images"
    masks_dir = r"C:\Users\root\OneDrive\Desktop\9classes_lesion\masks"
    
    # 支持的文件扩展名
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    mask_extensions = ['.png', '.PNG']
    
    print("="*80)
    print("🔍 图像与掩码匹配验证工具")
    print("="*80)
    print(f"📂 图像目录: {images_dir}")
    print(f"📂 掩码目录: {masks_dir}")
    print()
    
    try:
        start_time = time.time()
        
        # 扫描文件
        print("🔍 正在扫描文件...")
        images_dict = scan_directory(images_dir, image_extensions)
        masks_dict = scan_directory(masks_dir, mask_extensions)
        
        print(f"📸 发现图像文件: {len(images_dict)} 个")
        print(f"🎨 发现掩码文件: {len(masks_dict)} 个")
        print()
        
        # 分析匹配
        print("📊 正在分析匹配情况...")
        matched_pairs, orphan_images, orphan_masks = analyze_matching(images_dict, masks_dict)
        
        # 显示报告
        display_matching_report(matched_pairs, orphan_images, orphan_masks, images_dir, masks_dir)
        
        # 询问是否导出匹配列表
        if matched_pairs:
            print("\n💾 导出选项:")
            export_choice = input("是否导出匹配文件对列表？(y/N): ").strip().lower()
            if export_choice in ['y', 'yes', '是']:
                output_file = "matched_pairs_list.txt"
                export_matched_list(matched_pairs, output_file)
        
        # 询问是否清理孤儿文件
        if orphan_images or orphan_masks:
            print("\n🧹 清理选项:")
            clean_choice = input("是否处理孤儿文件？(y/N): ").strip().lower()
            if clean_choice in ['y', 'yes', '是']:
                clean_orphan_files(orphan_images, orphan_masks)
        
        # 最终总结
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n⏱️  分析耗时: {duration:.2f} 秒")
        print("✨ 匹配验证完成！")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 程序异常退出: {str(e)}")
        sys.exit(1)
    
    input("\n按Enter键退出...")


if __name__ == "__main__":
    main() 