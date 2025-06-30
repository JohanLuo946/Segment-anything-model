#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集划分脚本
将图像和掩码按照8:1:1的比例划分为train、val、test数据集
"""

import os
import sys
import shutil
import random
import time
from pathlib import Path
import json
from collections import defaultdict


def scan_matched_pairs(images_dir, masks_dir):
    """
    扫描并匹配图像和掩码文件对
    
    Args:
        images_dir (str): 图像目录路径
        masks_dir (str): 掩码目录路径
        
    Returns:
        list: [(image_path, mask_path), ...] 匹配的文件对列表
    """
    # 支持的文件扩展名
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    mask_extensions = ['.png', '.PNG']
    
    # 扫描图像文件
    images_dict = {}
    if os.path.exists(images_dir):
        for file in os.listdir(images_dir):
            if any(file.lower().endswith(ext.lower()) for ext in image_extensions):
                base_name = os.path.splitext(file)[0]
                full_path = os.path.join(images_dir, file)
                images_dict[base_name] = full_path
    
    # 扫描掩码文件
    masks_dict = {}
    if os.path.exists(masks_dir):
        for file in os.listdir(masks_dir):
            if any(file.lower().endswith(ext.lower()) for ext in mask_extensions):
                base_name = os.path.splitext(file)[0]
                full_path = os.path.join(masks_dir, file)
                masks_dict[base_name] = full_path
    
    # 找出匹配的对
    matched_pairs = []
    for base_name in images_dict.keys():
        if base_name in masks_dict:
            matched_pairs.append((images_dict[base_name], masks_dict[base_name]))
    
    return matched_pairs


def split_dataset(matched_pairs, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    按比例划分数据集
    
    Args:
        matched_pairs (list): 匹配的文件对列表
        train_ratio (float): 训练集比例
        val_ratio (float): 验证集比例  
        test_ratio (float): 测试集比例
        random_seed (int): 随机种子
        
    Returns:
        tuple: (train_pairs, val_pairs, test_pairs)
    """
    # 验证比例总和
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"⚠️  警告：比例总和不等于1.0 ({total_ratio})")
    
    # 设置随机种子确保可重现
    random.seed(random_seed)
    
    # 随机打乱
    pairs_copy = matched_pairs.copy()
    random.shuffle(pairs_copy)
    
    total_count = len(pairs_copy)
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)
    test_count = total_count - train_count - val_count  # 确保所有样本都被分配
    
    # 划分数据集
    train_pairs = pairs_copy[:train_count]
    val_pairs = pairs_copy[train_count:train_count + val_count]
    test_pairs = pairs_copy[train_count + val_count:]
    
    return train_pairs, val_pairs, test_pairs, (train_count, val_count, test_count)


def create_dataset_structure(output_dir):
    """
    创建数据集文件夹结构
    
    Args:
        output_dir (str): 输出根目录
        
    Returns:
        dict: 各个子目录的路径字典
    """
    dirs = {
        'train_images': os.path.join(output_dir, 'train', 'images'),
        'train_masks': os.path.join(output_dir, 'train', 'masks'),
        'val_images': os.path.join(output_dir, 'val', 'images'),
        'val_masks': os.path.join(output_dir, 'val', 'masks'),
        'test_images': os.path.join(output_dir, 'test', 'images'),
        'test_masks': os.path.join(output_dir, 'test', 'masks'),
    }
    
    # 创建所有目录
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 创建目录: {dir_path}")
    
    return dirs


def copy_files(file_pairs, target_images_dir, target_masks_dir, operation='copy'):
    """
    复制或移动文件到目标目录
    
    Args:
        file_pairs (list): 文件对列表
        target_images_dir (str): 目标图像目录
        target_masks_dir (str): 目标掩码目录
        operation (str): 'copy' 或 'move'
        
    Returns:
        tuple: (success_count, error_count)
    """
    success_count = 0
    error_count = 0
    
    for img_path, mask_path in file_pairs:
        try:
            # 获取文件名
            img_name = os.path.basename(img_path)
            mask_name = os.path.basename(mask_path)
            
            # 目标路径
            target_img_path = os.path.join(target_images_dir, img_name)
            target_mask_path = os.path.join(target_masks_dir, mask_name)
            
            # 复制或移动文件
            if operation == 'copy':
                shutil.copy2(img_path, target_img_path)
                shutil.copy2(mask_path, target_mask_path)
            elif operation == 'move':
                shutil.move(img_path, target_img_path)
                shutil.move(mask_path, target_mask_path)
            
            success_count += 1
            
        except Exception as e:
            print(f"❌ 处理失败 {os.path.basename(img_path)}: {str(e)}")
            error_count += 1
    
    return success_count, error_count


def save_split_info(output_dir, train_pairs, val_pairs, test_pairs, split_config):
    """
    保存数据集划分信息
    
    Args:
        output_dir (str): 输出目录
        train_pairs (list): 训练集文件对
        val_pairs (list): 验证集文件对
        test_pairs (list): 测试集文件对
        split_config (dict): 划分配置信息
    """
    split_info = {
        'split_time': time.strftime("%Y-%m-%d %H:%M:%S"),
        'total_samples': len(train_pairs) + len(val_pairs) + len(test_pairs),
        'train_count': len(train_pairs),
        'val_count': len(val_pairs),
        'test_count': len(test_pairs),
        'train_ratio': split_config.get('train_ratio', 0.8),
        'val_ratio': split_config.get('val_ratio', 0.1),
        'test_ratio': split_config.get('test_ratio', 0.1),
        'random_seed': split_config.get('random_seed', 42),
        'train_files': [{'image': os.path.basename(img), 'mask': os.path.basename(mask)} 
                       for img, mask in train_pairs],
        'val_files': [{'image': os.path.basename(img), 'mask': os.path.basename(mask)} 
                     for img, mask in val_pairs],
        'test_files': [{'image': os.path.basename(img), 'mask': os.path.basename(mask)} 
                      for img, mask in test_pairs],
    }
    
    info_file = os.path.join(output_dir, 'dataset_split_info.json')
    try:
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(split_info, f, ensure_ascii=False, indent=2)
        print(f"📄 数据集划分信息已保存到: {info_file}")
    except Exception as e:
        print(f"❌ 保存划分信息失败: {str(e)}")


def display_split_summary(train_pairs, val_pairs, test_pairs, actual_counts):
    """
    显示数据集划分摘要
    
    Args:
        train_pairs (list): 训练集文件对
        val_pairs (list): 验证集文件对
        test_pairs (list): 测试集文件对
        actual_counts (tuple): 实际分配数量
    """
    total_count = len(train_pairs) + len(val_pairs) + len(test_pairs)
    train_count, val_count, test_count = actual_counts
    
    print("="*80)
    print("📊 数据集划分摘要")
    print("="*80)
    print(f"📈 总样本数: {total_count}")
    print()
    print("📋 划分结果:")
    print(f"   🚂 训练集(train): {train_count:>6} 样本 ({train_count/total_count*100:>5.1f}%)")
    print(f"   🔬 验证集(val):   {val_count:>6} 样本 ({val_count/total_count*100:>5.1f}%)")
    print(f"   🧪 测试集(test):  {test_count:>6} 样本 ({test_count/total_count*100:>5.1f}%)")
    print()
    
    # 显示部分文件示例
    if train_pairs:
        print("🚂 训练集示例文件:")
        for i, (img_path, mask_path) in enumerate(train_pairs[:3], 1):
            img_name = os.path.basename(img_path)
            mask_name = os.path.basename(mask_path)
            print(f"   [{i}] {img_name} ↔ {mask_name}")
        if len(train_pairs) > 3:
            print(f"   ... 还有 {len(train_pairs) - 3} 对文件")
        print()
    
    if val_pairs:
        print("🔬 验证集示例文件:")
        for i, (img_path, mask_path) in enumerate(val_pairs[:3], 1):
            img_name = os.path.basename(img_path)
            mask_name = os.path.basename(mask_path)
            print(f"   [{i}] {img_name} ↔ {mask_name}")
        if len(val_pairs) > 3:
            print(f"   ... 还有 {len(val_pairs) - 3} 对文件")
        print()


def main():
    """主函数"""
    # 配置参数
    source_images_dir = r"C:\Users\root\OneDrive\Desktop\9classes_lesion\images"
    source_masks_dir = r"C:\Users\root\OneDrive\Desktop\9classes_lesion\masks"
    output_base_dir = r"C:\Users\root\OneDrive\Desktop\9classes_lesion"
    
    # 划分比例
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1
    random_seed = 42
    
    print("="*80)
    print("🔄 数据集划分工具")
    print("="*80)
    print(f"📂 源图像目录: {source_images_dir}")
    print(f"📂 源掩码目录: {source_masks_dir}")
    print(f"📂 输出目录: {output_base_dir}")
    print(f"📊 划分比例: 训练集{train_ratio*100:.0f}% | 验证集{val_ratio*100:.0f}% | 测试集{test_ratio*100:.0f}%")
    print(f"🎲 随机种子: {random_seed}")
    print()
    
    try:
        start_time = time.time()
        
        # 扫描匹配的文件对
        print("🔍 正在扫描匹配的图像和掩码文件...")
        matched_pairs = scan_matched_pairs(source_images_dir, source_masks_dir)
        
        if not matched_pairs:
            print("❌ 没有找到匹配的图像和掩码文件对！")
            print("请先确保图像和掩码文件已正确匹配。")
            return
        
        print(f"✅ 找到 {len(matched_pairs)} 对匹配的文件")
        print()
        
        # 划分数据集
        print("📊 正在划分数据集...")
        train_pairs, val_pairs, test_pairs, actual_counts = split_dataset(
            matched_pairs, train_ratio, val_ratio, test_ratio, random_seed
        )
        
        # 显示划分摘要
        display_split_summary(train_pairs, val_pairs, test_pairs, actual_counts)
        
        # 询问用户确认
        print("❓ 操作选项:")
        print("1. 复制文件到新的数据集目录 (推荐)")
        print("2. 移动文件到新的数据集目录")
        print("3. 仅生成划分信息，不复制文件")
        print("4. 取消操作")
        
        while True:
            choice = input("请选择操作 (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                break
            print("请输入有效选项 (1-4)")
        
        if choice == '4':
            print("🚫 操作已取消")
            return
        
        # 创建目录结构
        print("\n📁 正在创建数据集目录结构...")
        dirs = create_dataset_structure(output_base_dir)
        print()
        
        # 处理文件
        operation = 'copy' if choice == '1' else 'move' if choice == '2' else 'none'
        
        if operation != 'none':
            print(f"📦 正在{'复制' if operation == 'copy' else '移动'}文件...")
            
            # 处理训练集
            print("🚂 处理训练集...")
            train_success, train_error = copy_files(
                train_pairs, dirs['train_images'], dirs['train_masks'], operation
            )
            
            # 处理验证集
            print("🔬 处理验证集...")
            val_success, val_error = copy_files(
                val_pairs, dirs['val_images'], dirs['val_masks'], operation
            )
            
            # 处理测试集
            print("🧪 处理测试集...")
            test_success, test_error = copy_files(
                test_pairs, dirs['test_images'], dirs['test_masks'], operation
            )
            
            # 显示处理结果
            total_success = train_success + val_success + test_success
            total_error = train_error + val_error + test_error
            
            print()
            print("📈 文件处理结果:")
            print(f"   ✅ 成功处理: {total_success} 对文件")
            print(f"   ❌ 处理失败: {total_error} 对文件")
            print()
        
        # 保存划分信息
        split_config = {
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'random_seed': random_seed
        }
        save_split_info(output_base_dir, train_pairs, val_pairs, test_pairs, split_config)
        
        # 显示最终目录结构
        print("📁 生成的数据集目录结构:")
        print(f"   {output_base_dir}/")
        print(f"   ├── train/")
        print(f"   │   ├── images/ ({len(train_pairs)} 张图像)")
        print(f"   │   └── masks/  ({len(train_pairs)} 个掩码)")
        print(f"   ├── val/")
        print(f"   │   ├── images/ ({len(val_pairs)} 张图像)")
        print(f"   │   └── masks/  ({len(val_pairs)} 个掩码)")
        print(f"   ├── test/")
        print(f"   │   ├── images/ ({len(test_pairs)} 张图像)")
        print(f"   │   └── masks/  ({len(test_pairs)} 个掩码)")
        print(f"   └── dataset_split_info.json (划分详细信息)")
        
        # 完成时间
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n⏱️  处理耗时: {duration:.2f} 秒")
        print("✨ 数据集划分完成！")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 程序异常退出: {str(e)}")
        sys.exit(1)
    
    input("\n按Enter键退出...")


if __name__ == "__main__":
    main() 