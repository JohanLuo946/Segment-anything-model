#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量NII文件转彩色PNG掩码脚本
支持从配置文件读取颜色映射，批量处理多个文件
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import nibabel as nib
from pathlib import Path
import time


def load_color_config(config_path):
    """
    从配置文件加载颜色映射
    
    Args:
        config_path (str): 配置文件路径
        
    Returns:
        tuple: (颜色映射字典, 标签描述字典)
    """
    try:
        print(f"📋 正在加载配置文件: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 解析颜色映射
        color_mapping = {0: [0, 0, 0]}  # 背景默认为黑色
        label_descriptions = {0: "背景"}
        
        if 'Models' in config and 'ColorLabelTableModel' in config['Models']:
            color_table = config['Models']['ColorLabelTableModel']
            
            for item in color_table:
                label_id = item['ID']
                # 颜色从RGBA转换为RGB
                color = item['Color'][:3]  # 取前3个值（RGB）
                desc = item['Desc']
                
                color_mapping[label_id] = color
                label_descriptions[label_id] = desc
        
        print(f"✅ 成功加载 {len(color_mapping)} 种颜色配置")
        print("🎨 颜色映射:")
        for label_id, color in color_mapping.items():
            desc = label_descriptions.get(label_id, "未知")
            print(f"   ID {label_id} ({desc}): RGB{color}")
        
        return color_mapping, label_descriptions
        
    except Exception as e:
        print(f"❌ 加载配置文件失败: {str(e)}")
        print("🔄 使用默认颜色映射")
        
        # 默认颜色映射
        default_color_mapping = {
            0: [0, 0, 0],           # 背景
            1: [255, 0, 0],         # 红色
            2: [0, 255, 0],         # 绿色
            3: [0, 0, 255],         # 蓝色
            4: [255, 255, 0],       # 黄色
            5: [0, 255, 255],       # 青色
        }
        
        default_descriptions = {
            0: "背景",
            1: "类别1",
            2: "类别2", 
            3: "类别3",
            4: "类别4",
            5: "类别5"
        }
        
        return default_color_mapping, default_descriptions


def load_nii_file(nii_path):
    """
    加载.nii文件
    
    Args:
        nii_path (str): .nii文件路径
        
    Returns:
        tuple: (数据数组, 头信息)
    """
    try:
        # 加载NII文件
        nii_img = nib.load(nii_path)
        data = nii_img.get_fdata()
        header = nii_img.header
        
        return data, header
        
    except Exception as e:
        print(f"❌ 加载NII文件失败: {str(e)}")
        return None, None


def process_mask_data(data, label_descriptions, verbose=False):
    """
    处理掩码数据，保持原始标签ID用于颜色映射
    
    Args:
        data (numpy.ndarray): 原始数据
        label_descriptions (dict): 标签描述字典
        verbose (bool): 是否输出详细信息
        
    Returns:
        numpy.ndarray: 处理后的掩码数据（保持原始标签值）
    """
    if verbose:
        print("🔧 正在处理掩码数据...")
    
    # 获取数据的基本信息
    unique_values = np.unique(data)
    if verbose:
        print(f"📋 原始唯一像素值: {unique_values}")
        print(f"📊 原始数据类型: {data.dtype}")
        print(f"📊 原始数据范围: [{data.min():.6f}, {data.max():.6f}]")
    
    # 如果是3D数据，取中间切片或者找到最有信息的切片
    if len(data.shape) == 3:
        if verbose:
            print(f"📐 检测到3D数据，形状: {data.shape}")
        
        # 计算每个切片的非零像素数量，选择最有信息的切片
        slice_info = []
        for i in range(data.shape[2]):
            slice_data = data[:, :, i]
            non_zero_count = np.count_nonzero(slice_data)
            slice_info.append((i, non_zero_count))
        
        # 选择非零像素最多的切片
        best_slice_idx = max(slice_info, key=lambda x: x[1])[0]
        data_2d = data[:, :, best_slice_idx]
        if verbose:
            print(f"🎯 选择切片 {best_slice_idx}，非零像素数: {slice_info[best_slice_idx][1]}")
        
    elif len(data.shape) == 2:
        if verbose:
            print("📐 检测到2D数据")
        data_2d = data
    else:
        print(f"⚠️  不支持的数据维度: {data.shape}")
        return None
    
    if verbose:
        print(f"📊 2D数据形状: {data_2d.shape}")
        print(f"📊 2D数据范围: [{data_2d.min():.6f}, {data_2d.max():.6f}]")
    
    # 将数据转换为整数标签
    raw_mask_data = np.round(data_2d).astype(np.uint8)
    if verbose:
        print(f"📊 原始转换后数据范围: [{raw_mask_data.min()}, {raw_mask_data.max()}]")
    
    # 🔧 关键修复：所有标签ID进行-1操作以匹配配置文件
    if verbose:
        print("🔧 执行标签ID映射：所有ID进行-1操作")
    mask_data = raw_mask_data.copy()
    
    # 对所有非零像素进行-1操作
    non_zero_mask = mask_data > 0
    mask_data[non_zero_mask] = mask_data[non_zero_mask] - 1
    
    if verbose:
        print(f"📊 映射后数据范围: [{mask_data.min()}, {mask_data.max()}]")
        
        # 显示映射过程
        raw_unique = np.unique(raw_mask_data)
        mapped_unique = np.unique(mask_data)
        print("🔄 标签映射对照:")
        print(f"   原始标签: {raw_unique}")
        print(f"   映射后标签: {mapped_unique}")
        
        for raw_val in raw_unique:
            if raw_val > 0:  # 跳过背景
                mapped_val = raw_val - 1
                mapped_name = label_descriptions.get(mapped_val, "未知")
                print(f"   {raw_val} -> {mapped_val} ({mapped_name})")
    
    return mask_data


def labels_to_color_mask(label_mask, color_mapping, label_descriptions, verbose=False):
    """
    将标签掩码转换为彩色掩码
    
    Args:
        label_mask (numpy.ndarray): 标签掩码数据
        color_mapping (dict): 颜色映射字典
        label_descriptions (dict): 标签描述字典
        verbose (bool): 是否输出详细信息
        
    Returns:
        numpy.ndarray: RGB彩色掩码 (H, W, 3)
    """
    if verbose:
        print("🎨 正在转换为彩色掩码...")
        print(f"📊 输入标签掩码形状: {label_mask.shape}")
        print(f"📊 输入标签掩码范围: [{label_mask.min()}, {label_mask.max()}]")
        print(f"📊 输入标签掩码数据类型: {label_mask.dtype}")
    
    height, width = label_mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 统计使用的颜色
    used_colors = []
    
    # 遍历每个标签ID，映射到对应颜色
    unique_labels = np.unique(label_mask)
    if verbose:
        print(f"🏷️  要转换的标签: {unique_labels}")
    
    for label_id in unique_labels:
        label_int = int(label_id)
        if verbose:
            print(f"🔄 处理标签ID {label_int}...")
        
        if label_int in color_mapping:
            color = color_mapping[label_int]
            mask = label_mask == label_id
            color_mask[mask] = color
            
            pixel_count = np.sum(mask)
            label_name = label_descriptions.get(label_int, '未知')
            used_colors.append(f"ID {label_int} ({label_name}): RGB{color} - {pixel_count} 像素")
            
            if verbose:
                print(f"   ✅ ID {label_int} ({label_name}): {pixel_count} 像素 -> RGB{color}")
                
                # 验证颜色是否正确应用
                if pixel_count > 0:
                    sample_indices = np.where(mask)
                    if len(sample_indices[0]) > 0:
                        sample_color = color_mask[sample_indices[0][0], sample_indices[1][0]]
                        print(f"   🔍 验证样本像素颜色: {sample_color}")
        else:
            if verbose:
                print(f"⚠️  未知标签ID {label_int}，使用白色")
            mask = label_mask == label_id
            color_mask[mask] = [255, 255, 255]  # 白色表示未知标签
            pixel_count = np.sum(mask)
            used_colors.append(f"ID {label_int} (未知): RGB[255, 255, 255] - {pixel_count} 像素")
    
    if verbose:
        print("🎨 最终颜色映射结果:")
        for color_info in used_colors:
            print(f"   {color_info}")
        
        # 验证彩色掩码的颜色分布
        print("🔍 彩色掩码颜色验证:")
        unique_colors = np.unique(color_mask.reshape(-1, 3), axis=0)
        for color in unique_colors:
            count = np.sum(np.all(color_mask == color, axis=2))
            print(f"   RGB{color}: {count} 像素")
        
        print(f"✅ 彩色掩码转换完成，形状: {color_mask.shape}")
    
    return color_mask


def transform_mask(mask_data, verbose=False):
    """
    对掩码进行几何变换以匹配原图像
    - 顺时针旋转90°
    - 镜像操作
    
    Args:
        mask_data (numpy.ndarray): 原始掩码数据
        verbose (bool): 是否输出详细信息
        
    Returns:
        numpy.ndarray: 变换后的掩码数据
    """
    if verbose:
        print("🔄 正在进行几何变换...")
        print(f"📐 原始形状: {mask_data.shape}")
    
    # 1. 顺时针旋转90°
    # numpy.rot90 默认是逆时针旋转，k=-1表示顺时针旋转90°
    rotated_mask = np.rot90(mask_data, k=-1)
    if verbose:
        print(f"🔄 旋转后形状: {rotated_mask.shape}")
    
    # 2. 镜像操作（水平翻转）
    mirrored_mask = np.fliplr(rotated_mask)
    if verbose:
        print(f"🪞 镜像后形状: {mirrored_mask.shape}")
        print("✅ 几何变换完成")
    
    return mirrored_mask


def process_single_nii(nii_path, output_dir, color_mapping, label_descriptions, verbose=False):
    """
    处理单个NII文件
    
    Args:
        nii_path (str): 输入文件路径
        output_dir (str): 输出目录
        color_mapping (dict): 颜色映射
        label_descriptions (dict): 标签描述
        verbose (bool): 是否输出详细信息
        
    Returns:
        bool: 处理是否成功
    """
    try:
        if verbose:
            print(f"🔍 正在处理: {os.path.basename(nii_path)}")
        
        # 加载NII文件
        data, header = load_nii_file(nii_path)
        if data is None:
            return False
        
        # 处理掩码数据
        mask_data = process_mask_data(data, label_descriptions, verbose=verbose)
        if mask_data is None:
            return False
        
        # 生成输出文件名（在几何变换前确定，以便在verbose模式下显示）
        input_filename = os.path.basename(nii_path)
        base_name = os.path.splitext(input_filename)[0]
        
        # 🔧 优化文件名：去掉_jpg_Label后缀，使用简洁命名
        if base_name.endswith('_jpg_Label'):
            clean_name = base_name.replace('_jpg_Label', '')
        else:
            clean_name = base_name
        
        output_filename = f"{clean_name}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        if verbose:
            print(f"📝 文件名处理:")
            print(f"   原始文件名: {input_filename}")
            print(f"   处理后文件名: {output_filename}")
        
        # 进行几何变换（在标签级别）
        transformed_mask = transform_mask(mask_data, verbose=verbose)
        
        # 转换为彩色掩码
        color_mask = labels_to_color_mask(transformed_mask, color_mapping, label_descriptions, verbose=verbose)
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存彩色掩码
        mask_image = Image.fromarray(color_mask, mode='RGB')
        mask_image.save(output_path, 'PNG')
        
        if verbose:
            print(f"✅ 彩色掩码保存成功!")
            print(f"📁 输出路径: {output_path}")
            print(f"📏 最终图像尺寸: {mask_image.size}")
            print(f"🎨 图像模式: RGB彩色")
        
        return True
        
    except Exception as e:
        print(f"❌ 处理文件失败: {str(e)}")
        return False


def batch_process_nii_files(input_dir, output_dir, config_path=None):
    """
    批量处理NII文件转换为彩色PNG掩码
    
    Args:
        input_dir (str): 输入目录
        output_dir (str): 输出目录
        config_path (str): 配置文件路径
    """
    print("="*80)
    print("🔬 批量NII文件转彩色PNG掩码转换器")
    print("="*80)
    
    # 加载颜色配置
    if config_path and os.path.exists(config_path):
        color_mapping, label_descriptions = load_color_config(config_path)
    else:
        print("⚠️  未提供配置文件或文件不存在，使用默认配置")
        color_mapping = {
            0: [0, 0, 0],
            1: [255, 0, 0],
            2: [0, 255, 0],
            3: [0, 0, 255],
            4: [255, 255, 0],
            5: [0, 255, 255]
        }
        label_descriptions = {
            0: "背景", 1: "类别1", 2: "类别2", 
            3: "类别3", 4: "类别4", 5: "类别5"
        }
    
    # 查找所有.nii文件
    nii_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.nii'):
                nii_files.append(os.path.join(root, file))
    
    total_files = len(nii_files)
    
    if total_files == 0:
        print("⚠️  未找到.nii文件！")
        return
    
    print(f"📦 找到 {total_files} 个.nii文件")
    print(f"📁 输入目录: {input_dir}")
    print(f"📁 输出目录: {output_dir}")
    print()
    
    # 处理统计
    success_count = 0
    error_count = 0
    start_time = time.time()
    
    # 批量处理
    for i, nii_path in enumerate(nii_files, 1):
        print(f"[{i}/{total_files}] 处理: {os.path.basename(nii_path)}")
        print(f"📂 位置: {os.path.dirname(nii_path)}")
        
        # 第一个文件使用详细模式，用于验证处理流程
        verbose_mode = (i == 1)
        if verbose_mode:
            print("🔍 详细模式：显示第一个文件的完整处理过程")
        
        success = process_single_nii(nii_path, output_dir, color_mapping, label_descriptions, verbose=verbose_mode)
        
        if success:
            print(f"✅ 成功")
            success_count += 1
        else:
            print(f"❌ 失败")
            error_count += 1
        
        print("-" * 60)
    
    # 显示最终统计
    end_time = time.time()
    duration = end_time - start_time
    
    print()
    print("="*80)
    print("📊 批量处理完成统计")
    print("="*80)
    print(f"⏱️  总耗时: {duration:.2f} 秒")
    print(f"📦 总文件数: {total_files}")
    print(f"✅ 成功处理: {success_count}")
    print(f"❌ 处理失败: {error_count}")
    
    if error_count == 0:
        print("🎉 所有文件处理成功！")
    else:
        print(f"⚠️  有 {error_count} 个文件处理失败")


def main():
    """主函数"""
    # 配置路径
    input_dir = r"C:\Users\root\OneDrive\Desktop\像素级标注-2-标签修正后\分割汇总-after"
    output_dir = r"C:\Users\root\OneDrive\Desktop\9classes_lesion\masks"
    config_path = r"标注配置.json"
    
    print(f"📂 输入目录: {input_dir}")
    print(f"📂 输出目录: {output_dir}")
    print(f"📋 配置文件: {config_path}")
    print()
    
    try:
        batch_process_nii_files(input_dir, output_dir, config_path)
        print("\n✨ 程序执行完成！")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 程序异常退出: {str(e)}")
        sys.exit(1)
    
    input("\n按Enter键退出...")


if __name__ == "__main__":
    main() 