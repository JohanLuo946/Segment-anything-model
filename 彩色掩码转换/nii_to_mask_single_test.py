#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NII文件转彩色PNG掩码测试脚本
将单个.nii文件转换为彩色.png掩码文件，保持标注软件的颜色映射
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import nibabel as nib
from pathlib import Path


# 从标注配置文件中提取的颜色映射
COLOR_MAPPING = {
    0: [0, 0, 0],           # 背景 - 黑色
    1: [255, 0, 0],         # sd - 红色
    2: [0, 255, 0],         # sdxr - 绿色
    3: [0, 0, 255],         # sdbb - 蓝色
    4: [255, 255, 0],       # sma - 黄色
    5: [0, 255, 255],       # sdnz - 青色
    6: [255, 0, 255],       # sdxj - 品红色
    7: [255, 239, 213],     # rkssz - 浅橙色
    8: [0, 0, 205],         # sdbbz - 深蓝色
    9: [205, 133, 63],      # rtzl - 棕色
    10: [210, 180, 140],    # sd_left - 浅棕色
    11: [102, 205, 170],    # sd_right - 海绿色
    12: [205, 53, 203],     # sdbbq - 紫红色
    13: [205, 176, 57],     # hdfyb - 橄榄绿
    14: [205, 198, 189]     # sdry - 浅灰色
}

# 标签描述映射
LABEL_DESCRIPTIONS = {
    0: "背景",
    1: "sd",
    2: "sdxr", 
    3: "sdbb",
    4: "sma",
    5: "sdnz",
    6: "sdxj",
    7: "rkssz",
    8: "sdbbz",
    9: "rtzl",
    10: "sd_left",
    11: "sd_right",
    12: "sdbbq",
    13: "hdfyb",
    14: "sdry"
}


def load_nii_file(nii_path):
    """
    加载.nii文件
    
    Args:
        nii_path (str): .nii文件路径
        
    Returns:
        tuple: (数据数组, 头信息)
    """
    try:
        print(f"🔍 正在加载NII文件: {os.path.basename(nii_path)}")
        
        # 加载NII文件
        nii_img = nib.load(nii_path)
        data = nii_img.get_fdata()
        header = nii_img.header
        
        print(f"📊 数据形状: {data.shape}")
        print(f"📊 数据类型: {data.dtype}")
        print(f"📊 数值范围: [{data.min():.2f}, {data.max():.2f}]")
        print(f"📊 唯一值数量: {len(np.unique(data))}")
        print(f"📊 头信息维度: {header.get_data_shape()}")
        
        return data, header
        
    except Exception as e:
        print(f"❌ 加载NII文件失败: {str(e)}")
        return None, None


def process_mask_data(data):
    """
    处理掩码数据，保持原始标签ID用于颜色映射
    
    Args:
        data (numpy.ndarray): 原始数据
        
    Returns:
        numpy.ndarray: 处理后的掩码数据（保持原始标签值）
    """
    print("🔧 正在处理掩码数据...")
    
    # 获取数据的基本信息
    unique_values = np.unique(data)
    print(f"📋 原始唯一像素值: {unique_values}")
    print(f"📊 原始数据类型: {data.dtype}")
    print(f"📊 原始数据范围: [{data.min():.6f}, {data.max():.6f}]")
    
    # 显示检测到的标签类型
    detected_labels = []
    for val in unique_values:
        val_int = int(round(val))  # 使用round确保正确转换
        if val_int in LABEL_DESCRIPTIONS:
            detected_labels.append(f"原始值 {val:.6f} -> ID {val_int}: {LABEL_DESCRIPTIONS[val_int]}")
        else:
            detected_labels.append(f"原始值 {val:.6f} -> ID {val_int}: 未知标签")
    
    print("🏷️  检测到的标签映射:")
    for label in detected_labels:
        print(f"   {label}")
    
    # 如果是3D数据，取中间切片或者找到最有信息的切片
    if len(data.shape) == 3:
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
        print(f"🎯 选择切片 {best_slice_idx}，非零像素数: {slice_info[best_slice_idx][1]}")
        
    elif len(data.shape) == 2:
        print("📐 检测到2D数据")
        data_2d = data
    else:
        print(f"⚠️  不支持的数据维度: {data.shape}")
        return None
    
    # 打印2D数据信息
    print(f"📊 2D数据形状: {data_2d.shape}")
    print(f"📊 2D数据范围: [{data_2d.min():.6f}, {data_2d.max():.6f}]")
    
    # 将数据转换为整数标签
    raw_mask_data = np.round(data_2d).astype(np.uint8)
    print(f"📊 原始转换后数据范围: [{raw_mask_data.min()}, {raw_mask_data.max()}]")
    
    # 🔧 关键修复：所有标签ID进行-1操作以匹配配置文件
    print("🔧 执行标签ID映射：所有ID进行-1操作")
    mask_data = raw_mask_data.copy()
    
    # 对所有非零像素进行-1操作
    non_zero_mask = mask_data > 0
    mask_data[non_zero_mask] = mask_data[non_zero_mask] - 1
    
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
            raw_name = "未知"
            mapped_name = LABEL_DESCRIPTIONS.get(mapped_val, "未知")
            mapped_color = COLOR_MAPPING.get(mapped_val, [255, 255, 255])
            print(f"   {raw_val} -> {mapped_val} ({mapped_name}) RGB{mapped_color}")
    
    # 统计每个标签的像素数量
    unique_labels, counts = np.unique(mask_data, return_counts=True)
    print("📊 最终标签统计:")
    for label, count in zip(unique_labels, counts):
        label_name = LABEL_DESCRIPTIONS.get(int(label), "未知")
        percentage = (count / mask_data.size) * 100
        color = COLOR_MAPPING.get(int(label), [255, 255, 255])
        print(f"   ID {label} ({label_name}): {count} 像素 ({percentage:.2f}%) -> 颜色 RGB{color}")
        
        # 特别标注sma标签
        if label_name == "sma":
            print(f"   ⭐ 发现sma标签！应该显示为黄色 RGB{color}")
    
    print(f"✅ 掩码处理完成，最终标签范围: [{mask_data.min()}, {mask_data.max()}]")
    
    return mask_data


def labels_to_color_mask(label_mask):
    """
    将标签掩码转换为彩色掩码
    
    Args:
        label_mask (numpy.ndarray): 标签掩码数据
        
    Returns:
        numpy.ndarray: RGB彩色掩码 (H, W, 3)
    """
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
    print(f"🏷️  要转换的标签: {unique_labels}")
    
    for label_id in unique_labels:
        label_int = int(label_id)
        print(f"🔄 处理标签ID {label_int}...")
        
        if label_int in COLOR_MAPPING:
            color = COLOR_MAPPING[label_int]
            mask = label_mask == label_id
            color_mask[mask] = color
            
            pixel_count = np.sum(mask)
            label_name = LABEL_DESCRIPTIONS.get(label_int, '未知')
            used_colors.append(f"ID {label_int} ({label_name}): RGB{color} - {pixel_count} 像素")
            
            print(f"   ✅ ID {label_int} ({label_name}): {pixel_count} 像素 -> RGB{color}")
            
            # 验证颜色是否正确应用
            if pixel_count > 0:
                sample_indices = np.where(mask)
                if len(sample_indices[0]) > 0:
                    sample_color = color_mask[sample_indices[0][0], sample_indices[1][0]]
                    print(f"   🔍 验证样本像素颜色: {sample_color}")
        else:
            print(f"⚠️  未知标签ID {label_int}，使用白色")
            mask = label_mask == label_id
            color_mask[mask] = [255, 255, 255]  # 白色表示未知标签
            pixel_count = np.sum(mask)
            used_colors.append(f"ID {label_int} (未知): RGB[255, 255, 255] - {pixel_count} 像素")
    
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


def transform_mask(mask_data):
    """
    对掩码进行几何变换以匹配原图像
    - 顺时针旋转90°
    - 镜像操作
    
    Args:
        mask_data (numpy.ndarray): 原始掩码数据
        
    Returns:
        numpy.ndarray: 变换后的掩码数据
    """
    print("🔄 正在进行几何变换...")
    
    # 打印原始形状
    print(f"📐 原始形状: {mask_data.shape}")
    
    # 1. 顺时针旋转90°
    # numpy.rot90 默认是逆时针旋转，k=-1表示顺时针旋转90°
    rotated_mask = np.rot90(mask_data, k=-1)
    print(f"🔄 旋转后形状: {rotated_mask.shape}")
    
    # 2. 镜像操作（水平翻转）
    # 可以根据实际需要选择水平翻转或垂直翻转
    mirrored_mask = np.fliplr(rotated_mask)  # 水平翻转
    # 如果需要垂直翻转，使用: mirrored_mask = np.flipud(rotated_mask)
    
    print(f"🪞 镜像后形状: {mirrored_mask.shape}")
    print("✅ 几何变换完成")
    
    return mirrored_mask


def save_mask_png(mask_data, output_path):
    """
    保存彩色掩码为PNG文件
    
    Args:
        mask_data (numpy.ndarray): 标签掩码数据
        output_path (str): 输出文件路径
    """
    try:
        print(f"💾 正在保存彩色掩码文件: {os.path.basename(output_path)}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 进行几何变换（在标签级别）
        transformed_label_mask = transform_mask(mask_data)
        
        # 转换为彩色掩码
        color_mask = labels_to_color_mask(transformed_label_mask)
        
        # 转换为PIL图像并保存（RGB模式）
        mask_image = Image.fromarray(color_mask, mode='RGB')
        mask_image.save(output_path, 'PNG')
        
        print(f"✅ 彩色掩码保存成功!")
        print(f"📁 输出路径: {output_path}")
        print(f"📏 最终图像尺寸: {mask_image.size}")
        print(f"🎨 图像模式: RGB彩色")
        
        return True
        
    except Exception as e:
        print(f"❌ 保存掩码失败: {str(e)}")
        return False


def nii_to_mask_single(nii_path, output_dir):
    """
    将单个NII文件转换为彩色PNG掩码
    
    Args:
        nii_path (str): 输入的.nii文件路径
        output_dir (str): 输出目录
        
    Returns:
        bool: 转换是否成功
    """
    print("="*70)
    print("🔬 NII文件转彩色PNG掩码转换器")
    print("="*70)
    print("🎨 支持14种病灶类型的颜色映射")
    print("🔄 包含几何变换：顺时针90°旋转 + 水平翻转")
    print("="*70)
    
    # 检查输入文件
    if not os.path.exists(nii_path):
        print(f"❌ 输入文件不存在: {nii_path}")
        return False
    
    if not nii_path.lower().endswith('.nii'):
        print(f"❌ 不是.nii文件: {nii_path}")
        return False
    
    # 加载NII文件
    data, header = load_nii_file(nii_path)
    if data is None:
        return False
    
    # 处理掩码数据
    mask_data = process_mask_data(data)
    if mask_data is None:
        return False
    
    # 生成输出文件名
    input_filename = os.path.basename(nii_path)
    base_name = os.path.splitext(input_filename)[0]  # 去掉.nii扩展名
    
    # 🔧 优化文件名：去掉_jpg_Label后缀，使用简洁命名
    if base_name.endswith('_jpg_Label'):
        clean_name = base_name.replace('_jpg_Label', '')
    else:
        clean_name = base_name
    
    output_filename = f"{clean_name}.png"
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"📝 文件名处理:")
    print(f"   原始文件名: {input_filename}")
    print(f"   处理后文件名: {output_filename}")
    
    # 保存掩码
    success = save_mask_png(mask_data, output_path)
    
    if success:
        print("🎉 转换完成！")
        return True
    else:
        print("💥 转换失败！")
        return False


def main():
    """主函数"""
    # 测试路径
    nii_path = r"C:\Users\root\OneDrive\Desktop\像素级标注-2-标签修正后\分割汇总-after\声门癌sma\喉癌_白洪和133004003172905_20181219_190144380_jpg_Label.nii"
    output_dir = r"C:\Users\root\OneDrive\Desktop\像素级标注-2-标签修正后\test"
    
    print(f"📂 输入文件: {nii_path}")
    print(f"📂 输出目录: {output_dir}")
    print()
    
    try:
        success = nii_to_mask_single(nii_path, output_dir)
        
        if success:
            print("\n✨ 程序执行完成！")
        else:
            print("\n💥 程序执行失败！")
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