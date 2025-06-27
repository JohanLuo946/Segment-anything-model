#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔥 SAM ViT-B声带分割测试脚本 - 老哥定制版 🔥
包含详细的结果解释和更好的可视化
支持：背景、左声带、右声带、声带小结、声带白斑、声带乳头状瘤
与train_sam_optimized.py完全兼容！
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置matplotlib使用英文，避免中文字体问题
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# SAM模型定义（与训练脚本完全一致）
class EnhancedSAMModel(nn.Module):
    """老哥的强化SAM模型 - 与训练脚本完全一致！"""
    
    def __init__(self, sam_model, num_classes):
        super().__init__()
        self.sam = sam_model
        self.num_classes = num_classes
        
        # 🔥 多尺度特征融合分割头（与训练脚本一致）
        self.segmentation_head = nn.Sequential(
            # 第一层：特征提取
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 第二层：特征细化
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 第三层：分类输出
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        
        # 🎯 注意力模块 - 让模型主动关注病灶（与训练脚本一致）
        self.attention = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 冻结SAM的部分参数，只微调关键部分
        self.freeze_sam_components()
        
        print("🎯 EnhancedSAMModel loaded (compatible with trained model)")
    
    def freeze_sam_components(self):
        """冻结SAM的大部分参数，只训练必要的部分"""
        # 冻结image_encoder的前面几层
        layers = list(self.sam.image_encoder.children())
        for i, layer in enumerate(layers[:-3]):  # 只解冻最后3层
            for param in layer.parameters():
                param.requires_grad = False
    
    def forward(self, images):
        batch_size = images.shape[0]
        
        # SAM图像编码
        image_embeddings = self.sam.image_encoder(images)
        
        # 🎯 注意力增强
        attention_map = self.attention(image_embeddings)
        enhanced_features = image_embeddings * attention_map
        
        # 多尺度分割
        segmentation_logits = self.segmentation_head(enhanced_features)
        
        # 上采样到原始尺寸
        segmentation_logits = F.interpolate(
            segmentation_logits,
            size=(images.shape[2], images.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        
        # 虚拟IoU预测（保持兼容性）
        iou_predictions = torch.ones(batch_size, 1).to(images.device) * 0.8
        
        return segmentation_logits, iou_predictions

def test_single_image_sam(model_path, image_path, save_dir="/root/autodl-tmp/SAM/results/predictions/lesion"):
    """🔥 SAM ViT-B优化模型6类分割单张图片测试 - 老哥定制版"""
    
    # 确保保存目录存在
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    # 加载SAM模型
    print(f"📂 Loading SAM model: {model_path}")
    try:
        from segment_anything import sam_model_registry
        
        # 创建SAM基础模型
        sam = sam_model_registry["vit_b"](checkpoint=None)  # 先创建架构
        
        # 创建我们的分割模型
        model = EnhancedSAMModel(sam, 6).to(device)  # 6类输出
        
        # 加载训练好的权重
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # 显示模型性能信息
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            print(f"✅ Model performance: mIoU={metrics.get('mIoU', 0):.4f}")
        
    except ImportError:
        print("❌ segment_anything not found, installing...")
        os.system("pip install segment-anything")
        from segment_anything import sam_model_registry
        sam = sam_model_registry["vit_b"](checkpoint=None)  # 使用ViT-b模型
        model = EnhancedSAMModel(sam, 6).to(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    
    # 读取和预处理图像
    print(f"🔍 Processing image: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    original_size = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # SAM标准预处理（1024x1024）
    target_size = 1024
    image_resized = cv2.resize(image_rgb, (target_size, target_size))
    
    # 标准化（SAM使用的均值和标准差）
    pixel_mean = np.array([123.675, 116.28, 103.53])
    pixel_std = np.array([58.395, 57.12, 57.375])
    image_normalized = (image_resized - pixel_mean) / pixel_std
    
    # 转换为tensor
    image_tensor = torch.from_numpy(image_normalized).float().permute(2, 0, 1).unsqueeze(0).to(device)
    
    # 预测
    print("🔮 Running inference...")
    with torch.no_grad():
        outputs, iou_preds = model(image_tensor)
        
        # 调整到原始尺寸
        outputs = F.interpolate(outputs, size=original_size, mode='bilinear', align_corners=False)
        
        # 获取预测结果
        pred_probs = F.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
        pred_mask = np.argmax(pred_probs, axis=0).astype(np.uint8)
    
    # 4类别定义（只显示病灶，隐藏左右声带）
    class_names = [
        'Background',      # ID: 0
        'Left',   # ID: 1 (隐藏显示)
        'Right',  # ID: 2 (隐藏显示)
        'Vocal Nodules',   # ID: 105 -> 3 声带小结
        'Vocal Leukoplakia', # ID: 23 -> 4 声带白斑
        'Vocal Papilloma'  # ID: 146 -> 5 声带乳头状瘤
    ]
    
    # 显示用的类别名称（只包含病灶）
    display_class_names = [
        'Background',
        'Vocal Nodules',    # 声带小结
        'Vocal Leukoplakia', # 声带白斑
        'Vocal Papilloma'   # 声带乳头状瘤
    ]
    
    # 颜色映射：背景+3种病灶
    display_colors = [
        [0, 0, 0],        # 背景-黑色
        [255, 255, 0],    # 声带小结-黄色
        [255, 0, 255],    # 声带白斑-洋红色
        [0, 255, 255],    # 声带乳头状瘤-青色
    ]
    
    # 完整的类别到显示类别的映射
    class_to_display = {
        0: 0,  # 背景 -> 背景
        1: 0,  # 左声带 -> 背景（隐藏）
        2: 0,  # 右声带 -> 背景（隐藏）
        3: 1,  # 声带小结 -> 显示类别1
        4: 2,  # 声带白斑 -> 显示类别2
        5: 3,  # 声带乳头状瘤 -> 显示类别3
    }
    
    # 创建彩色分割图（只显示病灶）
    display_mask = np.zeros_like(pred_mask)
    for original_class, display_class in class_to_display.items():
        display_mask[pred_mask == original_class] = display_class
    
    colored_mask = np.zeros((*display_mask.shape, 3), dtype=np.uint8)
    for i, color in enumerate(display_colors):
        colored_mask[display_mask == i] = color
    
    # 创建叠加图
    overlay = cv2.addWeighted(image_rgb, 0.7, colored_mask, 0.3, 0)
    
    # 统计各类别像素数（只统计病灶类别）
    unique, counts = np.unique(display_mask, return_counts=True)
    total_pixels = display_mask.size
    
    print("\n📊 Lesion Segmentation Statistics:")
    for cls, count in zip(unique, counts):
        percentage = count / total_pixels * 100
        if cls == 0:
            print(f"  {display_class_names[cls]}: {count:,} pixels ({percentage:.2f}%)")
        else:
            print(f"  {display_class_names[cls]}: {count:,} pixels ({percentage:.3f}%)")
    
    # 额外统计原始类别中的声带信息（仅用于内部统计）
    original_unique, original_counts = np.unique(pred_mask, return_counts=True)
    vocal_fold_pixels = 0
    for cls, count in zip(original_unique, original_counts):
        if cls in [1, 2]:  # 左右声带
            vocal_fold_pixels += count
    
    if vocal_fold_pixels > 0:
        vocal_fold_percentage = vocal_fold_pixels / total_pixels * 100
        print(f"  [Hidden] Vocal Folds Total: {vocal_fold_pixels:,} pixels ({vocal_fold_percentage:.2f}%)")
    
    # 创建可视化图
    fig = plt.figure(figsize=(20, 16))
    filename = Path(image_path).name
    fig.suptitle(f'🔥 SAM ViT-B Vocal Lesion Segmentation Analysis - 老哥优化版: {filename}', fontsize=18, fontweight='bold')
    
    # 创建网格布局 - 3行4列
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.2)
    
    # 第一行：基本结果
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image_rgb)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(colored_mask)
    ax2.set_title('Segmentation Result', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(overlay)
    ax3.set_title('Overlay (70% Original + 30% Mask)', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # 添加图例（只显示病灶类别）
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.axis('off')
    legend_elements = []
    for i, (name, color) in enumerate(zip(display_class_names, display_colors)):
        legend_elements.append(patches.Patch(color=np.array(color)/255, label=name))
    ax4.legend(handles=legend_elements, loc='center', fontsize=12, title='Lesion Legend', title_fontsize=14)
    ax4.set_title('Color Legend', fontsize=14, fontweight='bold')
    
    # 第二行：病灶置信度分析
    lesion_indices = [3, 4, 5]  # 声带小结、声带白斑、声带乳头状瘤
    lesion_titles = [
        'Vocal Nodules Confidence',
        'Vocal Leukoplakia Confidence', 
        'Vocal Papilloma Confidence'
    ]
    
    for i in range(3):
        ax = fig.add_subplot(gs[1, i])
        lesion_idx = lesion_indices[i]
        im = ax.imshow(pred_probs[lesion_idx], cmap='hot', vmin=0, vmax=1)
        ax.set_title(lesion_titles[i], fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('Confidence', fontsize=10)
    
    # 第二行右侧：病灶统计信息
    ax = fig.add_subplot(gs[1, 3])
    ax.axis('off')
    
    # 计算最大置信度用于统计
    max_confidence = np.max(pred_probs, axis=0)
    
    stats_text = "Lesion Detection Statistics:\n\n"
    
    # 只统计病灶类别
    for cls, count in zip(unique, counts):
        if cls > 0:  # 跳过背景
            percentage = count / total_pixels * 100
            original_cls = [k for k, v in class_to_display.items() if v == cls][0]
            avg_conf = pred_probs[original_cls][pred_mask == original_cls].mean() if count > 0 else 0
            stats_text += f"{display_class_names[cls]}:\n"
            stats_text += f"  Area: {percentage:.4f}%\n"
            stats_text += f"  Confidence: {avg_conf:.3f}\n\n"
    
    # 检查是否有病灶被检测到
    lesion_detected = any(cls > 0 and counts[unique == cls][0] > total_pixels * 0.0001 for cls in unique)
    
    if not lesion_detected:
        stats_text += "No significant lesions detected.\n\n"
    
    # 添加整体置信度统计（仅病灶区域）
    lesion_mask = display_mask > 0
    if np.any(lesion_mask):
        lesion_conf = max_confidence[lesion_mask].mean()
        stats_text += f"Lesion Avg Confidence: {lesion_conf:.3f}\n"
    
    stats_text += f"Image Size: {original_size[1]}×{original_size[0]}"
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    ax.set_title('Lesion Statistics', fontsize=14, fontweight='bold')
    
    # 第三行：背景和整体分析
    analysis_titles = [
        'Background Confidence',
        'Max Lesion Confidence',
        'Lesion Probability Map'
    ]
    
    # 第三行第1列：背景置信度
    ax = fig.add_subplot(gs[2, 0])
    im = ax.imshow(pred_probs[0], cmap='gray', vmin=0, vmax=1)
    ax.set_title(analysis_titles[0], fontsize=12, fontweight='bold')
    ax.axis('off')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Confidence', fontsize=10)
    
    # 第三行第2列：最大病灶置信度
    ax = fig.add_subplot(gs[2, 1])
    lesion_max_conf = np.max(pred_probs[3:6], axis=0)  # 3个病灶类别的最大置信度
    im = ax.imshow(lesion_max_conf, cmap='hot', vmin=0, vmax=1)
    ax.set_title(analysis_titles[1], fontsize=12, fontweight='bold')
    ax.axis('off')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Confidence', fontsize=10)
    
    # 第三行第3列：病灶概率分布图
    ax = fig.add_subplot(gs[2, 2])
    # 组合所有病灶类别的概率
    combined_lesion_prob = np.sum(pred_probs[3:6], axis=0)
    im = ax.imshow(combined_lesion_prob, cmap='plasma', vmin=0, vmax=1)
    ax.set_title(analysis_titles[2], fontsize=12, fontweight='bold')
    ax.axis('off')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Probability', fontsize=10)
    
    # 第三行右侧：详细诊断报告
    ax = fig.add_subplot(gs[2, 3])
    ax.axis('off')
    
    report_text = "🏥 DIAGNOSTIC REPORT:\n\n"
    
    # 计算每种病灶的面积和置信度
    for cls in [1, 2, 3]:  # display_mask中的病灶类别
        if cls in unique:
            count = counts[unique == cls][0]
            percentage = count / total_pixels * 100
            original_cls = [k for k, v in class_to_display.items() if v == cls][0]
            avg_conf = pred_probs[original_cls][pred_mask == original_cls].mean()
            max_conf = pred_probs[original_cls][pred_mask == original_cls].max() if count > 0 else 0
            
            if percentage > 0.001:  # 阈值降低到0.001%
                report_text += f"✓ {display_class_names[cls]} DETECTED:\n"
                report_text += f"  Coverage: {percentage:.4f}%\n"
                report_text += f"  Confidence: {avg_conf:.3f} (max: {max_conf:.3f})\n"
                if percentage > 0.1:
                    report_text += f"  ⚠️ Significant finding!\n"
                report_text += "\n"
    
    # 整体评估
    total_lesion_area = sum(counts[unique == cls][0] for cls in [1, 2, 3] if cls in unique)
    total_lesion_percentage = total_lesion_area / total_pixels * 100
    
    if total_lesion_percentage > 0.01:
        report_text += f"📊 TOTAL LESION AREA: {total_lesion_percentage:.3f}%\n"
        if total_lesion_percentage > 1.0:
            report_text += "🔴 High lesion burden detected\n"
        elif total_lesion_percentage > 0.1:
            report_text += "🟡 Moderate lesion presence\n"
        else:
            report_text += "🟢 Minor lesion presence\n"
    else:
        report_text += "✅ NO SIGNIFICANT LESIONS\n"
    
    ax.text(0.05, 0.95, report_text, transform=ax.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax.set_title('Medical Report', fontsize=14, fontweight='bold')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存结果
    result_path = save_dir / f"{Path(image_path).stem}_sam_analysis.png"
    plt.savefig(result_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"💾 Analysis saved: {result_path}")
    
    plt.show()
    
    return pred_mask, pred_probs, colored_mask, overlay

if __name__ == "__main__":
    # 测试参数 - 使用新训练的SAM模型路径
    model_path = "autodl-tmp/SAM/results/models/run_3/models/best_model.pth"
    
    # 获取图像路径
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
         # 默认测试图片
        image_path = "autodl-tmp/SAM/sdbb+rtzl.jpg"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please complete SAM training first or specify correct model path")
        print("Expected model path: /root/autodl-tmp/SAM/results/models/best_model.pth")
    elif not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        print("Please specify correct image path")
    else:
        try:
            print("="*80)
            print("🔥 SAM ViT-B VOCAL LESION SEGMENTATION ANALYSIS - 老哥优化版")
            print("="*80)
            print(f"📂 Model: {model_path}")
            print(f"🖼️  Image: {image_path}")
            print("\n📋 Understanding the Results:")
            print("• Row 1: Original image, lesion segmentation, overlay, and color legend")
            print("• Row 2: Individual lesion confidence maps + detection statistics")  
            print("• Row 3: Background analysis, max lesion confidence, probability map + medical report")
            print("• Color coding: Yellow=Vocal Nodules, Magenta=Vocal Leukoplakia, Cyan=Vocal Papilloma")
            print("• Note: Left and right vocal folds are hidden in this analysis")
            print("• Confidence interpretation: Bright=high confidence, Dark=low confidence")
            print("="*80)
            
            test_single_image_sam(model_path, image_path)
            print("✅ SAM lesion segmentation analysis completed successfully!")
        except Exception as e:
            print(f"❌ Error during testing: {e}")
            import traceback
            traceback.print_exc() 