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

def test_single_image_sam(model_path, image_path, save_dir="/root/autodl-tmp/SAM/results/predictions"):
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
    
    # 6类别定义（与训练脚本ID映射一致）
    class_names = [
        'Background',      # ID: 0
        'Left', # ID: 170 -> 1
        'Right',# ID: 184 -> 2
        'sdxj',    # ID: 105 -> 3
        'sdbb',# ID: 23 -> 4
        'rtzl'  # ID: 146 -> 5
    ]
    
    class_colors = [
        [0, 0, 0],        # 背景-黑色
        [255, 0, 0],      # 左声带-红色
        [0, 255, 0],      # 右声带-绿色  
        [255, 255, 0],    # 声带小结-黄色
        [255, 0, 255],    # 声带白斑-洋红色
        [0, 255, 255],    # 声带乳头状瘤-青色
    ]
    
    # 创建彩色分割图
    colored_mask = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    for i, color in enumerate(class_colors):
        colored_mask[pred_mask == i] = color
    
    # 创建叠加图
    overlay = cv2.addWeighted(image_rgb, 0.7, colored_mask, 0.3, 0)
    
    # 统计各类别像素数
    unique, counts = np.unique(pred_mask, return_counts=True)
    total_pixels = pred_mask.size
    
    print("\n📊 Segmentation Statistics:")
    for cls, count in zip(unique, counts):
        percentage = count / total_pixels * 100
        print(f"  {class_names[cls]}: {count:,} pixels ({percentage:.2f}%)")
    
    # 创建可视化图
    fig = plt.figure(figsize=(20, 16))
    filename = Path(image_path).name
    fig.suptitle(f'🔥 SAM ViT-B Vocal Fold Segmentation Analysis - 老哥优化版: {filename}', fontsize=18, fontweight='bold')
    
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
    
    # 添加图例
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.axis('off')
    legend_elements = []
    for i, (name, color) in enumerate(zip(class_names, class_colors)):
        legend_elements.append(patches.Patch(color=np.array(color)/255, label=name))
    ax4.legend(handles=legend_elements, loc='center', fontsize=12, title='Class Legend', title_fontsize=14)
    ax4.set_title('Color Legend', fontsize=14, fontweight='bold')
    
    # 第二行：前3类置信度分析
    confidence_titles = [
        'Background Confidence',
        'Left',
        'Right'
    ]
    
    for i in range(3):
        ax = fig.add_subplot(gs[1, i])
        im = ax.imshow(pred_probs[i], cmap='hot', vmin=0, vmax=1)
        ax.set_title(confidence_titles[i], fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('Confidence', fontsize=10)
    
    # 第二行右侧：统计信息
    ax = fig.add_subplot(gs[1, 3])
    ax.axis('off')
    
    # 计算最大置信度用于统计
    max_confidence = np.max(pred_probs, axis=0)
    
    stats_text = "Detailed Statistics:\n\n"
    for cls, count in zip(unique, counts):
        percentage = count / total_pixels * 100
        avg_conf = pred_probs[cls][pred_mask == cls].mean() if count > 0 else 0
        stats_text += f"{class_names[cls]}:\n"
        stats_text += f"  Pixels: {count:,} ({percentage:.2f}%)\n"
        stats_text += f"  Avg Confidence: {avg_conf:.3f}\n\n"
    
    # 添加整体置信度统计
    overall_conf = max_confidence.mean()
    stats_text += f"Overall Avg Confidence: {overall_conf:.3f}\n"
    stats_text += f"Image Size: {original_size[1]}×{original_size[0]}"
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax.set_title('Statistics', fontsize=14, fontweight='bold')
    
    # 第三行：病灶类别置信度分析
    lesion_titles = [
        'sdxj',
        'sdbb', 
        'rtzl'
    ]
    
    for i in range(3):
        ax = fig.add_subplot(gs[2, i])
        lesion_idx = i + 3  # 病灶类别从索引3开始
        im = ax.imshow(pred_probs[lesion_idx], cmap='hot', vmin=0, vmax=1)
        ax.set_title(lesion_titles[i], fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('Confidence', fontsize=10)
    
    # 第三行右侧：病灶统计
    ax = fig.add_subplot(gs[2, 3])
    ax.axis('off')
    
    lesion_stats_text = "Lesion Analysis:\n\n"
    lesion_detected = False
    
    for cls in [3, 4, 5]:  # 病灶类别
        if cls in unique:
            count = counts[unique == cls][0]
            percentage = count / total_pixels * 100
            avg_conf = pred_probs[cls][pred_mask == cls].mean()
            
            lesion_stats_text += f"{class_names[cls]}:\n"
            lesion_stats_text += f"  Area: {percentage:.3f}%\n"
            lesion_stats_text += f"  Confidence: {avg_conf:.3f}\n\n"
            
            if percentage > 0.01:  # 如果面积大于0.01%
                lesion_detected = True
    
    if not lesion_detected:
        lesion_stats_text += "No significant lesions detected.\n"
    
    ax.text(0.05, 0.95, lesion_stats_text, transform=ax.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax.set_title('Lesion Detection', fontsize=14, fontweight='bold')
    
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
        image_path = "autodl-tmp/SAM/data/test/images/声带白斑中重_冯润虎133004001875398_20160301_011014090.jpg"
    
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
            print("🔥 SAM ViT-B VOCAL FOLD SEGMENTATION ANALYSIS - 老哥优化版")
            print("="*80)
            print(f"📂 Model: {model_path}")
            print(f"🖼️  Image: {image_path}")
            print("\n📋 Understanding the Results:")
            print("• Row 1: Original image, segmentation result, overlay, and color legend")
            print("• Row 2: Background and vocal fold confidence maps + statistics")
            print("• Row 3: Lesion confidence maps + lesion detection summary")
            print("• Color coding: Red=Left fold, Green=Right fold, Yellow=sdxj, Magenta=sdbb, Cyan=rtzl")
            print("• Confidence interpretation: Bright=high confidence, Dark=low confidence")
            print("="*80)
            
            test_single_image_sam(model_path, image_path)
            print("✅ SAM segmentation analysis completed successfully!")
        except Exception as e:
            print(f"❌ Error during testing: {e}")
            import traceback
            traceback.print_exc() 