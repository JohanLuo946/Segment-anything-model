#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM通用病灶批量测试脚本 - 测试集整体指标计算
功能：加载训练好的模型，对整个测试集进行预测，计算病灶Dice/IoU和整体mDice/mIoU。
自动过滤测试集，只保留包含指定ID的图像，并统计过滤后数量。
不进行可视化，只输出指标结果。
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from segment_anything import sam_model_registry
from collections import defaultdict

# ===== 💪 配置类（与训练脚本一致） =====
class TestConfig:
    IMAGE_SIZE = 1024
    NUM_CLASSES = 2
    BATCH_SIZE = 1  
    NUM_WORKERS = 4
    SAM_MODEL_TYPE = "vit_b"
    PIXEL_MEAN = [123.675, 116.28, 103.53]
    PIXEL_STD = [58.395, 57.12, 57.375]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    LESION_ID = 29  
    LESION_NAME = "声带白斑"  
    
    ID_MAPPING = {
        0: 0,          
        LESION_ID: 1,  
    }
    CLASS_NAMES = ["背景", LESION_NAME]
    
    LESION_CLASSES = [1]  # 病灶类别

config = TestConfig()

class DSCEnhancedSAMModel(torch.nn.Module):
    """DSC增强SAM模型 - 与训练脚本一致"""
    
    def __init__(self, sam_model, num_classes):
        super().__init__()
        self.sam = sam_model
        self.num_classes = num_classes
        
        # 增强的分割头 - 为DSC优化
        self.segmentation_head = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout2d(0.1),  # 添加dropout防过拟合
            
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout2d(0.05),
            
            # 添加残差连接
            torch.nn.Conv2d(64, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.Conv2d(32, num_classes, kernel_size=1)
        )
        
        # 增强注意力机制
        self.attention = torch.nn.Sequential(
            torch.nn.Conv2d(256, 64, kernel_size=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 16, kernel_size=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 1, kernel_size=1),
            torch.nn.Sigmoid()
        )
        
        # 增强边界细化模块 - 专门对抗过度分割
        self.boundary_refine = torch.nn.Sequential(
            torch.nn.Conv2d(num_classes, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, num_classes, kernel_size=1),
            torch.nn.Tanh()  # 使用tanh限制输出范围
        )
        
        # 增强边界收缩模块 - 专门对抗HD=52.26px严重偏移
        self.boundary_contract = torch.nn.Sequential(
            torch.nn.Conv2d(num_classes, 16, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 8, kernel_size=3, padding=1),
            torch.nn.Sigmoid(),
            torch.nn.Conv2d(8, num_classes, kernel_size=1)
        )
    
    def forward(self, images):
        batch_size = images.shape[0]
        
        image_embeddings = self.sam.image_encoder(images)
        
        attention_map = self.attention(image_embeddings)
        enhanced_features = image_embeddings * attention_map
        
        segmentation_logits = self.segmentation_head(enhanced_features)
        
        # 边界细化和收缩 - 激进优化边界精确性
        boundary_refinement = self.boundary_refine(segmentation_logits)
        refined_logits = segmentation_logits + 0.06 * boundary_refinement  # 进一步降低细化权重
        
        # 强化边界收缩 - 对抗HD=52.26px严重偏移 
        boundary_contraction = self.boundary_contract(refined_logits)
        refined_logits = refined_logits - 0.12 * boundary_contraction  # 大幅增加收缩力度 (0.05→0.12)
        
        refined_logits = F.interpolate(
            refined_logits,
            size=(images.shape[2], images.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        
        return refined_logits

# ===== 🎯 测试数据集类（简化版，无增强/采样） =====
class TestDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"找到 {len(self.image_files)} 个测试图像")
        
        # 自动过滤，只保留包含指定LESION_ID的图像
        self.filter_lesion_only()
    
    def filter_lesion_only(self):
        """过滤测试集，只保留掩码中包含指定LESION_ID的图像，并统计数量"""
        filtered_files = []
        for file in tqdm(self.image_files, desc=f"过滤测试集（只保留含ID={config.LESION_ID}的图像）"):
            mask_file = file.replace('.jpg', '.png').replace('.jpeg', '.png')
            mask_path = os.path.join(self.masks_dir, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None and config.LESION_ID in np.unique(mask):
                filtered_files.append(file)
        
        original_count = len(self.image_files)
        self.image_files = filtered_files
        filtered_count = len(self.image_files)
        
        print(f"过滤前测试集数量: {original_count}")
        print(f"过滤后测试集数量（含ID={config.LESION_ID}）: {filtered_count}")
        if filtered_count == 0:
            print(f"警告：过滤后测试集为空！请检查数据中是否包含ID={config.LESION_ID}。")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask_file = image_file.replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(self.masks_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 应用ID映射（其他类转为背景）
        mapped_mask = np.zeros_like(mask)
        for original_id, new_id in config.ID_MAPPING.items():
            mapped_mask[mask == original_id] = new_id
        unknown_mask = ~np.isin(mask, list(config.ID_MAPPING.keys()))
        mapped_mask[unknown_mask] = 0
        
        # 调整大小
        image = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
        mask = cv2.resize(mapped_mask, (config.IMAGE_SIZE, config.IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        
        # 转换为tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()
        
        # 标准化
        mean = torch.tensor(config.PIXEL_MEAN).view(3, 1, 1) / 255.0
        std = torch.tensor(config.PIXEL_STD).view(3, 1, 1) / 255.0
        image = (image - mean) / std
        
        return image, mask

# ===== 📊 评估指标计算类 =====
class ComprehensiveMetrics:
    """综合评估指标计算类 - 支持2类评估"""
    
    def __init__(self):
        self.class_dices = defaultdict(list)
        self.class_ious = defaultdict(list)
    
    def compute_metrics(self, pred: np.ndarray, target: np.ndarray, cls_id: int) -> tuple:
        """计算单个类的Dice和IoU"""
        pred_mask = (pred == cls_id)
        target_mask = (target == cls_id)
        
        intersection = np.logical_and(pred_mask, target_mask).sum()
        pred_sum = pred_mask.sum()
        target_sum = target_mask.sum()
        
        if pred_sum + target_sum == 0:
            return 1.0, 1.0
        
        if pred_sum == 0 or target_sum == 0:
            return 0.0, 0.0
        
        dice = 2.0 * intersection / (pred_sum + target_sum)
        iou = intersection / (pred_sum + target_sum - intersection)
        
        return float(dice), float(iou)
    
    def update(self, pred: np.ndarray, target: np.ndarray):
        """更新评估指标"""
        for cls_id in range(config.NUM_CLASSES):
            dice, iou = self.compute_metrics(pred, target, cls_id)
            self.class_dices[cls_id].append(dice)
            self.class_ious[cls_id].append(iou)
    
    def get_metrics(self):
        metrics = {
            'per_class': {},
            'overall': {}
        }
        
        # 每个类的平均
        for cls_id in range(config.NUM_CLASSES):
            if self.class_dices[cls_id]:
                mean_dice = np.mean(self.class_dices[cls_id])
                mean_iou = np.mean(self.class_ious[cls_id])
            else:
                mean_dice = 0.0
                mean_iou = 0.0
            
            metrics['per_class'][cls_id] = {
                'dice': mean_dice,
                'iou': mean_iou
            }
        
        # 整体 mDice 和 mIoU (所有类的平均)
        mdice = np.mean([metrics['per_class'][cls]['dice'] for cls in range(config.NUM_CLASSES)])
        miou = np.mean([metrics['per_class'][cls]['iou'] for cls in range(config.NUM_CLASSES)])
        
        metrics['overall'] = {
            'mDice': mdice,
            'mIoU': miou
        }
        
        # 病灶特定 (class 1)
        metrics['lesion'] = metrics['per_class'].get(1, {'dice': 0.0, 'iou': 0.0})
        
        return metrics

# ===== 🔍 加载模型 =====
def load_model(model_path, sam_checkpoint):
    sam = sam_model_registry[config.SAM_MODEL_TYPE](checkpoint=sam_checkpoint)
    model = DSCEnhancedSAMModel(sam, config.NUM_CLASSES)
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.DEVICE)
    model.eval()
    print(f"模型加载成功: {model_path}")
    return model

# ===== 主函数：批量测试并计算指标 =====
def main():
    # 在IDE中编辑以下路径变量，然后直接运行脚本
    test_images_dir = "/root/autodl-tmp/SAM/12classes_lesion/test/images"  
    test_masks_dir = "/root/autodl-tmp/SAM/12classes_lesion/test/masks"    
    model_path = "/root/autodl-tmp/SAM/results/models/dsc_enhanced_sdbb_4/models/best_model_overall_dice.pth"  
    sam_checkpoint = "/root/autodl-tmp/SAM/pre_models/sam_vit_b_01ec64.pth"  
    
    # 加载模型
    model = load_model(model_path, sam_checkpoint)
    
    # 加载数据集
    test_dataset = TestDataset(test_images_dir, test_masks_dir)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=False)
    
    # 初始化评估器
    metrics_calc = ComprehensiveMetrics()
    
    print(f"\n开始测试，共 {len(test_dataset)} 个样本...")
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="测试进度"):
            images = images.to(config.DEVICE)
            masks = masks.cpu().numpy()  # [B, H, W]
            
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()  # [B, H, W]
            
            for i in range(len(preds)):
                pred = preds[i]
                target = masks[i]
                
                # 更新评估指标
                metrics_calc.update(pred, target)
    
    # 获取评估结果
    results = metrics_calc.get_metrics()
    
    print(f"\n{'='*60}")
    print(f"🎯 测试结果总结")
    print(f"{'='*60}")
    
    # 病灶指标
    print(f"\n📊 病灶 ({config.LESION_NAME}) 指标:")
    print(f"  Dice: {results['lesion']['dice']:.4f}")
    print(f"  IoU: {results['lesion']['iou']:.4f}")
    
    # 整体指标
    print(f"\n🏆 整体平均指标:")
    print(f"  mDice: {results['overall']['mDice']:.4f}")
    print(f"  mIoU: {results['overall']['mIoU']:.4f}")
    
    print(f"\n{'='*60}")
    print(f"测试完成！总样本数: {len(test_dataset)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()