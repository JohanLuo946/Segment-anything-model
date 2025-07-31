#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM通用病灶批量测试脚本 - 测试集整体Dice计算 
功能：加载训练好的模型，对整个测试集进行预测，计算平均Dice分数（背景和指定病灶类）。
采用test_fold_Dice_12class.py的计算逻辑，适应2类。
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
    """DSC增强SAM模型 - 与train_2classes_v2.py完全一致"""
    
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
        
        # 边界细化模块
        self.boundary_refine = torch.nn.Sequential(
            torch.nn.Conv2d(num_classes, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, num_classes, kernel_size=1),
            torch.nn.Tanh()  # 使用tanh限制输出范围
        )
    
    def forward(self, images):
        batch_size = images.shape[0]
        
        image_embeddings = self.sam.image_encoder(images)
        
        attention_map = self.attention(image_embeddings)
        enhanced_features = image_embeddings * attention_map
        
        segmentation_logits = self.segmentation_head(enhanced_features)
        
        # 边界细化
        boundary_refinement = self.boundary_refine(segmentation_logits)
        refined_logits = segmentation_logits + 0.1 * boundary_refinement
        
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
        
        # 新增：自动过滤，只保留包含指定LESION_ID的图像
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

# ===== 📊 评估指标计算类（适应2类，基于test_fold_Dice_12class.py） =====
class ComprehensiveMetrics:
    """综合评估指标计算类 - 支持2类评估"""
    
    def __init__(self):
        # 多维度评估指标
        self.pure_lesion_metrics = defaultdict(list)   # 纯病灶评估：只考虑病灶区域的指标
        
        # 新增：每张图像的整体micro和macro指标
        self.whole_image_dice_list = []  # 整张图像的Dice
        self.whole_image_lesion_dice_list = []  # 整张图像的病灶Dice
    
    def compute_metrics(self, pred: np.ndarray, target: np.ndarray) -> tuple:
        """计算Dice和IoU"""
        intersection = np.logical_and(pred, target).sum()
        pred_sum = pred.sum()
        target_sum = target.sum()
        
        if pred_sum + target_sum == 0:
            return 1.0, 1.0  # 完全匹配
        
        if pred_sum == 0 or target_sum == 0:
            return 0.0, 0.0  # 完全不匹配
            
        dice = 2.0 * intersection / (pred_sum + target_sum)
        union = pred_sum + target_sum - intersection
        iou = intersection / union
        
        return float(dice), float(iou)
    
    def compute_pure_lesion_metrics(self, pred: np.ndarray, target: np.ndarray, lesion_mask: np.ndarray) -> tuple:
        """计算纯病灶区域的Dice和IoU"""
        if not lesion_mask.any():
            return 0.0, 0.0
            
        pred_lesion = pred[lesion_mask]
        target_lesion = target[lesion_mask]
        
        return self.compute_metrics(pred_lesion, target_lesion)
    
    def update(self, pred: np.ndarray, target: np.ndarray):
        """更新评估指标"""
        # 获取实际存在的类别（包括背景）
        present_classes = np.unique(target)
        
        # 创建病灶区域掩码（所有病灶类别的并集）
        lesion_mask = np.zeros_like(target, dtype=bool)
        for cls_id in config.LESION_CLASSES:
            if cls_id in present_classes:
                lesion_mask |= (target == cls_id)
        
        # 计算每个存在类别的指标
        total_inter = 0
        total_pred = 0
        total_gt = 0
        
        for cls_id in present_classes:
            pred_mask = (pred == cls_id)
            target_mask = (target == cls_id)
            
            # 用于whole_image_dice (micro)
            inter = np.logical_and(pred_mask, target_mask).sum()
            p_sum = pred_mask.sum()
            g_sum = target_mask.sum()
            total_inter += inter
            total_pred += p_sum
            total_gt += g_sum
            
            # 2. 如果是病灶类别，计算额外指标
            if cls_id in config.LESION_CLASSES and target_mask.sum() > 0:
                # 计算纯病灶区域的指标
                pure_dice, pure_iou = self.compute_pure_lesion_metrics(
                    pred_mask, target_mask, lesion_mask)
                self.pure_lesion_metrics[int(cls_id)].append({
                    'dice': pure_dice,
                    'iou': pure_iou
                })
        
        # 计算整张图像的dice
        if present_classes.size > 0:
            # whole_image_dice (micro)
            if total_pred + total_gt > 0:
                whole_image_dice = 2.0 * total_inter / (total_pred + total_gt)
            else:
                whole_image_dice = 1.0
            self.whole_image_dice_list.append(whole_image_dice)
        
        lesion_present = [cls for cls in present_classes if cls in config.LESION_CLASSES]
        if lesion_present:
            lesion_inter = 0
            lesion_pred = 0
            lesion_gt = 0
            
            for cls_id in lesion_present:
                pred_mask = (pred == cls_id)
                target_mask = (target == cls_id)
                inter = np.logical_and(pred_mask, target_mask).sum()
                p_sum = pred_mask.sum()
                g_sum = target_mask.sum()
                lesion_inter += inter
                lesion_pred += p_sum
                lesion_gt += g_sum
            
            # whole_image_lesion_dice (micro for lesions)
            if lesion_pred + lesion_gt > 0:
                whole_image_lesion_dice = 2.0 * lesion_inter / (lesion_pred + lesion_gt)
            else:
                whole_image_lesion_dice = 1.0
            self.whole_image_lesion_dice_list.append(whole_image_lesion_dice)

    def get_metrics(self):
        metrics = {
            'per_class': {},
            'overall': {},
            'lesion_overall': {}
        }
        
        for cls_id in self.pure_lesion_metrics.keys():
            pure_scores = self.pure_lesion_metrics[cls_id]
            pure_dice = np.mean([s['dice'] for s in pure_scores])
            pure_iou = np.mean([s['iou'] for s in pure_scores])
            pure_dice_std = np.std([s['dice'] for s in pure_scores])
            pure_iou_std = np.std([s['iou'] for s in pure_scores])
            
            metrics['per_class'][cls_id] = {
                'pure_lesion_dice': float(pure_dice),
                'pure_lesion_dice_std': float(pure_dice_std),
                'pure_lesion_iou': float(pure_iou),
                'pure_lesion_iou_std': float(pure_iou_std),
            }
        
        metrics['overall'] = {
            'whole_image_dice': float(np.mean(self.whole_image_dice_list)) if self.whole_image_dice_list else 0.0,
            'whole_image_dice_std': float(np.std(self.whole_image_dice_list)) if self.whole_image_dice_list else 0.0,
        }
        
        metrics['lesion_overall'] = {
            'whole_image_lesion_dice': float(np.mean(self.whole_image_lesion_dice_list)) if self.whole_image_lesion_dice_list else 0.0,
            'whole_image_lesion_dice_std': float(np.std(self.whole_image_lesion_dice_list)) if self.whole_image_lesion_dice_list else 0.0,
        }
        
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

# ===== 主函数：批量测试并计算平均Dice =====
def main():
    # 在IDE中编辑以下路径变量，然后直接运行脚本
    test_images_dir = "/root/autodl-tmp/SAM/sdbb/test/images"  
    test_masks_dir = "/root/autodl-tmp/SAM/sdbb/test/masks"    
    model_path = "/root/autodl-tmp/SAM/results/models/dsc_enhanced_sdbb_4/models/best_model_lesion_dice.pth"  
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
    print(f"🎯 DSC测试结果总结")
    print(f"{'='*60}")
    
    # 显示病灶类别指标
    for class_id in config.LESION_CLASSES:
        if class_id in results['per_class']:
            cls_results = results['per_class'][class_id]
            
            print(f"\n📊 {config.LESION_NAME} 类别指标:")
            print(f"  pure_lesion_dice: {cls_results['pure_lesion_dice']:.4f} ± {cls_results['pure_lesion_dice_std']:.4f}")
            print(f"  pure_lesion_iou:  {cls_results['pure_lesion_iou']:.4f} ± {cls_results['pure_lesion_iou_std']:.4f}")
    
    # 整体指标
    print(f"\n🏆 整体平均指标:")
    print(f"  whole_image_dice: {results['overall']['whole_image_dice']:.4f} ± {results['overall']['whole_image_dice_std']:.4f}")
    
    # 病灶专项指标
    print(f"\n🎯 病灶专项表现:")
    print(f"  whole_image_lesion_dice: {results['lesion_overall']['whole_image_lesion_dice']:.4f} ± {results['lesion_overall']['whole_image_lesion_dice_std']:.4f}")
    
    print(f"\n{'='*60}")
    print(f"测试完成！总样本数: {len(test_dataset)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()