#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔥 SAM声带病灶分割 - 老哥定制优化版本 🔥
专治类别不平衡！小目标分割杀手锏！
作者：小柯（资深技术老哥）
"""

import os
import sys
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
import logging
from tqdm import tqdm
import gc
from collections import defaultdict, Counter

# 设置日志 - 老哥风格
logging.basicConfig(level=logging.INFO, format='%(asctime)s -  - %(message)s')
logger = logging.getLogger(__name__)

# ===== 💪 硬核优化配置 =====
class OptimizedConfig:
    """老哥的杀手级配置 - 专治类别不平衡！"""
    
    # 路径配置
    TRAIN_IMAGES_DIR = "/root/autodl-tmp/SAM/data/train/images"
    TRAIN_MASKS_DIR = "/root/autodl-tmp/SAM/data/train/masks"
    VAL_IMAGES_DIR = "/root/autodl-tmp/SAM/data/val/images"
    VAL_MASKS_DIR = "/root/autodl-tmp/SAM/data/val/masks"
    SAM_MODEL_PATH = "/root/autodl-tmp/SAM/pre_models/sam_vit_b_01ec64.pth"  # 使用ViT-b模型
    RESULTS_DIR = "/root/autodl-tmp/SAM/results/models/run_2"
    
    # 基础配置
    NUM_CLASSES = 6
    IMAGE_SIZE = 1024
    BATCH_SIZE = 2      
    NUM_WORKERS = 4
    
    # 类别映射 - 老哥指定的正确映射
    ID_MAPPING = {
        0: 0,    # 背景
        170: 1,  # 左声带
        184: 2,  # 右声带
        105: 3,  # 声带小结
        23: 4,   # 声带白斑
        146: 5,  # 声带乳头状瘤
    }
    
    CLASS_NAMES = [
        "背景", "左声带", "右声带", "声带小结", "声带白斑", "声带乳头状瘤"
    ]
    
    # 🔥 动态权重策略 - 根据类别难度自适应调整
    INITIAL_CLASS_WEIGHTS = [0.1, 1.0, 1.0, 30.0, 35.0, 32.0]  # 病灶权重爆炸式提升！
    DYNAMIC_WEIGHT_UPDATE = True    # 开启动态权重调整
    
    # 训练配置 - 老哥优化版
    NUM_EPOCHS = 150         # 多训练一些epoch，给病灶学习时间
    LEARNING_RATE = 2e-4    # 稍微提高学习率
    WEIGHT_DECAY = 1e-4
    GRADIENT_ACCUMULATION_STEPS = 2  # 减少梯度累积，更频繁更新
    
    # SAM配置
    SAM_MODEL_TYPE = "vit_b"
    PIXEL_MEAN = [123.675, 116.28, 103.53]
    PIXEL_STD = [58.395, 57.12, 57.375]
    
    # 设备配置
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MIXED_PRECISION = True
    
    # 🚀 优化策略开关
    USE_FOCAL_LOSS = True           # Focal Loss 专治难分类
    USE_DICE_LOSS = True            # Dice Loss 专治小目标
    USE_SMART_SAMPLING = True       # 智能采样，病灶样本优先
    USE_MULTI_SCALE_LOSS = False    # 多尺度损失 - SAM不支持，暂时关闭
    USE_LABEL_SMOOTHING = True      # 标签平滑，防过拟合
    USE_EDGE_LOSS = True            # 🔥 边缘感知损失 - 新增！
    USE_BOUNDARY_LOSS = True        # 🎯 边界损失 - 新增！
    
    # Focal Loss 参数
    FOCAL_ALPHA = [0.1, 1.0, 1.0, 4.0, 5.0, 4.5]  # 各类别focal权重
    FOCAL_GAMMA = 2.0               # 难度关注参数
    
    # 智能采样参数
    LESION_OVERSAMPLE_FACTOR = 5.0  # 病灶样本过采样倍数
    BACKGROUND_UNDERSAMPLE_FACTOR = 0.3  # 背景样本欠采样
    
    # 多尺度训练参数
    MULTI_SCALE_SIZES = [512, 768, 1024]  # 多尺度训练尺寸
    SCALE_CHANGE_FREQUENCY = 10     # 每10个epoch切换一次尺度
    
    # 显存优化
    CLEAR_CACHE_EVERY = 3
    PIN_MEMORY = False
    
    # 保存和评估
    SAVE_EVERY = 10
    EVAL_EVERY = 1          # 更频繁验证，及时发现问题
    EARLY_STOPPING_PATIENCE = 20

config = OptimizedConfig()

# ===== 🎯 智能数据集类 - 老哥特制 =====
class SmartVocalFoldDataset(Dataset):
    """老哥的智能数据集 - 专门照顾病灶样本！"""
    
    def __init__(self, images_dir, masks_dir, is_train=True, current_epoch=0):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.is_train = is_train
        self.current_epoch = current_epoch
        
        # 获取图像文件
        self.image_files = []
        for file in os.listdir(images_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                mask_file = file.replace('.jpg', '.png').replace('.jpeg', '.png')
                mask_path = os.path.join(masks_dir, mask_file)
                if os.path.exists(mask_path):
                    self.image_files.append(file)
        
        logger.info(f"找到 {len(self.image_files)} 个图像-掩码对")
        
        # 🔥 分析样本分布，制定采样策略
        if is_train and config.USE_SMART_SAMPLING:
            self.analyze_sample_distribution()
    
    def analyze_sample_distribution(self):
        """分析样本分布，老哥要知己知彼！"""
        logger.info("老哥正在分析样本分布...")
        
        self.sample_weights = []
        lesion_samples = 0
        background_only_samples = 0
        
        for image_file in tqdm(self.image_files, desc="分析样本"):
            mask_file = image_file.replace('.jpg', '.png').replace('.jpeg', '.png')
            mask_path = os.path.join(self.masks_dir, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # 应用ID映射
            mapped_mask = self.apply_id_mapping(mask)
            
            # 检查是否包含病灶（类别3,4,5）
            unique_ids = np.unique(mapped_mask)
            has_lesion = any(id in [3, 4, 5] for id in unique_ids)
            
            if has_lesion:
                # 包含病灶的样本，权重爆炸！
                weight = config.LESION_OVERSAMPLE_FACTOR
                lesion_samples += 1
            elif len(unique_ids) == 1 and unique_ids[0] == 0:
                # 纯背景样本，权重削减
                weight = config.BACKGROUND_UNDERSAMPLE_FACTOR
                background_only_samples += 1
            else:
                # 普通样本，正常权重
                weight = 1.0
            
            self.sample_weights.append(weight)
        
        logger.info(f"老哥分析完毕：")
        logger.info(f"  🔥 病灶样本: {lesion_samples} 个 (权重x{config.LESION_OVERSAMPLE_FACTOR})")
        logger.info(f"  😴 纯背景样本: {background_only_samples} 个 (权重x{config.BACKGROUND_UNDERSAMPLE_FACTOR})")
        logger.info(f"  😊 普通样本: {len(self.image_files)-lesion_samples-background_only_samples} 个")
    
    def get_weighted_sampler(self):
        """获取智能采样器 - 老哥的秘密武器"""
        if hasattr(self, 'sample_weights'):
            return WeightedRandomSampler(
                weights=self.sample_weights,
                num_samples=len(self.sample_weights),
                replacement=True
            )
        return None
    
    def apply_id_mapping(self, mask):
        """应用ID映射转换"""
        mapped_mask = np.zeros_like(mask)
        for original_id, new_id in config.ID_MAPPING.items():
            mapped_mask[mask == original_id] = new_id
        
        # 未知ID映射为背景
        unknown_mask = np.ones_like(mask, dtype=bool)
        for original_id in config.ID_MAPPING.keys():
            unknown_mask &= (mask != original_id)
        mapped_mask[unknown_mask] = 0
        
        return mapped_mask
    
    def smart_augmentation(self, image, mask):
        """老哥的智能增强 - 专门照顾病灶"""
        if not self.is_train:
            return image, mask
        
        # 检查是否包含病灶
        unique_ids = np.unique(mask)
        has_lesion = any(id in [3, 4, 5] for id in unique_ids)
        
        if has_lesion:
            # 病灶样本，温和增强，保护细节
            if random.random() < 0.3:
                # 小幅旋转
                angle = random.uniform(-5, 5)
                h, w = image.shape[:2]
                center = (w//2, h//2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(image, M, (w, h))
                mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
            
            if random.random() < 0.4:
                # 亮度微调
                factor = random.uniform(0.9, 1.1)
                image = np.clip(image * factor, 0, 255).astype(np.uint8)
        else:
            # 非病灶样本，可以更激进的增强
            if random.random() < 0.5:
                # 更大范围旋转
                angle = random.uniform(-10, 10)
                h, w = image.shape[:2]
                center = (w//2, h//2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(image, M, (w, h))
                mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
        
        return image, mask
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 🔥 SAM模型要求固定尺寸1024x1024
        current_size = config.IMAGE_SIZE
        
        # 加载图像
        image_file = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 加载掩码
        mask_file = image_file.replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(self.masks_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 应用ID映射
        mask = self.apply_id_mapping(mask)
        
        # 智能增强
        image, mask = self.smart_augmentation(image, mask)
        
        # 调整尺寸
        image = cv2.resize(image, (current_size, current_size))
        mask = cv2.resize(mask, (current_size, current_size), interpolation=cv2.INTER_NEAREST)
        
        # 转换为tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()
        
        # 标准化
        mean = torch.tensor(config.PIXEL_MEAN).view(3, 1, 1) / 255.0
        std = torch.tensor(config.PIXEL_STD).view(3, 1, 1) / 255.0
        image = (image - mean) / std
        
        return image, mask, image_file

# ===== 🔥 杀手级损失函数 - 老哥特制强化版 =====
class KillerLoss(nn.Module):
    """老哥的杀手级损失函数 - 专治类别不平衡！现在带Sobel边缘检测！"""
    
    def __init__(self, class_weights=None, focal_alpha=None, focal_gamma=2.0):
        super().__init__()
        self.class_weights = class_weights
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # 基础损失函数
        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).float())
        else:
            self.ce_loss = nn.CrossEntropyLoss()
        
        # 🔥 Sobel算子 - 专门检测边缘
        self.register_buffer('sobel_x', torch.tensor([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        ], dtype=torch.float32).unsqueeze(0))
        
        self.register_buffer('sobel_y', torch.tensor([
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        ], dtype=torch.float32).unsqueeze(0))
        
        logger.info("杀手级损失函数装配完毕！🔥 现在带边缘检测！")
    
    def sobel_edge_detection(self, mask):
        """Sobel算子边缘检测 - 老哥的边缘杀手锏"""
        # 转换为float并添加batch和channel维度
        if len(mask.shape) == 3:  # [B, H, W]
            mask = mask.unsqueeze(1).float()  # [B, 1, H, W]
        elif len(mask.shape) == 2:  # [H, W]
            mask = mask.unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]
        
        # 应用Sobel算子
        edge_x = F.conv2d(mask, self.sobel_x, padding=1)
        edge_y = F.conv2d(mask, self.sobel_y, padding=1)
        
        # 计算边缘强度
        edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2)
        
        # 二值化边缘（阈值可调）
        edge_binary = (edge_magnitude > 0.1).float()
        
        return edge_binary.squeeze(1)  # 移除channel维度
    
    def edge_aware_loss(self, predictions, targets):
        """边缘感知损失 - 专门优化病灶边缘（修复混合精度兼容性）"""
        # 只对病灶类别计算边缘损失
        lesion_classes = [3, 4, 5]  # 声带小结、白斑、乳头状瘤
        edge_loss = torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype)
        lesion_count = 0
        
        # 使用logits而不是softmax，避免混合精度问题
        predictions_logits = predictions  # 直接使用logits
        
        for lesion_class in lesion_classes:
            # 获取该病灶类别的真实掩码和预测logits
            target_class = (targets == lesion_class).float()
            pred_logits = predictions_logits[:, lesion_class]  # 使用logits
            
            # 检查是否存在该病灶
            if target_class.sum() > 0:
                # 计算真实边缘
                target_edges = self.sobel_edge_detection(target_class)
                
                # 在边缘区域计算损失
                edge_mask = target_edges > 0
                if edge_mask.sum() > 0:
                    # 🔥 使用binary_cross_entropy_with_logits，混合精度安全
                    edge_pred_loss = F.binary_cross_entropy_with_logits(
                        pred_logits[edge_mask], 
                        target_class[edge_mask],
                        reduction='mean'
                    )
                    edge_loss = edge_loss + edge_pred_loss
                    lesion_count += 1
        
        # 确保返回tensor
        if lesion_count > 0:
            return edge_loss / lesion_count
        else:
            return edge_loss
    
    def boundary_loss(self, predictions, targets):
        """边界损失 - 让病灶边界更清晰（修复混合精度兼容性）"""
        # 使用sigmoid而不是softmax，更适合边缘检测
        predictions_sigmoid = torch.sigmoid(predictions)
        boundary_loss = torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype)
        lesion_count = 0
        
        lesion_classes = [3, 4, 5]
        
        for lesion_class in lesion_classes:
            target_class = (targets == lesion_class).float()
            
            if target_class.sum() > 0:
                pred_class = predictions_sigmoid[:, lesion_class]
                
                # 计算预测的边缘
                pred_edges = self.sobel_edge_detection(pred_class)
                target_edges = self.sobel_edge_detection(target_class)
                
                # 边缘一致性损失
                edge_consistency = F.mse_loss(pred_edges, target_edges)
                boundary_loss = boundary_loss + edge_consistency
                lesion_count += 1
        
        # 确保返回tensor
        if lesion_count > 0:
            return boundary_loss / lesion_count
        else:
            return boundary_loss
    
    def focal_loss(self, predictions, targets):
        """Focal Loss - 专治难分类样本"""
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # 计算alpha权重
        if self.focal_alpha is not None:
            alpha_t = torch.tensor(self.focal_alpha).to(predictions.device)[targets]
        else:
            alpha_t = 1.0
        
        # Focal Loss公式
        focal_loss = alpha_t * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()
    
    def dice_loss(self, predictions, targets):
        """Dice Loss - 专治小目标"""
        smooth = 1e-5
        predictions = torch.softmax(predictions, dim=1)
        
        dice_loss = 0
        for class_idx in range(config.NUM_CLASSES):
            pred_class = predictions[:, class_idx]
            target_class = (targets == class_idx).float()
            
            intersection = (pred_class * target_class).sum()
            union = pred_class.sum() + target_class.sum()
            
            dice = (2 * intersection + smooth) / (union + smooth)
            dice_loss += 1 - dice
        
        return dice_loss / config.NUM_CLASSES
    
    def forward(self, predictions, targets):
        loss_dict = {}
        total_loss = 0
        
        # 1. CrossEntropy Loss (基础)
        if config.USE_LABEL_SMOOTHING:
            # 标签平滑，防过拟合
            ce_loss = F.cross_entropy(predictions, targets, label_smoothing=0.1)
        else:
            ce_loss = self.ce_loss(predictions, targets)
        loss_dict['ce_loss'] = ce_loss.item()
        total_loss += 0.3 * ce_loss
        
        # 2. Focal Loss (难分类样本)
        if config.USE_FOCAL_LOSS:
            focal_loss = self.focal_loss(predictions, targets)
            loss_dict['focal_loss'] = focal_loss.item()
            total_loss += 0.3 * focal_loss
        
        # 3. Dice Loss (小目标)
        if config.USE_DICE_LOSS:
            dice_loss = self.dice_loss(predictions, targets)
            loss_dict['dice_loss'] = dice_loss.item()
            total_loss += 0.2 * dice_loss
        
        # 🔥 4. Edge-Aware Loss (边缘感知) - 新增！
        if config.USE_EDGE_LOSS:
            try:
                edge_loss = self.edge_aware_loss(predictions, targets)
                if isinstance(edge_loss, torch.Tensor):
                    loss_dict['edge_loss'] = edge_loss.item()
                    total_loss += 0.1 * edge_loss
                else:
                    loss_dict['edge_loss'] = float(edge_loss)
                    total_loss += 0.1 * torch.tensor(edge_loss, device=predictions.device)
            except Exception as e:
                # 如果边缘损失计算失败，记录但不中断训练
                loss_dict['edge_loss'] = 0.0
                logger.warning(f"边缘损失计算失败: {e}")
        else:
            loss_dict['edge_loss'] = 0.0
        
        # 🎯 5. Boundary Loss (边界损失) - 新增！
        if config.USE_BOUNDARY_LOSS:
            try:
                boundary_loss = self.boundary_loss(predictions, targets)
                if isinstance(boundary_loss, torch.Tensor):
                    loss_dict['boundary_loss'] = boundary_loss.item()
                    total_loss += 0.1 * boundary_loss
                else:
                    loss_dict['boundary_loss'] = float(boundary_loss)
                    total_loss += 0.1 * torch.tensor(boundary_loss, device=predictions.device)
            except Exception as e:
                # 如果边界损失计算失败，记录但不中断训练
                loss_dict['boundary_loss'] = 0.0
                logger.warning(f"边界损失计算失败: {e}")
        else:
            loss_dict['boundary_loss'] = 0.0
        
        return total_loss, loss_dict

# ===== 🚀 强化SAM模型 - 老哥改装版 =====
class EnhancedSAMModel(nn.Module):
    """老哥的强化SAM模型 - 专治小目标！"""
    
    def __init__(self, sam_model, num_classes):
        super().__init__()
        self.sam = sam_model
        self.num_classes = num_classes
        
        # 🔥 多尺度特征融合分割头
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
        
        # 🎯 注意力模块 - 让模型主动关注病灶
        self.attention = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 冻结SAM的部分参数，只微调关键部分
        self.freeze_sam_components()
        
        logger.info("强化SAM模型改装完毕！专治小目标！🎯")
    
    def freeze_sam_components(self):
        """冻结SAM的大部分参数，只训练必要的部分"""
        # 冻结image_encoder的前面几层
        layers = list(self.sam.image_encoder.children())
        for i, layer in enumerate(layers[:-3]):  # 只解冻最后3层
            for param in layer.parameters():
                param.requires_grad = False
        
        logger.info("SAM参数冻结完毕，只训练关键部分！")
    
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

# ===== 🧠 智能指标计算器 - 老哥定制优化版 =====
class SmartMetrics:
    """老哥的智能指标计算器 - 只计算存在的类别，专门监控病灶学习进度"""
    
    def __init__(self, num_classes, class_names):
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()
    
    def reset(self):
        self.class_ious = np.zeros(self.num_classes)
        self.class_dices = np.zeros(self.num_classes)
        self.class_f1s = np.zeros(self.num_classes)
        self.class_counts = np.zeros(self.num_classes)
        self.total_correct = 0
        self.total_pixels = 0
        
        # 🔥 记录每个batch中实际存在的类别
        self.present_classes_per_batch = []
        
        # 🔥 专门监控病灶进度
        self.lesion_progress = {
            'sdxj': [],  # 声带小结
            'sdbb': [],  # 声带白斑  
            'rtzl': []   # 声带乳头状瘤
        }
    
    def update(self, predictions, targets):
        predictions = torch.argmax(predictions, dim=1)
        
        # 整体准确率
        self.total_correct += (predictions == targets).sum().item()
        self.total_pixels += targets.numel()
        
        # 🔥 记录当前batch中存在的类别
        batch_present_classes = set()
        
        # 各类别IoU、Dice和F1
        for class_idx in range(self.num_classes):
            pred_mask = (predictions == class_idx)
            target_mask = (targets == class_idx)
            
            intersection = (pred_mask & target_mask).sum().item()
            union = (pred_mask | target_mask).sum().item()
            pred_sum = pred_mask.sum().item()
            target_sum = target_mask.sum().item()
            
            # 🎯 只有当类别在真实标签或预测中存在时才计算指标
            if union > 0:
                iou = intersection / union
                self.class_ious[class_idx] += iou
                self.class_counts[class_idx] += 1
                batch_present_classes.add(class_idx)
                
                # Dice计算
                if pred_sum + target_sum > 0:
                    dice = 2 * intersection / (pred_sum + target_sum)
                    self.class_dices[class_idx] += dice
                
                # F1 Score计算
                precision = intersection / (pred_sum + 1e-8)
                recall = intersection / (target_sum + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                self.class_f1s[class_idx] += f1
        
        # 记录当前batch的存在类别
        self.present_classes_per_batch.append(batch_present_classes)
        
        # 🔥 专门监控病灶类别进度
        lesion_map = {3: 'sdxj', 4: 'sdbb', 5: 'rtzl'}
        for class_idx, lesion_name in lesion_map.items():
            if self.class_counts[class_idx] > 0:
                current_iou = self.class_ious[class_idx] / self.class_counts[class_idx]
                self.lesion_progress[lesion_name].append(current_iou)
    
    def compute(self):
        accuracy = self.total_correct / self.total_pixels if self.total_pixels > 0 else 0
        
        # 🎯 优化版本：只计算实际存在的类别
        class_ious = {}
        class_dices = {}
        class_f1s = {}
        
        # 传统方式（包含所有类别）
        traditional_ious = []
        traditional_dices = []
        traditional_f1s = []
        
        # 优化方式（只计算存在的类别）
        optimized_ious = []
        optimized_dices = []
        optimized_f1s = []
        present_class_names = []
        
        for i, name in enumerate(self.class_names):
            if self.class_counts[i] > 0:
                iou = self.class_ious[i] / self.class_counts[i]
                dice = self.class_dices[i] / self.class_counts[i]
                f1 = self.class_f1s[i] / self.class_counts[i]
                
                class_ious[name] = iou
                class_dices[name] = dice
                class_f1s[name] = f1
                
                # 优化方式：只计算存在的类别
                optimized_ious.append(iou)
                optimized_dices.append(dice)
                optimized_f1s.append(f1)
                present_class_names.append(name)
            else:
                class_ious[name] = 0.0
                class_dices[name] = 0.0
                class_f1s[name] = 0.0
            
            # 传统方式：包含所有类别
            traditional_ious.append(class_ious[name])
            traditional_dices.append(class_dices[name])
            traditional_f1s.append(class_f1s[name])
        
        # 计算平均指标
        # 传统方式
        mean_iou_traditional = np.mean(traditional_ious)
        mean_dice_traditional = np.mean(traditional_dices)
        mean_f1_traditional = np.mean(traditional_f1s)
        
        # 优化方式（只计算存在的类别）
        mean_iou_optimized = np.mean(optimized_ious) if optimized_ious else 0
        mean_dice_optimized = np.mean(optimized_dices) if optimized_dices else 0
        mean_f1_optimized = np.mean(optimized_f1s) if optimized_f1s else 0
        
        # 🔥 病灶专项指标
        lesion_ious = []
        lesion_dices = []
        lesion_f1s = []
        lesion_present_names = []
        
        for lesion_idx in [3, 4, 5]:  # 病灶类别
            if self.class_counts[lesion_idx] > 0:
                lesion_name = self.class_names[lesion_idx]
                lesion_ious.append(class_ious[lesion_name])
                lesion_dices.append(class_dices[lesion_name])
                lesion_f1s.append(class_f1s[lesion_name])
                lesion_present_names.append(lesion_name)
        
        # 病灶平均指标
        lesion_miou_optimized = np.mean(lesion_ious) if lesion_ious else 0
        lesion_mdice_optimized = np.mean(lesion_dices) if lesion_dices else 0
        lesion_mf1_optimized = np.mean(lesion_f1s) if lesion_f1s else 0
        
        # 传统病灶指标（包含0值）
        lesion_miou_traditional = np.mean([class_ious[self.class_names[i]] for i in [3, 4, 5]])
        lesion_mdice_traditional = np.mean([class_dices[self.class_names[i]] for i in [3, 4, 5]])
        lesion_mf1_traditional = np.mean([class_f1s[self.class_names[i]] for i in [3, 4, 5]])
        
        # 🔥 病灶专项报告
        lesion_report = {}
        for lesion_name, progress in self.lesion_progress.items():
            if progress:
                lesion_report[lesion_name] = {
                    'current_iou': progress[-1],
                    'trend': 'improving' if len(progress) > 1 and progress[-1] > progress[0] else 'stable'
                }
        
        return {
            # 基础指标
            'accuracy': accuracy,
            
            # 传统指标（包含所有类别）
            'mIoU_traditional': mean_iou_traditional,
            'mDice_traditional': mean_dice_traditional,
            'mF1_traditional': mean_f1_traditional,
            
            # 🎯 优化指标（只计算存在的类别）
            'mIoU': mean_iou_optimized,  # 主要使用优化版本
            'mDice': mean_dice_optimized,
            'mF1': mean_f1_optimized,
            'present_classes': present_class_names,
            
            # 详细类别指标
            'class_ious': class_ious,
            'class_dices': class_dices,
            'class_f1s': class_f1s,
            
            # 病灶专项指标
            'lesion_mIoU_traditional': lesion_miou_traditional,
            'lesion_mDice_traditional': lesion_mdice_traditional,
            'lesion_mF1_traditional': lesion_mf1_traditional,
            'lesion_mIoU': lesion_miou_optimized,  # 主要使用优化版本
            'lesion_mDice': lesion_mdice_optimized,
            'lesion_mF1': lesion_mf1_optimized,
            'lesion_present_names': lesion_present_names,
            
            # 进度报告
            'lesion_report': lesion_report
        }

# ===== 🎮 超级训练器 - 老哥操刀 =====
class SuperTrainer:
    """老哥的超级训练器 - 智能调参，自动优化！"""
    
    def __init__(self):
        self.device = torch.device(config.DEVICE)
        self.scaler = GradScaler() if config.MIXED_PRECISION else None
        
        # 创建结果目录
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        os.makedirs(os.path.join(config.RESULTS_DIR, "models"), exist_ok=True)
        os.makedirs(os.path.join(config.RESULTS_DIR, "logs"), exist_ok=True)
        
        logger.info("🚀 老哥的超级训练器启动！")
        
        # 🔥 训练状态 - 必须在setup_training之前初始化！
        self.best_miou = 0.0
        self.patience_counter = 0
        self.current_weights = config.INITIAL_CLASS_WEIGHTS.copy()
        
        # 初始化组件
        self.setup_model()
        self.setup_data()
        self.setup_training()
        
        # 记录训练历史
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_miou': [], 'val_miou': [],
            'lesion_ious': {'sdxj': [], 'sdbb': [], 'rtzl': []}
        }
    
    def setup_model(self):
        """设置模型"""
        logger.info("老哥正在装配SAM模型...")
        
        try:
            from segment_anything import sam_model_registry
            sam = sam_model_registry[config.SAM_MODEL_TYPE](checkpoint=config.SAM_MODEL_PATH)
            sam.to(self.device)
            
            self.model = EnhancedSAMModel(sam, config.NUM_CLASSES)
            self.model.to(self.device)
            
            # 计算模型参数
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            logger.info(f"SAM模型装配完毕！")
            logger.info(f"  总参数: {total_params:,}")
            logger.info(f"  可训练参数: {trainable_params:,}")
            logger.info(f"  参数冻结率: {100*(1-trainable_params/total_params):.1f}%")
            
        except ImportError:
            logger.error("segment_anything没装，正在安装...")
            os.system("pip install segment-anything")
            from segment_anything import sam_model_registry
            sam = sam_model_registry[config.SAM_MODEL_TYPE](checkpoint=config.SAM_MODEL_PATH)
            sam.to(self.device)
            self.model = EnhancedSAMModel(sam, config.NUM_CLASSES)
            self.model.to(self.device)
    
    def setup_data(self):
        """设置数据加载器"""
        logger.info("老哥正在准备数据...")
        
        # 创建数据集
        self.train_dataset = SmartVocalFoldDataset(
            config.TRAIN_IMAGES_DIR,
            config.TRAIN_MASKS_DIR,
            is_train=True
        )
        
        self.val_dataset = SmartVocalFoldDataset(
            config.VAL_IMAGES_DIR,
            config.VAL_MASKS_DIR,
            is_train=False
        )
        
        # 🔥 智能采样器
        train_sampler = self.train_dataset.get_weighted_sampler()
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.BATCH_SIZE,
            sampler=train_sampler,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )
        
        logger.info(f"数据准备完毕！")
        logger.info(f"  训练样本: {len(self.train_dataset)}")
        logger.info(f"  验证样本: {len(self.val_dataset)}")
        logger.info(f"  智能采样: {'已启用' if train_sampler else '未启用'}")
    
    def setup_training(self):
        """设置训练组件"""
        logger.info("老哥正在配置训练参数...")
        
        # 🔥 分层学习率 - 不同部分用不同学习率
        param_groups = [
            # SAM backbone - 小学习率
            {'params': [p for n, p in self.model.sam.named_parameters() if p.requires_grad], 
             'lr': config.LEARNING_RATE * 0.1},
            # 分割头 - 大学习率  
            {'params': self.model.segmentation_head.parameters(), 
             'lr': config.LEARNING_RATE},
            # 注意力模块 - 中等学习率
            {'params': self.model.attention.parameters(), 
             'lr': config.LEARNING_RATE * 0.5}
        ]
        
        self.optimizer = optim.AdamW(
            param_groups,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # 🎯 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=15,  # 每20个epoch重启一次
            T_mult=2,  # 重启周期翻倍
            eta_min=1e-6
        )
        
        # 🔥 杀手级损失函数
        self.criterion = KillerLoss(
            class_weights=self.current_weights,
            focal_alpha=config.FOCAL_ALPHA,
            focal_gamma=config.FOCAL_GAMMA
        )
        self.criterion.to(self.device)
        
        # 智能指标计算器
        self.metrics = SmartMetrics(config.NUM_CLASSES, config.CLASS_NAMES)
        
        logger.info("训练配置完毕！准备开战！🔥")
    
    def dynamic_weight_adjustment(self, val_metrics):
        """动态调整类别权重 - 老哥的智能调参"""
        if not config.DYNAMIC_WEIGHT_UPDATE:
            return
        
        class_ious = val_metrics['class_ious']
        
        # 🔥 根据IoU表现动态调整权重
        new_weights = self.current_weights.copy()
        
        # 病灶类别权重调整策略
        lesion_classes = [3, 4, 5]  # 声带小结、白斑、乳头状瘤
        
        for i, class_name in enumerate(config.CLASS_NAMES):
            current_iou = class_ious.get(class_name, 0)
            
            if i in lesion_classes:
                # 病灶类别：IoU越低，权重越高
                if current_iou < 0.1:
                    new_weights[i] = min(new_weights[i] * 1.2, 50.0)  # 权重增加20%，上限50
                elif current_iou > 0.3:
                    new_weights[i] = max(new_weights[i] * 0.9, 5.0)   # 权重减少10%，下限5
        
        # 更新权重
        if new_weights != self.current_weights:
            self.current_weights = new_weights
            # 重新创建损失函数
            self.criterion = KillerLoss(
                class_weights=self.current_weights,
                focal_alpha=config.FOCAL_ALPHA,
                focal_gamma=config.FOCAL_GAMMA
            )
            self.criterion.to(self.device)
            
            logger.info("动态调整类别权重！")
            for i, (name, weight) in enumerate(zip(config.CLASS_NAMES, self.current_weights)):
                logger.info(f"  {name}: {weight:.2f}")

# 继续下一部分...
def main():
    """老哥的主函数"""
    logger.info("🔥🔥🔥 老哥的SAM优化训练开始！🔥🔥🔥")
    
    # 显示GPU信息
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"老哥的装备：{gpu_name} ({gpu_memory:.1f}GB)")
    else:
        logger.warning("没有GPU？这可不行！")
    
    # 检查路径
    paths_to_check = [
        config.TRAIN_IMAGES_DIR, config.TRAIN_MASKS_DIR,
        config.VAL_IMAGES_DIR, config.VAL_MASKS_DIR,
        config.SAM_MODEL_PATH
    ]
    
    for path in paths_to_check:
        if not os.path.exists(path):
            logger.error(f"路径不存在 {path}")
            sys.exit(1)
    
    # 打印优化策略
    logger.info("老哥的优化策略（升级版）：")
    logger.info(f"  🔥 Focal Loss: {'启用' if config.USE_FOCAL_LOSS else '关闭'}")
    logger.info(f"  🎯 Dice Loss: {'启用' if config.USE_DICE_LOSS else '关闭'}")
    logger.info(f"  🧠 智能采样: {'启用' if config.USE_SMART_SAMPLING else '关闭'}")
    logger.info(f"  📏 多尺度训练: {'启用' if config.USE_MULTI_SCALE_LOSS else '关闭'}")
    logger.info(f"  🛡️ 标签平滑: {'启用' if config.USE_LABEL_SMOOTHING else '关闭'}")
    logger.info(f"  🔥 边缘感知损失: {'启用' if config.USE_EDGE_LOSS else '关闭'} (Sobel算子)")
    logger.info(f"  🎯 边界优化损失: {'启用' if config.USE_BOUNDARY_LOSS else '关闭'} (病灶边界)")
    
    # 打印类别映射
    logger.info("老哥的类别映射：")
    for original_id, new_id in config.ID_MAPPING.items():
        class_name = config.CLASS_NAMES[new_id]
        weight = config.INITIAL_CLASS_WEIGHTS[new_id]
        logger.info(f"  {original_id} -> {new_id} ({class_name}) 权重:{weight}")
    
    # 开始训练
    trainer = SuperTrainer()
    trainer.train()

# 在SuperTrainer类中添加训练方法
def add_training_methods_to_trainer():
    """添加训练方法到SuperTrainer类"""
    
    def train_one_epoch(self, epoch):
        """训练一个epoch - 老哥精心调教"""
        self.model.train()
        self.metrics.reset()
        
        running_loss = 0.0
        running_loss_dict = defaultdict(float)
        num_batches = 0
        
        # 更新数据集的epoch（用于多尺度训练）
        self.train_dataset.current_epoch = epoch
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [训练]")
        
        for batch_idx, (images, masks, filenames) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # 🔥 混合精度训练
            if config.MIXED_PRECISION and self.scaler:
                with autocast():
                    predictions, _ = self.model(images)
                    loss, loss_dict = self.criterion(predictions, masks)
                    loss = loss / config.GRADIENT_ACCUMULATION_STEPS
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                predictions, _ = self.model(images)
                loss, loss_dict = self.criterion(predictions, masks)
                loss = loss / config.GRADIENT_ACCUMULATION_STEPS
                
                loss.backward()
                
                if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # 更新指标
            self.metrics.update(predictions.detach(), masks)
            
            # 记录损失
            running_loss += loss.item()
            for key, value in loss_dict.items():
                running_loss_dict[key] += value
            num_batches += 1
            
            # 更新进度条
            if num_batches > 0:
                avg_loss = running_loss / num_batches * config.GRADIENT_ACCUMULATION_STEPS
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
            
            # 🚀 定期清理显存
            if batch_idx % config.CLEAR_CACHE_EVERY == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        # 计算epoch指标
        epoch_metrics = self.metrics.compute()
        avg_loss = running_loss / num_batches * config.GRADIENT_ACCUMULATION_STEPS
        
        # 详细的损失分解
        loss_breakdown = {}
        for key, value in running_loss_dict.items():
            loss_breakdown[key] = value / num_batches
        
        # 更新学习率
        self.scheduler.step()
        
        return avg_loss, epoch_metrics, loss_breakdown
    
    def validate_one_epoch(self, epoch):
        """验证一个epoch - 老哥严格把关"""
        self.model.eval()
        self.metrics.reset()
        
        running_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [验证]")
        
        with torch.no_grad():
            for batch_idx, (images, masks, filenames) in enumerate(pbar):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                if config.MIXED_PRECISION:
                    with autocast():
                        predictions, _ = self.model(images)
                        loss, _ = self.criterion(predictions, masks)
                else:
                    predictions, _ = self.model(images)
                    loss, _ = self.criterion(predictions, masks)
                
                # 更新指标
                self.metrics.update(predictions, masks)
                running_loss += loss.item()
                num_batches += 1
                
                # 更新进度条
                if num_batches > 0:
                    avg_loss = running_loss / num_batches
                    pbar.set_postfix({'val_loss': f'{avg_loss:.4f}'})
        
        # 计算epoch指标
        epoch_metrics = self.metrics.compute()
        avg_loss = running_loss / num_batches if num_batches > 0 else 0
        
        return avg_loss, epoch_metrics
    
    def print_epoch_summary(self, epoch, train_loss, train_metrics, val_loss, val_metrics, loss_breakdown):
        """打印epoch总结 - 老哥式汇报（优化版）"""
        logger.info(f"\n{'='*70}")
        logger.info(f"🔥 第 {epoch+1} 个Epoch总结 - 老哥汇报 (优化指标版):")
        logger.info(f"{'='*70}")
        
        # 🎯 优化版基础指标
        logger.info(f"📊 整体表现 (仅计算存在的类别):")
        logger.info(f"  训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}")
        logger.info(f"  训练mIoU: {train_metrics['mIoU']:.4f} ({train_metrics['mIoU']*100:.1f}%) | 验证mIoU: {val_metrics['mIoU']:.4f} ({val_metrics['mIoU']*100:.1f}%)")
        logger.info(f"  训练mF1: {train_metrics['mF1']:.4f} ({train_metrics['mF1']*100:.1f}%) | 验证mF1: {val_metrics['mF1']:.4f} ({val_metrics['mF1']*100:.1f}%)")
        logger.info(f"  训练准确率: {train_metrics['accuracy']:.4f} | 验证准确率: {val_metrics['accuracy']:.4f}")
        
        # 显示存在的类别
        if 'present_classes' in val_metrics:
            logger.info(f"  存在类别: {', '.join(val_metrics['present_classes'])}")
        
        # 🔥 指标对比说明
        logger.info(f"📈 指标对比:")
        logger.info(f"  传统mIoU: {val_metrics.get('mIoU_traditional', 0):.4f} | 优化mIoU: {val_metrics['mIoU']:.4f} | 提升: {val_metrics['mIoU'] - val_metrics.get('mIoU_traditional', 0):+.4f}")
        logger.info(f"  传统mF1: {val_metrics.get('mF1_traditional', 0):.4f} | 优化mF1: {val_metrics['mF1']:.4f} | 提升: {val_metrics['mF1'] - val_metrics.get('mF1_traditional', 0):+.4f}")
        
        # 🔍 损失分解（包含新的边缘损失）
        logger.info(f"🔍 损失分解:")
        loss_order = ['ce_loss', 'focal_loss', 'dice_loss', 'edge_loss', 'boundary_loss']
        for loss_name in loss_order:
            if loss_name in loss_breakdown:
                loss_value = loss_breakdown[loss_name]
                if loss_name == 'edge_loss':
                    logger.info(f"  {loss_name}: {loss_value:.4f} 🔥 (边缘感知)")
                elif loss_name == 'boundary_loss':
                    logger.info(f"  {loss_name}: {loss_value:.4f} 🎯 (边界优化)")
                else:
                    logger.info(f"  {loss_name}: {loss_value:.4f}")
        
        # 🔥 各类别详细IoU - 老哥最关心的
        logger.info(f"🎯 各类别详细表现:")
        for class_name, iou in val_metrics['class_ious'].items():
            dice = val_metrics['class_dices'].get(class_name, 0)
            f1 = val_metrics['class_f1s'].get(class_name, 0)
            
            status = ""
            if class_name in ['声带小结', '声带白斑', '声带乳头状瘤']:
                if iou < 0.1:
                    status = "😰 需要关注!"
                elif iou < 0.3:
                    status = "😐 有待提高"
                elif iou < 0.5:
                    status = "😊 不错"
                else:
                    status = "🔥 优秀!"
            else:
                if iou < 0.5:
                    status = "😐 一般"
                elif iou < 0.7:
                    status = "😊 不错"
                else:
                    status = "🔥 优秀!"
            
            logger.info(f"  {class_name}: IoU={iou:.4f} | Dice={dice:.4f} | F1={f1:.4f} {status}")
        
        # 🏥 病灶专项详细报告
        if val_metrics.get('lesion_present_names'):
            logger.info(f"🏥 病灶专项表现 (仅检测到的病灶):")
            logger.info(f"  检测到病灶: {', '.join(val_metrics['lesion_present_names'])}")
            logger.info(f"  病灶mIoU: {val_metrics['lesion_mIoU']:.4f} ({val_metrics['lesion_mIoU']*100:.1f}%)")
            logger.info(f"  病灶mDice: {val_metrics['lesion_mDice']:.4f} ({val_metrics['lesion_mDice']*100:.1f}%)")
            logger.info(f"  病灶mF1: {val_metrics['lesion_mF1']:.4f} ({val_metrics['lesion_mF1']*100:.1f}%)")
            
            # 病灶对比
            logger.info(f"  病灶指标对比:")
            logger.info(f"    传统病灶mIoU: {val_metrics.get('lesion_mIoU_traditional', 0):.4f} | 优化: {val_metrics['lesion_mIoU']:.4f} | 提升: {val_metrics['lesion_mIoU'] - val_metrics.get('lesion_mIoU_traditional', 0):+.4f}")
        else:
            logger.info(f"🏥 病灶专项表现: 未检测到任何病灶")
        
        # 🚀 病灶学习进度报告
        if val_metrics.get('lesion_report'):
            logger.info(f"📈 病灶学习进度:")
            lesion_names = {'sdxj': '声带小结', 'sdbb': '声带白斑', 'rtzl': '声带乳头状瘤'}
            for lesion_code, report in val_metrics['lesion_report'].items():
                lesion_name = lesion_names.get(lesion_code, lesion_code)
                trend_emoji = "📈" if report['trend'] == 'improving' else "📊"
                logger.info(f"  {lesion_name}: {report['current_iou']:.4f} {trend_emoji}")
        
        # 学习率信息
        current_lr = self.optimizer.param_groups[0]['lr']
        logger.info(f"📚 当前学习率: {current_lr:.2e}")
        
        logger.info(f"{'='*70}\n")
    
    def save_checkpoint(self, epoch, val_metrics, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_miou': self.best_miou,
            'metrics': val_metrics,
            'config': config.__dict__,
            'class_weights': self.current_weights
        }
        
        # 保存常规检查点
        if (epoch + 1) % config.SAVE_EVERY == 0:
            checkpoint_path = os.path.join(config.RESULTS_DIR, "models", f"checkpoint_epoch_{epoch+1:03d}.pth")
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"检查点已保存 {checkpoint_path}")
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(config.RESULTS_DIR, "models", "best_model.pth")
            torch.save(checkpoint, best_path)
            logger.info(f"🏆 最佳模型已保存！mIoU: {val_metrics['mIoU']:.4f}")
    
    def save_training_plots(self):
        """保存训练曲线图"""
        if not self.history['train_loss']:
            return
        
        plt.figure(figsize=(15, 10))
        
        # 损失曲线
        plt.subplot(2, 3, 1)
        plt.plot(self.history['train_loss'], label='训练损失', color='red')
        plt.plot(self.history['val_loss'], label='验证损失', color='blue')
        plt.title('损失曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # mIoU曲线
        plt.subplot(2, 3, 2)
        plt.plot(self.history['train_miou'], label='训练mIoU', color='red')
        plt.plot(self.history['val_miou'], label='验证mIoU', color='blue')
        plt.title('mIoU曲线')
        plt.xlabel('Epoch')
        plt.ylabel('mIoU')
        plt.legend()
        plt.grid(True)
        
        # 🔥 病灶类别IoU趋势 - 老哥特制
        colors = ['green', 'orange', 'purple']
        lesion_names = {'sdxj': '声带小结', 'sdbb': '声带白斑', 'rtzl': '声带乳头状瘤'}
        
        for i, (lesion_code, color) in enumerate(zip(self.history['lesion_ious'].keys(), colors)):
            plt.subplot(2, 3, 3+i)
            if self.history['lesion_ious'][lesion_code]:
                plt.plot(self.history['lesion_ious'][lesion_code], color=color, linewidth=2)
                plt.title(f'{lesion_names[lesion_code]} IoU趋势')
                plt.xlabel('Validation Step')
                plt.ylabel('IoU')
                plt.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(config.RESULTS_DIR, "training_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"训练曲线已保存 {plot_path}")
    
    def train(self):
        """主训练循环 - 老哥亲自督战！"""
        logger.info("🚀🚀🚀 老哥开始督战训练！🚀🚀🚀")
        
        start_time = time.time()
        
        for epoch in range(config.NUM_EPOCHS):
            logger.info(f"\n🔥 第 {epoch+1}/{config.NUM_EPOCHS} 个Epoch开始！")
            
            # 训练阶段
            train_loss, train_metrics, loss_breakdown = self.train_one_epoch(epoch)
            
            # 验证阶段
            if (epoch + 1) % config.EVAL_EVERY == 0:
                val_loss, val_metrics = self.validate_one_epoch(epoch)
                
                # 打印详细报告
                self.print_epoch_summary(epoch, train_loss, train_metrics, val_loss, val_metrics, loss_breakdown)
                
                # 🔥 动态调整权重
                self.dynamic_weight_adjustment(val_metrics)
                
                # 记录历史
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['train_miou'].append(train_metrics['mIoU'])
                self.history['val_miou'].append(val_metrics['mIoU'])
                
                # 记录病灶进度
                for lesion_code, report in val_metrics.get('lesion_report', {}).items():
                    if lesion_code in self.history['lesion_ious']:
                        self.history['lesion_ious'][lesion_code].append(report['current_iou'])
                
                # 🏆 检查是否最佳模型
                current_miou = val_metrics['mIoU']
                is_best = current_miou > self.best_miou
                
                if is_best:
                    self.best_miou = current_miou
                    self.patience_counter = 0
                    logger.info(f"🎉 新纪录！mIoU: {self.best_miou:.4f}")
                else:
                    self.patience_counter += 1
                    logger.info(f"耐心等待中... ({self.patience_counter}/{config.EARLY_STOPPING_PATIENCE})")
                
                # 保存检查点
                self.save_checkpoint(epoch, val_metrics, is_best)
                
                # 早停检查
                if self.patience_counter >= config.EARLY_STOPPING_PATIENCE:
                    logger.info("没有改善，提前结束训练！")
                    break
            
            # 保存训练图表
            if (epoch + 1) % (config.SAVE_EVERY * 2) == 0:
                self.save_training_plots()
            
            # 🚀 显存清理
            torch.cuda.empty_cache()
            gc.collect()
        
        # 训练结束
        total_time = time.time() - start_time
        logger.info(f"\n🎊🎊🎊 训练完成！🎊🎊🎊")
        logger.info(f"总耗时: {total_time/3600:.2f} 小时")
        logger.info(f"最佳mIoU: {self.best_miou:.4f}")
        
        # 保存最终结果
        self.save_training_plots()
        
        # 保存训练历史
        history_path = os.path.join(config.RESULTS_DIR, "training_history.json")
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        
        logger.info(f"一切搞定！结果保存在 {config.RESULTS_DIR}")
    
    # 动态添加方法到SuperTrainer类
    SuperTrainer.train_one_epoch = train_one_epoch
    SuperTrainer.validate_one_epoch = validate_one_epoch
    SuperTrainer.print_epoch_summary = print_epoch_summary
    SuperTrainer.save_checkpoint = save_checkpoint
    SuperTrainer.save_training_plots = save_training_plots
    SuperTrainer.train = train

# 在主函数前调用
add_training_methods_to_trainer()

if __name__ == "__main__":
    main() 