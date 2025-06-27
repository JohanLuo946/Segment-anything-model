#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM声带病灶分割训练脚本 - AutoDL专用版本
完全自包含，无需yaml配置文件
作者：罗老哥
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
import logging
from tqdm import tqdm
import gc

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===== 配置部分 (硬编码，无需yaml) =====
class Config:
    """所有训练配置"""
    
    # 路径配置 - 根据老哥提供的AutoDL路径
    TRAIN_IMAGES_DIR = "/root/autodl-tmp/SAM/data/train/images"
    TRAIN_MASKS_DIR = "/root/autodl-tmp/SAM/data/train/masks"
    VAL_IMAGES_DIR = "/root/autodl-tmp/SAM/data/val/images"
    VAL_MASKS_DIR = "/root/autodl-tmp/SAM/data/val/masks"
    SAM_MODEL_PATH = "autodl-tmp/SAM/models/sam_vit_b_01ec64.pth"
    RESULTS_DIR = "/root/autodl-tmp/SAM/results"
    
    # 数据配置
    NUM_CLASSES = 6
    IMAGE_SIZE = 1024  # SAM标准输入尺寸
    BATCH_SIZE = 1     # 显存不足时降低到1
    NUM_WORKERS = 2    # 也减少worker数量
    
    # 类别映射 - 用户指定的正确映射
    ID_MAPPING = {
        0: 0,    # 背景
        170: 1,  # 左声带
        184: 2,  # 右声带
        105: 3,  # 声带小结
        23: 4,   # 声带白斑
        146: 5,  # 声带乳头状瘤
    }
    
    # 类别名称
    CLASS_NAMES = [
        "background",
        "left", 
        "right",
        "sdxj",
        "sdbb", 
        "rtzl"
    ]
    
    # 类别权重 (处理类别不平衡)
    CLASS_WEIGHTS = [0.5, 1.0, 1.0, 2.0, 3.0, 4.0]
    
    # 训练配置
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    GRADIENT_ACCUMULATION_STEPS = 8  # 增加梯度累积，等效batch_size=8
    
    # SAM配置
    SAM_MODEL_TYPE = "vit_b"
    PIXEL_MEAN = [123.675, 116.28, 103.53]
    PIXEL_STD = [58.395, 57.12, 57.375]
    
    # 设备配置
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MIXED_PRECISION = True
    
    # 显存优化设置
    CLEAR_CACHE_EVERY = 5      # 每5个batch清理一次显存
    GRADIENT_CHECKPOINTING = True  # 启用梯度检查点
    PIN_MEMORY = False         # 关闭pin_memory节省显存
    
    # 保存配置
    SAVE_EVERY = 10
    EVAL_EVERY = 1
    EARLY_STOPPING_PATIENCE = 10

config = Config()

# ===== 数据集类 =====
class VocalFoldDataset(Dataset):
    """声带病灶数据集"""
    
    def __init__(self, images_dir, masks_dir, transform=None, is_train=True):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.is_train = is_train
        
        # 获取所有图像文件
        self.image_files = []
        for file in os.listdir(images_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                mask_file = file.replace('.jpg', '.png').replace('.jpeg', '.png')
                mask_path = os.path.join(masks_dir, mask_file)
                if os.path.exists(mask_path):
                    self.image_files.append(file)
        
        logger.info(f"Found {len(self.image_files)} image-mask pairs in {images_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 加载图像
        image_file = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 加载掩码
        mask_file = image_file.replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(self.masks_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 应用ID映射转换
        mask = self.apply_id_mapping(mask)
        
        # 调整大小到SAM输入尺寸
        image = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
        mask = cv2.resize(mask, (config.IMAGE_SIZE, config.IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        
        # 转换为tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()
        
        # 标准化
        mean = torch.tensor(config.PIXEL_MEAN).view(3, 1, 1) / 255.0
        std = torch.tensor(config.PIXEL_STD).view(3, 1, 1) / 255.0
        image = (image - mean) / std
        
        return image, mask, image_file
    
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

# ===== SAM模型适配 =====
class SAMSegmentationModel(nn.Module):
    """SAM分割模型适配器 - 修正版本"""
    
    def __init__(self, sam_model, num_classes):
        super().__init__()
        self.sam = sam_model
        self.num_classes = num_classes
        
        # 在image_embeddings上添加多类分割头
        # SAM的image_encoder输出是256维特征
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        
        # 冻结部分组件
        self.freeze_components()
    
    def freeze_components(self):
        """冻结SAM的部分组件"""
        # 冻结image_encoder的大部分层，只微调后面几层
        layers_to_freeze = list(self.sam.image_encoder.children())[:-2]  # 冻结前面的层
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
    
    def forward(self, images, input_points=None, input_labels=None, input_boxes=None):
        batch_size = images.shape[0]
        
        # 图像编码
        image_embeddings = self.sam.image_encoder(images)
        
        # 直接在image_embeddings上做多类分割
        # image_embeddings shape: [B, 256, 64, 64] (for 1024x1024 input)
        segmentation_logits = self.segmentation_head(image_embeddings)
        
        # 上采样到原始尺寸
        segmentation_logits = torch.nn.functional.interpolate(
            segmentation_logits,
            size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        
        # 为了兼容性，也计算一个简单的IoU预测
        iou_predictions = torch.ones(batch_size, 1).to(images.device) * 0.5
        
        return segmentation_logits, iou_predictions

# ===== 损失函数 =====
class CombinedLoss(nn.Module):
    """组合损失函数"""
    
    def __init__(self, class_weights=None):
        super().__init__()
        self.class_weights = class_weights
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights).float()
        
        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)
        
    def forward(self, predictions, targets):
        # 交叉熵损失
        ce_loss = self.ce_loss(predictions, targets)
        
        # Dice损失
        dice_loss = self.dice_loss(predictions, targets)
        
        # 组合损失
        total_loss = 0.6 * ce_loss + 0.4 * dice_loss
        
        return total_loss, {"ce_loss": ce_loss.item(), "dice_loss": dice_loss.item()}
    
    def dice_loss(self, predictions, targets):
        """Dice损失"""
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

# ===== 评估指标 =====
class MetricsCalculator:
    """评估指标计算器"""
    
    def __init__(self, num_classes, class_names):
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()
    
    def reset(self):
        self.total_samples = 0
        self.class_iou = np.zeros(self.num_classes)
        self.class_dice = np.zeros(self.num_classes)
        self.class_precision = np.zeros(self.num_classes)
        self.class_recall = np.zeros(self.num_classes)
        self.class_f1 = np.zeros(self.num_classes)
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, predictions, targets):
        """更新指标"""
        predictions = torch.argmax(predictions, dim=1)
        
        for pred, target in zip(predictions, targets):
            pred_np = pred.cpu().numpy()
            target_np = target.cpu().numpy()
            
            # 更新混淆矩阵
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    self.confusion_matrix[i, j] += np.sum((target_np == i) & (pred_np == j))
            
            # 计算每个类别的指标
            for class_idx in range(self.num_classes):
                pred_mask = (pred_np == class_idx)
                target_mask = (target_np == class_idx)
                
                # IoU
                intersection = np.sum(pred_mask & target_mask)
                union = np.sum(pred_mask | target_mask)
                if union > 0:
                    self.class_iou[class_idx] += intersection / union
                
                # Dice
                if np.sum(pred_mask) + np.sum(target_mask) > 0:
                    dice = 2 * intersection / (np.sum(pred_mask) + np.sum(target_mask))
                    self.class_dice[class_idx] += dice
        
        self.total_samples += len(predictions)
    
    def compute(self):
        """计算最终指标"""
        # 平均IoU和Dice
        mean_iou = np.mean(self.class_iou) / self.total_samples if self.total_samples > 0 else 0
        mean_dice = np.mean(self.class_dice) / self.total_samples if self.total_samples > 0 else 0
        
        # 每个类别的IoU
        class_ious = self.class_iou / self.total_samples if self.total_samples > 0 else self.class_iou
        
        # 精确率、召回率、F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.confusion_matrix.flatten(), 
            self.confusion_matrix.flatten(), 
            average='weighted',
            zero_division=0
        )
        
        return {
            'mIoU': mean_iou,
            'mDice': mean_dice,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'class_ious': dict(zip(self.class_names, class_ious))
        }

# ===== 训练器 =====
class SAMTrainer:
    """SAM训练器"""
    
    def __init__(self):
        self.device = torch.device(config.DEVICE)
        self.scaler = GradScaler() if config.MIXED_PRECISION else None
        
        # 创建结果目录
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        os.makedirs(os.path.join(config.RESULTS_DIR, "models"), exist_ok=True)
        os.makedirs(os.path.join(config.RESULTS_DIR, "logs"), exist_ok=True)
        
        # 加载模型
        self.load_sam_model()
        
        # 创建数据加载器
        self.create_dataloaders()
        
        # 设置优化器和损失函数
        self.setup_training()
        
        # 训练历史
        self.train_history = {'loss': [], 'miou': []}
        self.val_history = {'loss': [], 'miou': []}
        self.best_miou = 0.0
        self.patience_counter = 0
    
    def load_sam_model(self):
        """加载SAM模型"""
        logger.info("Loading SAM model...")
        
        try:
            # 尝试加载segment_anything
            from segment_anything import sam_model_registry, SamPredictor
            
            # 加载预训练SAM模型
            sam = sam_model_registry[config.SAM_MODEL_TYPE](checkpoint=config.SAM_MODEL_PATH)
            sam.to(self.device)
            
            # 创建分割模型
            self.model = SAMSegmentationModel(sam, config.NUM_CLASSES)
            self.model.to(self.device)
            
            logger.info("SAM model loaded successfully!")
            
        except ImportError:
            logger.error("segment_anything not installed. Installing...")
            os.system("pip install git+https://github.com/facebookresearch/segment-anything.git")
            # 重新导入
            from segment_anything import sam_model_registry, SamPredictor
            sam = sam_model_registry[config.SAM_MODEL_TYPE](checkpoint=config.SAM_MODEL_PATH)
            sam.to(self.device)
            self.model = SAMSegmentationModel(sam, config.NUM_CLASSES)
            self.model.to(self.device)
    
    def create_dataloaders(self):
        """创建数据加载器"""
        logger.info("Creating data loaders...")
        
        # 训练数据集
        train_dataset = VocalFoldDataset(
            config.TRAIN_IMAGES_DIR, 
            config.TRAIN_MASKS_DIR, 
            is_train=True
        )
        
        # 验证数据集
        val_dataset = VocalFoldDataset(
            config.VAL_IMAGES_DIR, 
            config.VAL_MASKS_DIR, 
            is_train=False
        )
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY  # 使用配置中的设置
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY  # 使用配置中的设置
        )
        
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    def setup_training(self):
        """设置训练组件"""
        logger.info("Setting up training...")
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config.NUM_EPOCHS
        )
        
        # 损失函数
        self.criterion = CombinedLoss(config.CLASS_WEIGHTS)
        self.criterion.to(self.device)
        
        # 指标计算器
        self.metrics_calculator = MetricsCalculator(config.NUM_CLASSES, config.CLASS_NAMES)
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        self.metrics_calculator.reset()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        
        for batch_idx, (images, masks, filenames) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # 前向传播
            if config.MIXED_PRECISION:
                with autocast():
                    predictions, iou_preds = self.model(images)
                    loss, loss_dict = self.criterion(predictions, masks)
                    loss = loss / config.GRADIENT_ACCUMULATION_STEPS
            else:
                predictions, iou_preds = self.model(images)
                loss, loss_dict = self.criterion(predictions, masks)
                loss = loss / config.GRADIENT_ACCUMULATION_STEPS
            
            # 反向传播
            if config.MIXED_PRECISION:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                if config.MIXED_PRECISION:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
            
            # 更新指标
            total_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
            self.metrics_calculator.update(predictions, masks)
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'CE': f"{loss_dict['ce_loss']:.4f}",
                'Dice': f"{loss_dict['dice_loss']:.4f}"
            })
            
            # 清理GPU缓存（更频繁）
            if batch_idx % config.CLEAR_CACHE_EVERY == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        # 计算epoch指标
        metrics = self.metrics_calculator.compute()
        avg_loss = total_loss / len(self.train_loader)
        
        self.train_history['loss'].append(avg_loss)
        self.train_history['miou'].append(metrics['mIoU'])
        
        return avg_loss, metrics
    
    def validate(self, epoch):
        """验证"""
        self.model.eval()
        total_loss = 0.0
        self.metrics_calculator.reset()
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            
            for images, masks, filenames in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                if config.MIXED_PRECISION:
                    with autocast():
                        predictions, iou_preds = self.model(images)
                        loss, loss_dict = self.criterion(predictions, masks)
                else:
                    predictions, iou_preds = self.model(images)
                    loss, loss_dict = self.criterion(predictions, masks)
                
                total_loss += loss.item()
                self.metrics_calculator.update(predictions, masks)
                
                pbar.set_postfix({'Val Loss': f"{loss.item():.4f}"})
        
        # 计算验证指标
        metrics = self.metrics_calculator.compute()
        avg_loss = total_loss / len(self.val_loader)
        
        self.val_history['loss'].append(avg_loss)
        self.val_history['miou'].append(metrics['mIoU'])
        
        return avg_loss, metrics
    
    def save_model(self, epoch, metrics, is_best=False):
        """保存模型"""
        save_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': vars(config)
        }
        
        # 保存最新模型
        latest_path = os.path.join(config.RESULTS_DIR, "models", "latest.pth")
        torch.save(save_dict, latest_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(config.RESULTS_DIR, "models", "best_model.pth")
            torch.save(save_dict, best_path)
            logger.info(f"New best model saved with mIoU: {metrics['mIoU']:.4f}")
        
        # 定期保存检查点
        if (epoch + 1) % config.SAVE_EVERY == 0:
            checkpoint_path = os.path.join(config.RESULTS_DIR, "models", f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(save_dict, checkpoint_path)
    
    def train(self):
        """主训练循环"""
        logger.info("Starting training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model: SAM {config.SAM_MODEL_TYPE}")
        logger.info(f"Classes: {config.NUM_CLASSES}")
        logger.info(f"Batch size: {config.BATCH_SIZE}")
        logger.info(f"Learning rate: {config.LEARNING_RATE}")
        
        for epoch in range(config.NUM_EPOCHS):
            # 训练
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # 验证
            if (epoch + 1) % config.EVAL_EVERY == 0:
                val_loss, val_metrics = self.validate(epoch)
                
                # 打印详细指标
                logger.info(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
                logger.info(f"Train Loss: {train_loss:.4f}, Train mIoU: {train_metrics['mIoU']:.4f}")
                logger.info(f"Val Loss: {val_loss:.4f}, Val mIoU: {val_metrics['mIoU']:.4f}")
                logger.info(f"Val mDice: {val_metrics['mDice']:.4f}, Val F1: {val_metrics['f1']:.4f}")
                
                # 打印每个类别的IoU
                logger.info("各类别IoU:")
                for class_name, iou in val_metrics['class_ious'].items():
                    logger.info(f"  {class_name}: {iou:.4f}")
                
                # 检查是否为最佳模型
                is_best = val_metrics['mIoU'] > self.best_miou
                if is_best:
                    self.best_miou = val_metrics['mIoU']
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # 保存模型
                self.save_model(epoch, val_metrics, is_best)
                
                # 早停检查
                if self.patience_counter >= config.EARLY_STOPPING_PATIENCE:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            # 更新学习率
            self.scheduler.step()
            
            # 打印当前学习率
            current_lr = self.scheduler.get_last_lr()[0]
            logger.info(f"Epoch {epoch+1}, LR: {current_lr:.2e}")
        
        logger.info("Training completed!")
        logger.info(f"Best mIoU: {self.best_miou:.4f}")
        
        # 保存训练历史
        history_path = os.path.join(config.RESULTS_DIR, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump({
                'train_history': self.train_history,
                'val_history': self.val_history,
                'best_miou': self.best_miou
            }, f, indent=2)

# ===== 主函数 =====
def main():
    """主函数"""
    logger.info("=== SAM声带病灶分割训练 - AutoDL版本 ===")
    
    # 检查GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        logger.warning("CUDA not available, using CPU")
    
    # 检查路径
    paths_to_check = [
        config.TRAIN_IMAGES_DIR,
        config.TRAIN_MASKS_DIR,
        config.VAL_IMAGES_DIR,
        config.VAL_MASKS_DIR,
        config.SAM_MODEL_PATH
    ]
    
    for path in paths_to_check:
        if not os.path.exists(path):
            logger.error(f"Path not found: {path}")
            sys.exit(1)
    
    logger.info("All paths verified!")
    
    # 打印映射配置
    logger.info("类别映射配置:")
    for original_id, new_id in config.ID_MAPPING.items():
        class_name = config.CLASS_NAMES[new_id]
        logger.info(f"  {original_id} -> {new_id} ({class_name})")
    
    # 创建训练器并开始训练
    trainer = SAMTrainer()
    trainer.train()

if __name__ == "__main__":
    main() 