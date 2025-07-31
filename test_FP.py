#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔥 SAM 2类病灶分割 - 纯医学分割专用指标测试脚本

专注于真正有意义的医学分割指标：
🎯 分割质量指标: Dice系数、IoU、F1分数
🔥 边界精确性指标: Hausdorff距离、平均表面距离(ASD)、HD95

⚡ 已删除无意义的传统分类指标：
❌ 精确度(Precision) - 忽略空间连续性，对分割无意义
❌ 召回率(Recall) - 基于像素分类，无法评估边界质量
❌ 特异性(Specificity) - 对医学分割任务无实际价值

✅ 为什么边界距离指标更重要：
• Hausdorff距离: 直接测量边界最大偏移，识别过度分割
• 平均表面距离: 评估整体边界精确性，对所有FP敏感
• HD95: 鲁棒的边界质量评估，忽略异常点

🎯 评估标准: 边界距离 <5px(优秀) | 5-10px(良好) | >10px(需改进)
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import logging
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
from scipy.spatial.distance import directed_hausdorff
from scipy import ndimage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class TestConfig:
    # 测试数据配置
    TEST_IMAGES_DIR = "autodl-tmp/SAM/12classes_lesion/test/images"
    TEST_MASKS_DIR = "autodl-tmp/SAM/12classes_lesion/test/masks"
    
    # 模型配置
    MODEL_PATH = "autodl-tmp/SAM/results/models/dsc_enhanced_sdbb_4/models/best_model_lesion_dice.pth"
    SAM_MODEL_PATH = "autodl-tmp/SAM/pre_models/sam_vit_b_01ec64.pth"  # SAM预训练模型路径
    LESION_ID = 29
    LESION_NAME = "声带白斑"
    LESION_CODE = "sdbb"
    
    # 测试配置
    IMAGE_SIZE = 1024
    BATCH_SIZE = 1
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 分析配置
    CONFIDENCE_THRESHOLDS = [0.5, 0.6, 0.7, 0.8]  # 简化阈值
    SAVE_VISUALIZATIONS = True
    MAX_VISUALIZATIONS = 10  # 减少可视化数量
    
    # 输出目录
    RESULTS_DIR = f"test_results_{LESION_CODE}"
    
    # SAM配置
    PIXEL_MEAN = [123.675, 116.28, 103.53]
    PIXEL_STD = [58.395, 57.12, 57.375]
    
    ID_MAPPING = {0: 0, LESION_ID: 1}
    CLASS_NAMES = ["背景", LESION_NAME]

config = TestConfig()

# 导入训练脚本中的模型类
from train_2classes_v3 import DSCEnhancedSAMModel, DSCEnhancedDataset

class LesionOnlyDataset(DSCEnhancedDataset):
    """只包含目标病灶的测试数据集"""
    
    def __init__(self, images_dir, masks_dir):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        
        # 获取所有图像文件
        all_files = []
        for file in os.listdir(images_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                mask_file = file.replace('.jpg', '.png').replace('.jpeg', '.png')
                mask_path = os.path.join(masks_dir, mask_file)
                if os.path.exists(mask_path):
                    all_files.append(file)
        
        # 过滤：只保留包含目标病灶的图像
        self.image_files = []
        self.lesion_areas = []
        
        logger.info(f"开始过滤测试集，只保留含有{config.LESION_NAME}(ID={config.LESION_ID})的图像...")
        
        for file in tqdm(all_files, desc="过滤数据集"):
            mask_file = file.replace('.jpg', '.png').replace('.jpeg', '.png')
            mask_path = os.path.join(masks_dir, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if mask is not None and config.LESION_ID in np.unique(mask):
                self.image_files.append(file)
                lesion_area = np.sum(mask == config.LESION_ID)
                self.lesion_areas.append(lesion_area)
        
        logger.info(f"过滤完成：原始{len(all_files)}个 → 有效{len(self.image_files)}个")
        if self.lesion_areas:
            logger.info(f"病灶面积统计: 最小={min(self.lesion_areas)}, 最大={max(self.lesion_areas)}, 平均={np.mean(self.lesion_areas):.0f}")
    
    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask_file = image_file.replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(self.masks_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 保存原始尺寸和掩码
        original_height, original_width = image.shape[:2]
        original_mask = mask.copy()
        
        # 应用ID映射
        mask = self.apply_id_mapping(mask)
        
        # 调整尺寸
        image = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
        mask = cv2.resize(mask, (config.IMAGE_SIZE, config.IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        
        # 归一化
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()
        
        # SAM标准化
        mean = torch.tensor(config.PIXEL_MEAN).view(3, 1, 1) / 255.0
        std = torch.tensor(config.PIXEL_STD).view(3, 1, 1) / 255.0
        image = (image - mean) / std
        
        return image, mask, image_file, (original_height, original_width), original_mask

class MetricsCalculator:
    """医学分割专用指标计算器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.sample_metrics = []
        self.threshold_metrics = {th: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0} for th in config.CONFIDENCE_THRESHOLDS}
    
    def calculate_hausdorff_distance(self, pred_mask, target_mask):
        """计算Hausdorff距离 - 边界质量的关键指标"""
        try:
            # 获取边界点
            pred_boundary = self.get_boundary_points(pred_mask)
            target_boundary = self.get_boundary_points(target_mask)
            
            if len(pred_boundary) == 0 or len(target_boundary) == 0:
                return float('inf'), float('inf'), float('inf')
            
            # 计算双向Hausdorff距离
            hd1 = directed_hausdorff(pred_boundary, target_boundary)[0]
            hd2 = directed_hausdorff(target_boundary, pred_boundary)[0]
            hd = max(hd1, hd2)
            
            # 计算平均表面距离(ASD)
            asd = self.calculate_average_surface_distance(pred_boundary, target_boundary)
            
            # 计算HD95 (95分位数)
            hd95 = self.calculate_hd95(pred_boundary, target_boundary)
            
            return float(hd), float(asd), float(hd95)
            
        except Exception as e:
            logger.warning(f"计算HD距离时出错: {e}")
            return float('inf'), float('inf'), float('inf')
    
    def get_boundary_points(self, mask):
        """提取边界点坐标"""
        # 使用形态学操作获取边界
        boundary = mask - ndimage.binary_erosion(mask)
        y_coords, x_coords = np.where(boundary > 0)
        if len(y_coords) > 0:
            return np.column_stack([y_coords, x_coords])
        return np.array([])
    
    def calculate_average_surface_distance(self, pred_boundary, target_boundary):
        """计算平均表面距离"""
        if len(pred_boundary) == 0 or len(target_boundary) == 0:
            return float('inf')
        
        # 从预测边界到真实边界的平均距离
        dist1 = np.mean([np.min(np.linalg.norm(pred_boundary - target_pt, axis=1)) 
                        for target_pt in target_boundary])
        
        # 从真实边界到预测边界的平均距离  
        dist2 = np.mean([np.min(np.linalg.norm(target_boundary - pred_pt, axis=1)) 
                        for pred_pt in pred_boundary])
        
        return (dist1 + dist2) / 2
    
    def calculate_hd95(self, pred_boundary, target_boundary):
        """计算95分位数Hausdorff距离 - 更鲁棒的边界指标"""
        if len(pred_boundary) == 0 or len(target_boundary) == 0:
            return float('inf')
        
        # 计算所有点对距离的95分位数
        distances1 = [np.min(np.linalg.norm(pred_boundary - target_pt, axis=1)) 
                     for target_pt in target_boundary]
        distances2 = [np.min(np.linalg.norm(target_boundary - pred_pt, axis=1)) 
                     for pred_pt in pred_boundary]
        
        all_distances = distances1 + distances2
        if len(all_distances) > 0:
            return np.percentile(all_distances, 95)
        return float('inf')
    
    def calculate_sample_metrics(self, pred_prob, target_mask, filename):
        """计算分割专用指标 - 专注于医学意义"""
        sample_result = {'filename': str(filename), 'lesion_area': int(np.sum(target_mask == 1))}
        
        for threshold in config.CONFIDENCE_THRESHOLDS:
            pred_mask = (pred_prob > threshold).astype(np.uint8)
            target_binary = target_mask.astype(np.uint8)
            
            # 计算混淆矩阵 - 仅用于计算分割质量指标
            tp = np.sum((pred_mask == 1) & (target_binary == 1))
            fp = np.sum((pred_mask == 1) & (target_binary == 0))
            fn = np.sum((pred_mask == 0) & (target_binary == 1))
            tn = np.sum((pred_mask == 0) & (target_binary == 0))
            
            # 🎯 分割质量指标 - 最重要的指标
            epsilon = 1e-8
            dice = 2 * tp / (2 * tp + fp + fn + epsilon)
            iou = tp / (tp + fp + fn + epsilon)
            f1 = dice  # 对于分割任务，F1与Dice等价
            
            # FP率 - 仅作为错误分析参考
            fp_rate = fp / (fp + tn + epsilon)
            
            # 🔥 边界精确性指标 - 对过度分割最敏感
            hd, asd, hd95 = float('inf'), float('inf'), float('inf')
            if np.sum(target_binary) > 0 and np.sum(pred_mask) > 0:  # 只有当两者都有前景时才计算
                hd, asd, hd95 = self.calculate_hausdorff_distance(pred_mask, target_binary)
            elif np.sum(target_binary) > 0 and np.sum(pred_mask) == 0:  # 有真值但无预测
                hd, asd, hd95 = float('inf'), float('inf'), float('inf')
            elif np.sum(target_binary) == 0 and np.sum(pred_mask) > 0:  # 无真值但有预测(纯FP)
                hd, asd, hd95 = float('inf'), float('inf'), float('inf')
            else:  # 都没有
                hd, asd, hd95 = 0.0, 0.0, 0.0
            
            # 保存分割专用指标
            sample_result[f'dice_{threshold}'] = float(dice)
            sample_result[f'iou_{threshold}'] = float(iou)
            sample_result[f'f1_{threshold}'] = float(f1)
            sample_result[f'fp_count_{threshold}'] = int(fp)
            sample_result[f'fn_count_{threshold}'] = int(fn)
            sample_result[f'fp_rate_{threshold}'] = float(fp_rate)
            
            # 🔥 边界距离指标
            sample_result[f'hausdorff_{threshold}'] = float(hd) if hd != float('inf') else 999.0
            sample_result[f'asd_{threshold}'] = float(asd) if asd != float('inf') else 999.0
            sample_result[f'hd95_{threshold}'] = float(hd95) if hd95 != float('inf') else 999.0
            
            # 累积全局指标
            self.threshold_metrics[threshold]['tp'] += tp
            self.threshold_metrics[threshold]['fp'] += fp
            self.threshold_metrics[threshold]['fn'] += fn
            self.threshold_metrics[threshold]['tn'] += tn
            
            # 累积边界距离指标
            if 'hd_values' not in self.threshold_metrics[threshold]:
                self.threshold_metrics[threshold]['hd_values'] = []
                self.threshold_metrics[threshold]['asd_values'] = []
                self.threshold_metrics[threshold]['hd95_values'] = []
            
            if hd != float('inf'):
                self.threshold_metrics[threshold]['hd_values'].append(hd)
            if asd != float('inf'):
                self.threshold_metrics[threshold]['asd_values'].append(asd)
            if hd95 != float('inf'):
                self.threshold_metrics[threshold]['hd95_values'].append(hd95)
        
        self.sample_metrics.append(sample_result)
        return sample_result
    
    def get_summary_metrics(self):
        """获取分割专用汇总指标"""
        summary = {}
        
        for threshold in config.CONFIDENCE_THRESHOLDS:
            metrics = self.threshold_metrics[threshold]
            tp, fp, fn, tn = metrics['tp'], metrics['fp'], metrics['fn'], metrics['tn']
            
            epsilon = 1e-8
            # 🎯 分割质量指标
            dice = 2 * tp / (2 * tp + fp + fn + epsilon)
            iou = tp / (tp + fp + fn + epsilon)
            f1 = dice  # 对于分割任务，F1与Dice等价
            
            # FP率 - 仅作为错误分析参考
            fp_rate = fp / (fp + tn + epsilon)
            
            # 🔥 边界距离指标统计
            mean_hd = np.mean(metrics['hd_values']) if metrics['hd_values'] else 999.0
            mean_asd = np.mean(metrics['asd_values']) if metrics['asd_values'] else 999.0
            mean_hd95 = np.mean(metrics['hd95_values']) if metrics['hd95_values'] else 999.0
            
            std_hd = np.std(metrics['hd_values']) if metrics['hd_values'] else 0.0
            std_asd = np.std(metrics['asd_values']) if metrics['asd_values'] else 0.0
            std_hd95 = np.std(metrics['hd95_values']) if metrics['hd95_values'] else 0.0
            
            max_hd = np.max(metrics['hd_values']) if metrics['hd_values'] else 999.0
            max_asd = np.max(metrics['asd_values']) if metrics['asd_values'] else 999.0
            max_hd95 = np.max(metrics['hd95_values']) if metrics['hd95_values'] else 999.0
            
            summary[f'threshold_{threshold}'] = {
                # 🎯 分割质量指标
                'dice': float(dice),
                'iou': float(iou),
                'f1': float(f1),
                
                # 错误分析指标
                'fp_rate': float(fp_rate),
                'total_fp': int(fp),
                'total_fn': int(fn),
                'total_tp': int(tp),
                
                # 🔥 边界精确性指标 - 最重要的指标
                'mean_hausdorff': float(mean_hd),
                'std_hausdorff': float(std_hd),
                'max_hausdorff': float(max_hd),
                'mean_asd': float(mean_asd),
                'std_asd': float(std_asd),
                'max_asd': float(max_asd),
                'mean_hd95': float(mean_hd95),
                'std_hd95': float(std_hd95),
                'max_hd95': float(max_hd95),
                'valid_samples': len(metrics['hd_values'])  # 有效计算边界距离的样本数
            }
        
        return summary

class SimpleTester:
    """简化的测试器"""
    
    def __init__(self):
        self.device = torch.device(config.DEVICE)
        self.setup_directories()
        self.load_model()
        self.setup_data()
        self.metrics_calculator = MetricsCalculator()
    
    def setup_directories(self):
        """创建输出目录"""
        # 确保使用绝对路径并创建目录
        self.results_dir = os.path.abspath(config.RESULTS_DIR)
        self.viz_dir = os.path.join(self.results_dir, "visualizations")
        self.reports_dir = os.path.join(self.results_dir, "reports")
        
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        logger.info(f"输出目录已创建: {self.results_dir}")
        logger.info(f"可视化目录: {self.viz_dir}")
        logger.info(f"报告目录: {self.reports_dir}")
    
    def load_model(self):
        """加载模型"""
        logger.info(f"加载模型: {config.MODEL_PATH}")
        
        if not os.path.exists(config.MODEL_PATH):
            logger.error(f"模型文件不存在: {config.MODEL_PATH}")
            sys.exit(1)
        
        checkpoint = torch.load(config.MODEL_PATH, map_location=self.device)
        
        try:
            from segment_anything import sam_model_registry
            sam = sam_model_registry["vit_b"](checkpoint=config.SAM_MODEL_PATH)
            sam.to(self.device)
            
            self.model = DSCEnhancedSAMModel(sam, num_classes=2)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("模型加载成功！")
            logger.info(f"训练轮次: {checkpoint.get('epoch', 'Unknown')}")
            logger.info(f"最佳病灶Dice: {checkpoint.get('best_lesion_dice', 'Unknown')}")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            logger.error(f"请检查SAM模型路径: {config.SAM_MODEL_PATH}")
            sys.exit(1)
    
    def custom_collate_fn(self, batch):
        """自定义collate函数处理非张量数据"""
        images = torch.stack([item[0] for item in batch])
        masks = torch.stack([item[1] for item in batch])
        filenames = [item[2] for item in batch]
        original_sizes = [item[3] for item in batch]
        original_masks = [item[4] for item in batch]
        
        return images, masks, filenames, original_sizes, original_masks
    
    def setup_data(self):
        """设置测试数据"""
        self.test_dataset = LesionOnlyDataset(config.TEST_IMAGES_DIR, config.TEST_MASKS_DIR)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            collate_fn=self.custom_collate_fn
        )
        
        logger.info(f"有效测试样本: {len(self.test_dataset)} 个")
    
    def run_inference(self):
        """运行推理"""
        logger.info("开始推理...")
        
        all_results = []
        visualization_count = 0
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(self.test_loader, desc="推理中")):
                images, masks, filenames, original_sizes, original_masks = batch_data
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # 模型推理
                predictions, _ = self.model(images)
                pred_probs = torch.softmax(predictions, dim=1)
                
                # 处理每个样本
                for i in range(images.size(0)):
                    filename = filenames[i]
                    pred_prob = pred_probs[i, 1].cpu().numpy()  # 病灶类别概率
                    target_mask = masks[i].cpu().numpy()
                    original_size = original_sizes[i]  # (height, width)
                    
                    # 获取原始尺寸
                    height, width = original_size[0], original_size[1]
                    
                    # 调整到原始尺寸
                    pred_prob_resized = cv2.resize(pred_prob, (width, height))
                    target_mask_resized = cv2.resize(target_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                    
                    # 计算指标
                    sample_metrics = self.metrics_calculator.calculate_sample_metrics(
                        pred_prob_resized, target_mask_resized, filename
                    )
                    
                    # 保存结果
                    result = {
                        'filename': filename,
                        'pred_prob': pred_prob_resized,
                        'target_mask': target_mask_resized,
                        'metrics': sample_metrics
                    }
                    all_results.append(result)
                    
                    # 可视化部分结果
                    if config.SAVE_VISUALIZATIONS and visualization_count < config.MAX_VISUALIZATIONS:
                        self.visualize_result(result, visualization_count)
                        visualization_count += 1
        
        self.all_results = all_results
        logger.info(f"推理完成，处理了 {len(all_results)} 个样本")
    
    def visualize_result(self, result, idx):
        """简化的可视化"""
        filename = result['filename']
        pred_prob = result['pred_prob']
        target_mask = result['target_mask']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{filename}', fontsize=14)
        
        # Ground Truth
        axes[0, 0].imshow(target_mask, cmap='gray')
        axes[0, 0].set_title('Ground Truth')
        axes[0, 0].axis('off')
        
        # 预测概率
        im = axes[0, 1].imshow(pred_prob, cmap='hot', vmin=0, vmax=1)
        axes[0, 1].set_title('Prediction Probability')
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1])
        
        # 预测结果 (阈值=0.5)
        pred_mask_05 = (pred_prob > 0.5).astype(np.uint8)
        axes[0, 2].imshow(pred_mask_05, cmap='gray')
        axes[0, 2].set_title('Prediction (th=0.5)')
        axes[0, 2].axis('off')
        
        # 错误分析
        error_map = np.zeros((*pred_mask_05.shape, 3))
        error_map[target_mask == 1] = [0, 1, 0]  # TP - 绿色
        error_map[(pred_mask_05 == 1) & (target_mask == 0)] = [1, 0, 0]  # FP - 红色
        error_map[(pred_mask_05 == 0) & (target_mask == 1)] = [0, 0, 1]  # FN - 蓝色
        
        axes[1, 0].imshow(error_map)
        axes[1, 0].set_title('Error Analysis\nRed:FP, Blue:FN, Green:TP')
        axes[1, 0].axis('off')
        
        # 指标文本 - 突出分割专用指标
        metrics = result['metrics']
        hd = metrics['hausdorff_0.5']
        asd = metrics['asd_0.5']
        hd95 = metrics['hd95_0.5']
        
        # 格式化距离指标
        hd_str = f"{hd:.1f}" if hd < 900 else "∞"
        asd_str = f"{asd:.1f}" if asd < 900 else "∞"
        hd95_str = f"{hd95:.1f}" if hd95 < 900 else "∞"
        
        metrics_text = f"""🎯 分割质量 (阈值 0.5):
Dice: {metrics['dice_0.5']:.4f}
IoU: {metrics['iou_0.5']:.4f}
F1: {metrics['f1_0.5']:.4f}

🔥 边界精确性 (关键指标):
Hausdorff: {hd_str} px
ASD: {asd_str} px  
HD95: {hd95_str} px

📊 错误统计:
FP像素: {metrics['fp_count_0.5']}
FN像素: {metrics['fn_count_0.5']}
FP率: {metrics['fp_rate_0.5']:.4f}"""
        
        axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes, 
                       verticalalignment='top', fontsize=10, fontfamily='monospace')
        axes[1, 1].axis('off')
        
        # 概率分布
        axes[1, 2].hist(pred_prob.flatten(), bins=30, alpha=0.7, color='blue')
        axes[1, 2].axvline(0.5, color='red', linestyle='--', label='Threshold 0.5')
        axes[1, 2].set_title('Probability Distribution')
        axes[1, 2].set_xlabel('Probability')
        axes[1, 2].legend()
        
        plt.tight_layout()
        
        # 确保文件名安全
        safe_filename = filename.replace('.jpg', '').replace('.jpeg', '').replace('[', '_').replace(']', '_')
        save_path = os.path.join(self.viz_dir, f"sample_{idx+1:03d}_{safe_filename}.png")
        
        # 再次确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def convert_to_json_serializable(self, obj):
        """递归转换数据为JSON可序列化格式"""
        if isinstance(obj, dict):
            return {key: self.convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def generate_report(self):
        """生成简化报告"""
        logger.info("生成分析报告...")
        
        # 获取汇总指标
        summary_metrics = self.metrics_calculator.get_summary_metrics()
        
        # 样本级数据
        sample_df = pd.DataFrame(self.metrics_calculator.sample_metrics)
        
        # 找出最佳阈值
        best_threshold = 0.5
        best_f1 = 0
        for th in config.CONFIDENCE_THRESHOLDS:
            f1 = summary_metrics[f'threshold_{th}']['f1']
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = th
        
        # 生成报告 - 确保所有数据都是JSON可序列化的
        report = {
            'test_config': {
                'total_samples': int(len(self.test_dataset)),
                'lesion_type': str(config.LESION_NAME),
                'model_path': str(config.MODEL_PATH)
            },
            'summary_metrics': self.convert_to_json_serializable(summary_metrics),
            'best_threshold': float(best_threshold),
            'best_f1': float(best_f1),
            'sample_statistics': {
                'mean_lesion_area': float(np.mean(self.test_dataset.lesion_areas)),
                'std_lesion_area': float(np.std(self.test_dataset.lesion_areas)),
                'min_lesion_area': int(min(self.test_dataset.lesion_areas)),
                'max_lesion_area': int(max(self.test_dataset.lesion_areas))
            }
        }
        
        # 保存报告
        report_path = os.path.join(self.reports_dir, "test_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 保存样本数据
        sample_csv_path = os.path.join(self.reports_dir, "sample_metrics.csv")
        sample_df.to_csv(sample_csv_path, index=False, encoding='utf-8')
        
        # 创建性能图表
        self.create_performance_chart(summary_metrics, sample_df)
        
        logger.info(f"报告已保存到: {report_path}")
        return report
    
    def create_performance_chart(self, summary_metrics, sample_df):
        """创建性能图表"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        thresholds = config.CONFIDENCE_THRESHOLDS
        
        # 1. 分割质量指标随阈值变化 (重点关注)
        ax1 = axes[0, 0]
        metrics_to_plot = ['dice', 'iou', 'f1']  # 专注于最重要的指标
        colors = ['red', 'blue', 'green']
        for i, metric in enumerate(metrics_to_plot):
            values = [summary_metrics[f'threshold_{th}'][metric] for th in thresholds]
            ax1.plot(thresholds, values, marker='o', label=metric.upper(), 
                    linewidth=3, color=colors[i])
        ax1.set_xlabel('Confidence Threshold')
        ax1.set_ylabel('分割质量')
        ax1.set_title('🔥 核心分割质量指标')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # 2. 🔥 边界距离指标 - 对FP问题最敏感
        ax2 = axes[0, 1]
        hd_values = []
        asd_values = []
        hd95_values = []
        
        for th in thresholds:
            hd = summary_metrics[f'threshold_{th}']['mean_hausdorff']
            asd = summary_metrics[f'threshold_{th}']['mean_asd']  
            hd95 = summary_metrics[f'threshold_{th}']['mean_hd95']
            
            # 过滤异常值
            hd_values.append(hd if hd < 900 else np.nan)
            asd_values.append(asd if asd < 900 else np.nan)
            hd95_values.append(hd95 if hd95 < 900 else np.nan)
        
        ax2.plot(thresholds, hd_values, 'r-o', label='Hausdorff', linewidth=2, markersize=6)
        ax2.plot(thresholds, asd_values, 'b-s', label='ASD', linewidth=2, markersize=6)
        ax2.plot(thresholds, hd95_values, 'g-^', label='HD95', linewidth=2, markersize=6)
        
        ax2.set_xlabel('Confidence Threshold')
        ax2.set_ylabel('距离 (像素)')
        ax2.set_title('🎯 边界精确性指标 (越小越好)')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 添加参考线
        if not all(np.isnan(hd_values + asd_values + hd95_values)):
            ax2.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='良好阈值')
            ax2.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='可接受阈值')
        
        # 3. Dice分数分布
        ax3 = axes[0, 2]
        dice_scores = sample_df['dice_0.5']
        ax3.hist(dice_scores, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(dice_scores.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'均值: {dice_scores.mean():.3f}')
        ax3.set_xlabel('Dice Score')
        ax3.set_ylabel('Sample Count')
        ax3.set_title('Dice分数分布 (阈值=0.5)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 病灶大小 vs 性能
        ax4 = axes[1, 0]
        lesion_areas = sample_df['lesion_area']
        dice_scores = sample_df['dice_0.5']
        scatter = ax4.scatter(lesion_areas, dice_scores, alpha=0.6, c=dice_scores, cmap='viridis')
        ax4.set_xlabel('Lesion Area (pixels)')
        ax4.set_ylabel('Dice Score')
        ax4.set_title('病灶大小 vs Dice表现')
        plt.colorbar(scatter, ax=ax4)
        
        # 添加趋势线
        if len(lesion_areas) > 1:
            z = np.polyfit(lesion_areas, dice_scores, 1)
            p = np.poly1d(z)
            ax4.plot(lesion_areas, p(lesion_areas), "r--", alpha=0.8)
        
        # 5. FP分布
        ax5 = axes[1, 1]
        fp_counts = sample_df['fp_count_0.5']
        ax5.hist(fp_counts, bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax5.axvline(fp_counts.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'均值: {fp_counts.mean():.1f}')
        ax5.set_xlabel('FP Count')
        ax5.set_ylabel('Sample Count')
        ax5.set_title('FP数量分布 (阈值=0.5)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. 性能总结
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # 找最佳阈值
        best_th = thresholds[0]
        best_f1 = 0
        for th in thresholds:
            f1 = summary_metrics[f'threshold_{th}']['f1']
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
        
        # 获取边界距离指标
        best_hd = summary_metrics[f'threshold_{best_th}']['mean_hausdorff']
        best_asd = summary_metrics[f'threshold_{best_th}']['mean_asd']
        best_hd95 = summary_metrics[f'threshold_{best_th}']['mean_hd95']
        
        hd_str = f"{best_hd:.1f}px" if best_hd < 900 else "计算失败"
        asd_str = f"{best_asd:.1f}px" if best_asd < 900 else "计算失败"
        hd95_str = f"{best_hd95:.1f}px" if best_hd95 < 900 else "计算失败"
        
        summary_text = f"""🔥 医学分割专用评估:

📊 数据集统计:
• 有效样本: {len(sample_df)}个
• 病灶面积: {lesion_areas.mean():.0f}±{lesion_areas.std():.0f} px

🎯 分割质量 (阈值={best_th}):
• Dice系数: {summary_metrics[f'threshold_{best_th}']['dice']:.4f}
• IoU: {summary_metrics[f'threshold_{best_th}']['iou']:.4f}  
• F1分数: {summary_metrics[f'threshold_{best_th}']['f1']:.4f}

🔥 边界精确性 (最重要!):
• Hausdorff距离: {hd_str}
• 平均表面距离: {asd_str}
• HD95: {hd95_str}

📈 质量分布:
• 高质量(Dice>0.8): {np.sum(dice_scores > 0.8)}/{len(dice_scores)} ({100*np.sum(dice_scores > 0.8)/len(dice_scores):.1f}%)
• 优秀(Dice>0.9): {np.sum(dice_scores > 0.9)}/{len(dice_scores)} ({100*np.sum(dice_scores > 0.9)/len(dice_scores):.1f}%)

📊 错误分析:
• FP/FN比: {summary_metrics[f'threshold_{best_th}']['total_fp']/(summary_metrics[f'threshold_{best_th}']['total_fn']+1e-8):.1f}

🎯 边界质量评估标准:
• <5px: 优秀 | 5-10px: 良好 | >10px: 需改进
        """
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图表
        chart_path = os.path.join(self.reports_dir, "performance_analysis.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"性能图表已保存到: {chart_path}")
    
    def print_summary(self, report):
        """打印测试摘要"""
        logger.info("\n" + "="*60)
        logger.info("🔍 测试结果摘要")
        logger.info("="*60)
        
        config_info = report['test_config']
        logger.info(f"📊 测试数据: {config_info['total_samples']}个{config_info['lesion_type']}样本")
        
        best_th = report['best_threshold']
        best_metrics = report['summary_metrics'][f'threshold_{best_th}']
        
        logger.info(f"\n🎯 分割质量指标 (阈值={best_th}):")
        logger.info(f"   • Dice系数: {best_metrics['dice']:.4f}")
        logger.info(f"   • IoU: {best_metrics['iou']:.4f}")
        logger.info(f"   • F1分数: {best_metrics['f1']:.4f}")
        
        logger.info(f"\n🔥 边界精确性指标 (关键评估!):")
        hd = best_metrics['mean_hausdorff']
        asd = best_metrics['mean_asd']
        hd95 = best_metrics['mean_hd95']
        
        if hd < 999:
            logger.info(f"   • 平均Hausdorff距离: {hd:.2f} 像素")
            logger.info(f"   • 平均表面距离(ASD): {asd:.2f} 像素")
            logger.info(f"   • HD95: {hd95:.2f} 像素")
            logger.info(f"   • 最大边界偏移: {best_metrics['max_hausdorff']:.2f} 像素")
            
            # 根据边界距离评估质量
            if hd < 5:
                logger.info(f"   ✅ 边界质量: 优秀 (Hausdorff < 5px)")
            elif hd < 10:
                logger.info(f"   🟡 边界质量: 良好 (Hausdorff 5-10px)")
            else:
                logger.info(f"   🔴 边界质量: 需改进 (Hausdorff > 10px)")
        else:
            logger.info(f"   ⚠️  边界距离计算失败 (可能存在严重分割错误)")
        
        logger.info(f"\n📊 错误分析:")
        logger.info(f"   • FP率: {best_metrics['fp_rate']:.4f} ({best_metrics['fp_rate']*100:.2f}%)")
        logger.info(f"   • 总FP像素: {best_metrics['total_fp']:,}")
        logger.info(f"   • 总FN像素: {best_metrics['total_fn']:,}")
        
        # FP vs FN 分析
        fp_fn_ratio = best_metrics['total_fp'] / (best_metrics['total_fn'] + 1e-8)
        if fp_fn_ratio > 2:
            logger.info(f"   🔴 问题: 过度分割严重 (FP/FN比={fp_fn_ratio:.1f})")
        elif fp_fn_ratio > 1.2:
            logger.info(f"   🟡 问题: 轻微过度分割 (FP/FN比={fp_fn_ratio:.1f})")
        elif fp_fn_ratio < 0.5:
            logger.info(f"   🔵 问题: 分割不足 (FP/FN比={fp_fn_ratio:.1f})")
        else:
            logger.info(f"   ✅ FP/FN平衡良好 (比值={fp_fn_ratio:.1f})")
        
        # 样本统计
        sample_df = pd.DataFrame(self.metrics_calculator.sample_metrics)
        dice_scores = sample_df['dice_0.5']
        high_quality = np.sum(dice_scores > 0.8)
        excellent = np.sum(dice_scores > 0.9)
        
        logger.info(f"\n📈 样本质量分布:")
        logger.info(f"   • 高质量样本 (Dice>0.8): {high_quality}/{len(dice_scores)} ({100*high_quality/len(dice_scores):.1f}%)")
        logger.info(f"   • 优秀样本 (Dice>0.9): {excellent}/{len(dice_scores)} ({100*excellent/len(dice_scores):.1f}%)")
        
        logger.info("="*60 + "\n")
    
    def run_test(self):
        """运行完整测试"""
        logger.info("开始模型测试...")
        
        # 1. 推理
        self.run_inference()
        
        # 2. 生成报告
        report = self.generate_report()
        
        # 3. 打印摘要
        self.print_summary(report)
        
        logger.info(f"测试完成！结果保存在: {self.results_dir}")
        return report

def main():
    """主函数"""
    logger.info("🔍 启动病灶分割模型测试...")
    
    # 检查路径
    if not os.path.exists(config.TEST_IMAGES_DIR):
        logger.error(f"测试图像目录不存在: {config.TEST_IMAGES_DIR}")
        return
    
    if not os.path.exists(config.TEST_MASKS_DIR):
        logger.error(f"测试掩码目录不存在: {config.TEST_MASKS_DIR}")
        return
    
    if not os.path.exists(config.MODEL_PATH):
        logger.error(f"模型文件不存在: {config.MODEL_PATH}")
        logger.info("请修改 MODEL_PATH 为您的模型路径")
        return
    
    # 运行测试
    tester = SimpleTester()
    report = tester.run_test()
    
    logger.info("🎉 测试完成！")

if __name__ == "__main__":
    main() 