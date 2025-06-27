#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔥 SAM声带病灶分割 - 完整测试评估脚本
科学评价体系：只计算存在的类别，避免统计陷阱
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from tqdm import tqdm
import logging
from collections import defaultdict, Counter
import pandas as pd
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class TestConfig:
    """测试配置"""
    # 路径配置
    TEST_IMAGES_DIR = "autodl-tmp/SAM/6classdata/images"
    TEST_MASKS_DIR = "autodl-tmp/SAM/6classdata/masks"
    MODEL_PATH = "autodl-tmp/SAM/results/models/run_2/models/best_model.pth"
    RESULTS_DIR = "autodl-tmp/SAM/results/fold_test"
    
    # 基础配置
    NUM_CLASSES = 6
    IMAGE_SIZE = 1024
    BATCH_SIZE = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 类别映射
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
    
    # 病灶类别索引
    LESION_CLASSES = [3, 4, 5]
    LESION_NAMES = ["声带小结", "声带白斑", "声带乳头状瘤"]
    
    # SAM配置
    PIXEL_MEAN = [123.675, 116.28, 103.53]
    PIXEL_STD = [58.395, 57.12, 57.375]

config = TestConfig()

class ComprehensiveMetrics:
    """全面的评价指标计算器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        # 基础统计
        self.total_images = 0
        self.total_pixels = 0
        self.total_correct = 0
        
        # 类别级统计
        self.class_stats = {
            'intersections': np.zeros(config.NUM_CLASSES),
            'unions': np.zeros(config.NUM_CLASSES),
            'pred_counts': np.zeros(config.NUM_CLASSES),
            'true_counts': np.zeros(config.NUM_CLASSES),
            'image_counts': np.zeros(config.NUM_CLASSES)  # 每个类别出现在多少张图像中
        }
        
        # 图像级统计
        self.image_metrics = []
        
        # 病灶专项统计
        self.lesion_stats = {
            'detection_counts': np.zeros(len(config.LESION_CLASSES)),  # 检测到的次数
            'ground_truth_counts': np.zeros(len(config.LESION_CLASSES)),  # 真实存在的次数
            'size_analysis': {
                'small': {'count': 0, 'ious': []},    # <1000像素
                'medium': {'count': 0, 'ious': []},   # 1000-5000像素
                'large': {'count': 0, 'ious': []}     # >5000像素
            }
        }
        
        # 多病灶分析
        self.multi_lesion_stats = {
            'single_lesion_images': {'count': 0, 'ious': []},
            'multi_lesion_images': {'count': 0, 'ious': []},
            'no_lesion_images': {'count': 0, 'ious': []}
        }
    
    def update(self, predictions, targets, image_name):
        """更新指标"""
        predictions = torch.argmax(predictions, dim=1)
        
        self.total_images += 1
        self.total_pixels += targets.numel()
        self.total_correct += (predictions == targets).sum().item()
        
        # 分析当前图像
        image_analysis = self._analyze_single_image(predictions, targets, image_name)
        self.image_metrics.append(image_analysis)
        
        # 更新类别统计
        self._update_class_stats(predictions, targets)
        
        # 更新病灶统计
        self._update_lesion_stats(predictions, targets)
        
        # 更新多病灶统计
        self._update_multi_lesion_stats(predictions, targets, image_analysis)
    
    def _analyze_single_image(self, predictions, targets, image_name):
        """分析单张图像"""
        pred = predictions[0].cpu().numpy()
        target = targets[0].cpu().numpy()
        
        # 计算该图像的类别IoU
        image_class_ious = {}
        present_classes = []
        
        for class_idx in range(config.NUM_CLASSES):
            pred_mask = (pred == class_idx)
            target_mask = (target == class_idx)
            
            intersection = np.sum(pred_mask & target_mask)
            union = np.sum(pred_mask | target_mask)
            
            if union > 0:
                iou = intersection / union
                image_class_ious[config.CLASS_NAMES[class_idx]] = iou
                present_classes.append(class_idx)
        
        # 计算该图像的mIoU（只计算存在的类别）
        image_miou = np.mean(list(image_class_ious.values())) if image_class_ious else 0
        
        # 病灶分析
        lesion_analysis = self._analyze_lesions_in_image(pred, target)
        
        image_result = {
            'image_name': image_name,
            'present_classes': [config.CLASS_NAMES[i] for i in present_classes],
            'class_ious': image_class_ious,
            'image_miou': image_miou,
            'lesion_analysis': lesion_analysis,
            'pixel_accuracy': np.sum(pred == target) / target.size
        }
        
        return image_result
    
    def _analyze_lesions_in_image(self, pred, target):
        """分析图像中的病灶"""
        lesion_analysis = {
            'lesions_present': [],
            'lesions_detected': [],
            'lesion_ious': {},
            'lesion_sizes': {}
        }
        
        for i, lesion_class in enumerate(config.LESION_CLASSES):
            target_mask = (target == lesion_class)
            pred_mask = (pred == lesion_class)
            
            if np.sum(target_mask) > 0:  # 真实存在该病灶
                lesion_name = config.LESION_NAMES[i]
                lesion_analysis['lesions_present'].append(lesion_name)
                
                # 计算病灶大小
                lesion_size = np.sum(target_mask)
                lesion_analysis['lesion_sizes'][lesion_name] = lesion_size
                
                # 计算IoU
                intersection = np.sum(pred_mask & target_mask)
                union = np.sum(pred_mask | target_mask)
                
                if union > 0:
                    iou = intersection / union
                    lesion_analysis['lesion_ious'][lesion_name] = iou
                    
                    if intersection > 0:  # 检测到了
                        lesion_analysis['lesions_detected'].append(lesion_name)
        
        return lesion_analysis
    
    def _update_class_stats(self, predictions, targets):
        """更新类别统计"""
        pred = predictions[0].cpu().numpy()
        target = targets[0].cpu().numpy()
        
        for class_idx in range(config.NUM_CLASSES):
            pred_mask = (pred == class_idx)
            target_mask = (target == class_idx)
            
            intersection = np.sum(pred_mask & target_mask)
            union = np.sum(pred_mask | target_mask)
            
            if union > 0:
                self.class_stats['intersections'][class_idx] += intersection
                self.class_stats['unions'][class_idx] += union
                self.class_stats['image_counts'][class_idx] += 1
            
            self.class_stats['pred_counts'][class_idx] += np.sum(pred_mask)
            self.class_stats['true_counts'][class_idx] += np.sum(target_mask)
    
    def _update_lesion_stats(self, predictions, targets):
        """更新病灶统计"""
        pred = predictions[0].cpu().numpy()
        target = targets[0].cpu().numpy()
        
        for i, lesion_class in enumerate(config.LESION_CLASSES):
            target_mask = (target == lesion_class)
            pred_mask = (pred == lesion_class)
            
            if np.sum(target_mask) > 0:  # 真实存在
                self.lesion_stats['ground_truth_counts'][i] += 1
                
                # 大小分析
                lesion_size = np.sum(target_mask)
                intersection = np.sum(pred_mask & target_mask)
                union = np.sum(pred_mask | target_mask)
                
                if union > 0:
                    iou = intersection / union
                    
                    if lesion_size < 1000:
                        self.lesion_stats['size_analysis']['small']['count'] += 1
                        self.lesion_stats['size_analysis']['small']['ious'].append(iou)
                    elif lesion_size < 5000:
                        self.lesion_stats['size_analysis']['medium']['count'] += 1
                        self.lesion_stats['size_analysis']['medium']['ious'].append(iou)
                    else:
                        self.lesion_stats['size_analysis']['large']['count'] += 1
                        self.lesion_stats['size_analysis']['large']['ious'].append(iou)
                
                if intersection > 0:  # 检测到了
                    self.lesion_stats['detection_counts'][i] += 1
    
    def _update_multi_lesion_stats(self, predictions, targets, image_analysis):
        """更新多病灶统计"""
        lesions_count = len(image_analysis['lesion_analysis']['lesions_present'])
        image_miou = image_analysis['image_miou']
        
        if lesions_count == 0:
            self.multi_lesion_stats['no_lesion_images']['count'] += 1
            self.multi_lesion_stats['no_lesion_images']['ious'].append(image_miou)
        elif lesions_count == 1:
            self.multi_lesion_stats['single_lesion_images']['count'] += 1
            self.multi_lesion_stats['single_lesion_images']['ious'].append(image_miou)
        else:
            self.multi_lesion_stats['multi_lesion_images']['count'] += 1
            self.multi_lesion_stats['multi_lesion_images']['ious'].append(image_miou)
    
    def compute_final_metrics(self):
        """计算最终指标"""
        results = {}
        
        # 1. 基础指标
        results['basic_metrics'] = {
            'total_images': self.total_images,
            'pixel_accuracy': self.total_correct / self.total_pixels if self.total_pixels > 0 else 0
        }
        
        # 2. 类别级指标
        class_metrics = self._compute_class_metrics()
        results['class_metrics'] = class_metrics
        
        # 3. 整体mIoU（优化版和传统版）
        results['overall_metrics'] = self._compute_overall_metrics(class_metrics)
        
        # 4. 病灶专项指标
        results['lesion_metrics'] = self._compute_lesion_metrics(class_metrics)
        
        # 5. 多病灶分析
        results['multi_lesion_analysis'] = self._compute_multi_lesion_metrics()
        
        # 6. 图像级分析
        results['image_level_analysis'] = self._compute_image_level_metrics()
        
        return results
    
    def _compute_class_metrics(self):
        """计算类别指标"""
        class_metrics = {}
        
        for class_idx, class_name in enumerate(config.CLASS_NAMES):
            if self.class_stats['unions'][class_idx] > 0:
                iou = self.class_stats['intersections'][class_idx] / self.class_stats['unions'][class_idx]
                
                # Precision, Recall, F1
                precision = (self.class_stats['intersections'][class_idx] / 
                           self.class_stats['pred_counts'][class_idx] if self.class_stats['pred_counts'][class_idx] > 0 else 0)
                recall = (self.class_stats['intersections'][class_idx] / 
                         self.class_stats['true_counts'][class_idx] if self.class_stats['true_counts'][class_idx] > 0 else 0)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                class_metrics[class_name] = {
                    'IoU': iou,
                    'Precision': precision,
                    'Recall': recall,
                    'F1': f1,
                    'image_frequency': self.class_stats['image_counts'][class_idx] / self.total_images,
                    'present_in_images': int(self.class_stats['image_counts'][class_idx])
                }
            else:
                class_metrics[class_name] = {
                    'IoU': 0.0,
                    'Precision': 0.0,
                    'Recall': 0.0,
                    'F1': 0.0,
                    'image_frequency': 0.0,
                    'present_in_images': 0
                }
        
        return class_metrics
    
    def _compute_overall_metrics(self, class_metrics):
        """计算整体指标"""
        # 传统方式（包含所有类别）
        all_ious = [metrics['IoU'] for metrics in class_metrics.values()]
        all_f1s = [metrics['F1'] for metrics in class_metrics.values()]
        
        traditional_miou = np.mean(all_ious)
        traditional_mf1 = np.mean(all_f1s)
        
        # 优化方式（只计算存在的类别）
        existing_ious = [metrics['IoU'] for metrics in class_metrics.values() if metrics['present_in_images'] > 0]
        existing_f1s = [metrics['F1'] for metrics in class_metrics.values() if metrics['present_in_images'] > 0]
        existing_classes = [name for name, metrics in class_metrics.items() if metrics['present_in_images'] > 0]
        
        optimized_miou = np.mean(existing_ious) if existing_ious else 0
        optimized_mf1 = np.mean(existing_f1s) if existing_f1s else 0
        
        return {
            'traditional_mIoU': traditional_miou,
            'traditional_mF1': traditional_mf1,
            'optimized_mIoU': optimized_miou,
            'optimized_mF1': optimized_mf1,
            'existing_classes': existing_classes,
            'improvement': {
                'mIoU_improvement': optimized_miou - traditional_miou,
                'mF1_improvement': optimized_mf1 - traditional_mf1
            }
        }
    
    def _compute_lesion_metrics(self, class_metrics):
        """计算病灶专项指标"""
        lesion_metrics = {}
        
        # 各病灶的详细指标
        for i, lesion_name in enumerate(config.LESION_NAMES):
            # 检测率
            detection_rate = (self.lesion_stats['detection_counts'][i] / 
                            self.lesion_stats['ground_truth_counts'][i] 
                            if self.lesion_stats['ground_truth_counts'][i] > 0 else 0)
            
            lesion_metrics[lesion_name] = {
                'IoU': class_metrics[lesion_name]['IoU'],
                'F1': class_metrics[lesion_name]['F1'],
                'detection_rate': detection_rate,
                'ground_truth_count': int(self.lesion_stats['ground_truth_counts'][i]),
                'detected_count': int(self.lesion_stats['detection_counts'][i])
            }
        
        # 病灶整体指标
        lesion_ious = [lesion_metrics[name]['IoU'] for name in config.LESION_NAMES 
                      if lesion_metrics[name]['ground_truth_count'] > 0]
        lesion_f1s = [lesion_metrics[name]['F1'] for name in config.LESION_NAMES 
                     if lesion_metrics[name]['ground_truth_count'] > 0]
        
        overall_lesion_metrics = {
            'lesion_mIoU_optimized': np.mean(lesion_ious) if lesion_ious else 0,
            'lesion_mF1_optimized': np.mean(lesion_f1s) if lesion_f1s else 0,
            'lesion_mIoU_traditional': np.mean([lesion_metrics[name]['IoU'] for name in config.LESION_NAMES]),
            'overall_detection_rate': np.mean([lesion_metrics[name]['detection_rate'] for name in config.LESION_NAMES])
        }
        
        # 病灶大小敏感性分析
        size_analysis = {}
        for size_category, data in self.lesion_stats['size_analysis'].items():
            if data['count'] > 0:
                size_analysis[size_category] = {
                    'count': data['count'],
                    'avg_iou': np.mean(data['ious']),
                    'std_iou': np.std(data['ious'])
                }
            else:
                size_analysis[size_category] = {'count': 0, 'avg_iou': 0, 'std_iou': 0}
        
        return {
            'individual_lesions': lesion_metrics,
            'overall_lesion_metrics': overall_lesion_metrics,
            'size_sensitivity_analysis': size_analysis
        }
    
    def _compute_multi_lesion_metrics(self):
        """计算多病灶分析指标"""
        multi_lesion_analysis = {}
        
        for category, data in self.multi_lesion_stats.items():
            if data['count'] > 0:
                multi_lesion_analysis[category] = {
                    'count': data['count'],
                    'percentage': data['count'] / self.total_images * 100,
                    'avg_miou': np.mean(data['ious']),
                    'std_miou': np.std(data['ious'])
                }
            else:
                multi_lesion_analysis[category] = {
                    'count': 0, 'percentage': 0, 'avg_miou': 0, 'std_miou': 0
                }
        
        return multi_lesion_analysis
    
    def _compute_image_level_metrics(self):
        """计算图像级分析"""
        image_mious = [img['image_miou'] for img in self.image_metrics]
        
        return {
            'avg_image_miou': np.mean(image_mious),
            'std_image_miou': np.std(image_mious),
            'min_image_miou': np.min(image_mious),
            'max_image_miou': np.max(image_mious),
            'median_image_miou': np.median(image_mious)
        }

class TestDataset:
    """测试数据集"""
    
    def __init__(self, images_dir, masks_dir):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        
        # 获取图像文件
        self.image_files = []
        for file in os.listdir(images_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                mask_file = file.replace('.jpg', '.png').replace('.jpeg', '.png')
                mask_path = os.path.join(masks_dir, mask_file)
                if os.path.exists(mask_path):
                    self.image_files.append(file)
        
        logger.info(f"找到 {len(self.image_files)} 个测试样本")
    
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
        
        # 应用ID映射
        mask = self.apply_id_mapping(mask)
        
        # 调整尺寸
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

def load_model(model_path):
    """加载训练好的模型"""
    logger.info(f"加载模型: {model_path}")
    
    try:
        # 导入训练脚本中的模型类
        sys.path.append('.')
        from train_sobel_optimized import EnhancedSAMModel
        from segment_anything import sam_model_registry
        
        # 加载检查点
        checkpoint = torch.load(model_path, map_location=config.DEVICE)
        
        # 创建SAM模型
        sam = sam_model_registry["vit_b"](checkpoint="models/sam_vit_b_01ec64.pth")
        sam.to(config.DEVICE)
        
        # 创建增强模型
        model = EnhancedSAMModel(sam, config.NUM_CLASSES)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(config.DEVICE)
        model.eval()
        
        logger.info("模型加载成功！")
        return model
        
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return None

def run_inference(model, dataset, metrics):
    """运行推理"""
    logger.info("开始推理...")
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="推理进度"):
            image, mask, image_name = dataset[idx]
            
            # 添加batch维度
            image = image.unsqueeze(0).to(config.DEVICE)
            mask = mask.unsqueeze(0).to(config.DEVICE)
            
            # 模型推理
            try:
                predictions, _ = model(image)
                metrics.update(predictions, mask, image_name)
            except Exception as e:
                logger.error(f"推理失败 {image_name}: {e}")
                continue
    
    logger.info("推理完成！")

def generate_comprehensive_report(metrics_results, output_dir):
    """生成全面的测试报告"""
    logger.info("生成测试报告...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 生成JSON报告
    json_report_path = os.path.join(output_dir, "comprehensive_test_report.json")
    with open(json_report_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_results, f, indent=2, ensure_ascii=False, default=str)
    
    # 2. 生成可读性报告
    readable_report_path = os.path.join(output_dir, "test_report_readable.txt")
    generate_readable_report(metrics_results, readable_report_path)
    
    # 3. 生成CSV详细数据
    csv_report_path = os.path.join(output_dir, "detailed_class_metrics.csv")
    generate_csv_report(metrics_results, csv_report_path)
    
    # 4. 生成病灶详细CSV
    lesion_csv_path = os.path.join(output_dir, "lesion_detailed_metrics.csv")
    generate_lesion_csv_report(metrics_results, lesion_csv_path)
    
    # 5. 生成图像级详细CSV
    image_csv_path = os.path.join(output_dir, "image_level_metrics.csv")
    generate_image_csv_report(metrics_results, image_csv_path)
    
    logger.info(f"测试报告已生成到: {output_dir}")

def generate_readable_report(results, output_path):
    """生成可读性报告"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("🔥 SAM声带病灶分割 - 全面测试评估报告\n")
        f.write("=" * 80 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 基础信息
        f.write("📊 测试基础信息\n")
        f.write("-" * 40 + "\n")
        f.write(f"测试图像总数: {results['basic_metrics']['total_images']}\n")
        f.write(f"整体像素准确率: {results['basic_metrics']['pixel_accuracy']:.4f} ({results['basic_metrics']['pixel_accuracy']*100:.2f}%)\n\n")
        
        # 整体mIoU对比
        f.write("🎯 整体mIoU对比分析\n")
        f.write("-" * 40 + "\n")
        overall = results['overall_metrics']
        f.write(f"传统mIoU (包含所有类别): {overall['traditional_mIoU']:.4f} ({overall['traditional_mIoU']*100:.2f}%)\n")
        f.write(f"优化mIoU (只计算存在类别): {overall['optimized_mIoU']:.4f} ({overall['optimized_mIoU']*100:.2f}%)\n")
        f.write(f"mIoU提升: {overall['improvement']['mIoU_improvement']:+.4f} ({overall['improvement']['mIoU_improvement']*100:+.2f}%)\n")
        f.write(f"存在的类别: {', '.join(overall['existing_classes'])}\n\n")
        
        # 各类别详细表现
        f.write("🔍 各类别详细表现\n")
        f.write("-" * 40 + "\n")
        for class_name, metrics in results['class_metrics'].items():
            status = ""
            if class_name in config.LESION_NAMES:
                if metrics['IoU'] < 0.1:
                    status = "😰 需要关注"
                elif metrics['IoU'] < 0.3:
                    status = "😐 有待提高"
                elif metrics['IoU'] < 0.5:
                    status = "😊 不错"
                else:
                    status = "🔥 优秀"
            else:
                if metrics['IoU'] < 0.5:
                    status = "😐 一般"
                elif metrics['IoU'] < 0.7:
                    status = "😊 不错"
                else:
                    status = "🔥 优秀"
            
            f.write(f"{class_name}: IoU={metrics['IoU']:.4f} | F1={metrics['F1']:.4f} | 出现在{metrics['present_in_images']}张图像中 {status}\n")
        f.write("\n")
        
        # 病灶专项分析
        f.write("🏥 病灶专项分析\n")
        f.write("-" * 40 + "\n")
        lesion_metrics = results['lesion_metrics']
        f.write(f"病灶整体mIoU (优化): {lesion_metrics['overall_lesion_metrics']['lesion_mIoU_optimized']:.4f} ({lesion_metrics['overall_lesion_metrics']['lesion_mIoU_optimized']*100:.2f}%)\n")
        f.write(f"病灶整体检测率: {lesion_metrics['overall_lesion_metrics']['overall_detection_rate']:.4f} ({lesion_metrics['overall_lesion_metrics']['overall_detection_rate']*100:.2f}%)\n\n")
        
        f.write("各病灶详细表现:\n")
        for lesion_name, metrics in lesion_metrics['individual_lesions'].items():
            f.write(f"  {lesion_name}:\n")
            f.write(f"    IoU: {metrics['IoU']:.4f} ({metrics['IoU']*100:.2f}%)\n")
            f.write(f"    检测率: {metrics['detection_rate']:.4f} ({metrics['detection_rate']*100:.2f}%)\n")
            f.write(f"    真实存在: {metrics['ground_truth_count']}次 | 成功检测: {metrics['detected_count']}次\n")
        f.write("\n")
        
        # 病灶大小敏感性分析
        f.write("📏 病灶大小敏感性分析\n")
        f.write("-" * 40 + "\n")
        size_analysis = lesion_metrics['size_sensitivity_analysis']
        f.write(f"小病灶 (<1000像素): {size_analysis['small']['count']}个, 平均IoU={size_analysis['small']['avg_iou']:.4f}\n")
        f.write(f"中等病灶 (1000-5000像素): {size_analysis['medium']['count']}个, 平均IoU={size_analysis['medium']['avg_iou']:.4f}\n")
        f.write(f"大病灶 (>5000像素): {size_analysis['large']['count']}个, 平均IoU={size_analysis['large']['avg_iou']:.4f}\n\n")
        
        # 多病灶图像分析
        f.write("🔬 多病灶图像分析\n")
        f.write("-" * 40 + "\n")
        multi_lesion = results['multi_lesion_analysis']
        f.write(f"无病灶图像: {multi_lesion['no_lesion_images']['count']}张 ({multi_lesion['no_lesion_images']['percentage']:.1f}%), 平均mIoU={multi_lesion['no_lesion_images']['avg_miou']:.4f}\n")
        f.write(f"单病灶图像: {multi_lesion['single_lesion_images']['count']}张 ({multi_lesion['single_lesion_images']['percentage']:.1f}%), 平均mIoU={multi_lesion['single_lesion_images']['avg_miou']:.4f}\n")
        f.write(f"多病灶图像: {multi_lesion['multi_lesion_images']['count']}张 ({multi_lesion['multi_lesion_images']['percentage']:.1f}%), 平均mIoU={multi_lesion['multi_lesion_images']['avg_miou']:.4f}\n\n")
        
        # 图像级统计
        f.write("📈 图像级统计分析\n")
        f.write("-" * 40 + "\n")
        image_stats = results['image_level_analysis']
        f.write(f"平均图像mIoU: {image_stats['avg_image_miou']:.4f} ± {image_stats['std_image_miou']:.4f}\n")
        f.write(f"最佳图像mIoU: {image_stats['max_image_miou']:.4f}\n")
        f.write(f"最差图像mIoU: {image_stats['min_image_miou']:.4f}\n")
        f.write(f"中位数mIoU: {image_stats['median_image_miou']:.4f}\n\n")
        
        # 总结建议
        f.write("💡 总结与建议\n")
        f.write("-" * 40 + "\n")
        
        # 根据结果给出建议
        optimized_miou = overall['optimized_mIoU']
        lesion_miou = lesion_metrics['overall_lesion_metrics']['lesion_mIoU_optimized']
        
        if optimized_miou > 0.8:
            f.write("✅ 模型整体表现优秀！\n")
        elif optimized_miou > 0.6:
            f.write("😊 模型整体表现良好，有进一步优化空间。\n")
        else:
            f.write("😐 模型整体表现一般，建议继续训练或调整策略。\n")
        
        if lesion_miou > 0.5:
            f.write("✅ 病灶分割能力强！\n")
        elif lesion_miou > 0.3:
            f.write("😊 病灶分割能力中等，可考虑增加病灶样本权重。\n")
        else:
            f.write("😰 病灶分割能力较弱，建议重点优化病灶检测策略。\n")

def generate_csv_report(results, output_path):
    """生成CSV详细报告"""
    # 准备数据
    data = []
    for class_name, metrics in results['class_metrics'].items():
        row = {
            '类别': class_name,
            'IoU': f"{metrics['IoU']:.4f}",
            'Precision': f"{metrics['Precision']:.4f}",
            'Recall': f"{metrics['Recall']:.4f}",
            'F1': f"{metrics['F1']:.4f}",
            '出现图像数': metrics['present_in_images'],
            '出现频率': f"{metrics['image_frequency']:.4f}",
            '是否病灶': '是' if class_name in config.LESION_NAMES else '否'
        }
        data.append(row)
    
    # 保存CSV
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

def generate_lesion_csv_report(results, output_path):
    """生成病灶详细CSV报告"""
    lesion_data = []
    
    # 病灶整体指标
    overall_metrics = results['lesion_metrics']['overall_lesion_metrics']
    lesion_data.append({
        '指标类型': '病灶整体',
        '病灶名称': '所有病灶',
        'IoU': f"{overall_metrics['lesion_mIoU_optimized']:.4f}",
        'F1': f"{overall_metrics['lesion_mF1_optimized']:.4f}",
        '检测率': f"{overall_metrics['overall_detection_rate']:.4f}",
        '真实存在次数': '-',
        '成功检测次数': '-',
        '备注': '优化版本(只计算存在的病灶)'
    })
    
    # 各病灶详细指标
    for lesion_name, metrics in results['lesion_metrics']['individual_lesions'].items():
        lesion_data.append({
            '指标类型': '单个病灶',
            '病灶名称': lesion_name,
            'IoU': f"{metrics['IoU']:.4f}",
            'F1': f"{metrics['F1']:.4f}",
            '检测率': f"{metrics['detection_rate']:.4f}",
            '真实存在次数': metrics['ground_truth_count'],
            '成功检测次数': metrics['detected_count'],
            '备注': f"检测成功率: {metrics['detected_count']}/{metrics['ground_truth_count']}"
        })
    
    # 病灶大小敏感性分析
    size_analysis = results['lesion_metrics']['size_sensitivity_analysis']
    for size_name, data in [('小病灶(<1000像素)', 'small'), ('中等病灶(1000-5000像素)', 'medium'), ('大病灶(>5000像素)', 'large')]:
        lesion_data.append({
            '指标类型': '大小敏感性',
            '病灶名称': size_name,
            'IoU': f"{size_analysis[data]['avg_iou']:.4f}",
            'F1': '-',
            '检测率': '-',
            '真实存在次数': size_analysis[data]['count'],
            '成功检测次数': '-',
            '备注': f"标准差: {size_analysis[data]['std_iou']:.4f}"
        })
    
    # 保存CSV
    df = pd.DataFrame(lesion_data)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

def generate_image_csv_report(results, output_path):
    """生成图像级详细CSV报告"""
    image_data = []
    
    # 图像级统计
    image_stats = results['image_level_analysis']
    image_data.append({
        '统计类型': '图像级整体',
        '指标名称': '平均mIoU',
        '数值': f"{image_stats['avg_image_miou']:.4f}",
        '标准差': f"{image_stats['std_image_miou']:.4f}",
        '备注': '所有图像mIoU的平均值'
    })
    
    image_data.append({
        '统计类型': '图像级整体',
        '指标名称': '最佳mIoU',
        '数值': f"{image_stats['max_image_miou']:.4f}",
        '标准差': '-',
        '备注': '单张图像的最高mIoU'
    })
    
    image_data.append({
        '统计类型': '图像级整体',
        '指标名称': '最差mIoU',
        '数值': f"{image_stats['min_image_miou']:.4f}",
        '标准差': '-',
        '备注': '单张图像的最低mIoU'
    })
    
    image_data.append({
        '统计类型': '图像级整体',
        '指标名称': '中位数mIoU',
        '数值': f"{image_stats['median_image_miou']:.4f}",
        '标准差': '-',
        '备注': '所有图像mIoU的中位数'
    })
    
    # 多病灶分析
    multi_lesion = results['multi_lesion_analysis']
    for category_en, category_cn in [('no_lesion_images', '无病灶图像'), ('single_lesion_images', '单病灶图像'), ('multi_lesion_images', '多病灶图像')]:
        data = multi_lesion[category_en]
        image_data.append({
            '统计类型': '多病灶分析',
            '指标名称': category_cn,
            '数值': f"{data['avg_miou']:.4f}",
            '标准差': f"{data['std_miou']:.4f}",
            '备注': f"图像数量: {data['count']}张 ({data['percentage']:.1f}%)"
        })
    
    # 保存CSV
    df = pd.DataFrame(image_data)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

def main():
    """主函数"""
    logger.info("🔥 开始全面测试评估")
    
    # 检查路径
    if not os.path.exists(config.TEST_IMAGES_DIR):
        logger.error(f"测试图像目录不存在: {config.TEST_IMAGES_DIR}")
        return
    
    if not os.path.exists(config.TEST_MASKS_DIR):
        logger.error(f"测试掩码目录不存在: {config.TEST_MASKS_DIR}")
        return
    
    if not os.path.exists(config.MODEL_PATH):
        logger.error(f"模型文件不存在: {config.MODEL_PATH}")
        return
    
    # 创建结果目录
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # 加载数据集
    dataset = TestDataset(config.TEST_IMAGES_DIR, config.TEST_MASKS_DIR)
    
    # 加载模型
    model = load_model(config.MODEL_PATH)
    if model is None:
        return
    
    # 初始化指标计算器
    metrics = ComprehensiveMetrics()
    
    # 运行推理
    run_inference(model, dataset, metrics)
    
    # 计算最终指标
    results = metrics.compute_final_metrics()
    
    # 生成报告
    generate_comprehensive_report(results, config.RESULTS_DIR)
    
    logger.info("🎉 测试评估完成！")
    
    # 打印简要结果
    logger.info("=" * 60)
    logger.info("📊 测试结果简要:")
    logger.info(f"  测试图像: {results['basic_metrics']['total_images']}张")
    logger.info(f"  优化mIoU: {results['overall_metrics']['optimized_mIoU']:.4f} ({results['overall_metrics']['optimized_mIoU']*100:.1f}%)")
    logger.info(f"  病灶mIoU: {results['lesion_metrics']['overall_lesion_metrics']['lesion_mIoU_optimized']:.4f} ({results['lesion_metrics']['overall_lesion_metrics']['lesion_mIoU_optimized']*100:.1f}%)")
    logger.info(f"  病灶检测率: {results['lesion_metrics']['overall_lesion_metrics']['overall_detection_rate']:.4f} ({results['lesion_metrics']['overall_lesion_metrics']['overall_detection_rate']*100:.1f}%)")
    logger.info(f"  详细报告: {config.RESULTS_DIR}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main() 