#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM通用病灶测试脚本 - 单张图像可视化
功能：加载训练好的模型，对单张图像进行预测，并可视化结果：
1. 原图像
2. 预测mask overlay到原图像（半透明红色表示指定病灶）
3. 病灶边缘勾画（绿色轮廓）
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry

# ===== 💪 配置类（与训练脚本一致） =====
class TestConfig:
    IMAGE_SIZE = 1024
    NUM_CLASSES = 2
    SAM_MODEL_TYPE = "vit_b"
    PIXEL_MEAN = [123.675, 116.28, 103.53]
    PIXEL_STD = [58.395, 57.12, 57.375]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 用户修改区：指定病灶ID和名称
    LESION_ID = 29  
    LESION_NAME = "声带白斑" 
    
    ID_MAPPING = {
        0: 0,          # 背景
        LESION_ID: 1,  # 指定病灶
    }

config = TestConfig()

# ===== 🚀 模型类（与训练脚本一致，简化版） =====
class EnhancedSAMModel(torch.nn.Module):
    """DSC增强SAM模型 - 基于原有架构优化"""
    
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
        
        self.freeze_sam_components()
        
        print("DSC增强SAM模型改装完毕！")
    
    def freeze_sam_components(self):
        # 稍微减少冻结层数，保留更多可训练参数
        layers = list(self.sam.image_encoder.children())
        for i, layer in enumerate(layers[:-4]):  # 减少冻结层
            for param in layer.parameters():
                param.requires_grad = False
        
        print("SAM参数部分冻结完毕！")
    
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

# ===== 🎯 加载模型 =====
def load_model(model_path, sam_checkpoint):
    sam = sam_model_registry[config.SAM_MODEL_TYPE](checkpoint=sam_checkpoint)
    model = EnhancedSAMModel(sam, config.NUM_CLASSES)
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.DEVICE)
    model.eval()
    print(f"模型加载成功: {model_path}")
    return model

# ===== 🛠️ 图像预处理（与训练一致） =====
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]
    
    # 调整大小
    image_resized = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    
    # 转换为tensor
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
    
    # 标准化
    mean = torch.tensor(config.PIXEL_MEAN).view(3, 1, 1) / 255.0
    std = torch.tensor(config.PIXEL_STD).view(3, 1, 1) / 255.0
    image_tensor = (image_tensor - mean) / std
    
    # 添加batch维度
    image_tensor = image_tensor.unsqueeze(0).to(config.DEVICE)
    
    return image_tensor, image, original_size

# ===== 🔍 预测并获取mask =====
def predict_mask(model, image_tensor, original_size):
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_mask = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()
    
    # 调整回原大小
    pred_mask = cv2.resize(pred_mask.astype(np.uint8), (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
    
    return pred_mask

# ===== 📊 可视化结果 =====
def visualize_results(original_image, pred_mask):
    # 1. 原图像
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_image)
    axes[0].set_title("1. original")
    axes[0].axis('off')
    
    # 2. Overlay mask（红色半透明）
    overlay = original_image.copy()
    lesion_mask = (pred_mask == 1)  # 病灶类为1
    overlay[lesion_mask] = (overlay[lesion_mask] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
    axes[1].imshow(overlay)
    axes[1].set_title(f"2. Mask Overlay")
    axes[1].axis('off')
    
    # 3. 病灶边缘勾画（绿色轮廓）
    edges = cv2.Canny(pred_mask * 255, 100, 200)  # 使用Canny检测边缘
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_colored[edges > 0] = [0, 255, 0]  # 绿色
    contour_image = cv2.addWeighted(original_image, 1.0, edges_colored, 0.8, 0)
    axes[2].imshow(contour_image)
    axes[2].set_title(f"3. Lesion Edge")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 保存结果（可选）
    save_path = "autodl-tmp/SAM/results/models/run_rtzl/visualization_result.png"
    plt.savefig(save_path)
    print(f"可视化结果保存为: {save_path}")

def main():
    image_path = "/root/autodl-tmp/SAM/12classes_lesion/test/images/声带白斑中重_陈建辉D50217B85_20210824_240300572.jpg"  
    mask_path = "/root/autodl-tmp/SAM/12classes_lesion/test/masks/声带白斑中重_陈建辉D50217B85_20210824_240300572.png"  
    model_path = "/root/autodl-tmp/SAM/results/models/dsc_enhanced_sdbb/models/best_model_lesion_dice.pth"  
    sam_checkpoint = "/root/autodl-tmp/SAM/pre_models/sam_vit_b_01ec64.pth"  
    
    # 新增：检查mask是否包含指定LESION_ID
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None or config.LESION_ID not in np.unique(mask):
        print(f"警告：图像的mask不包含ID={config.LESION_ID} ({config.LESION_NAME})！预测结果可能无效。")
    else:
        print(f"确认：图像的mask包含ID={config.LESION_ID} ({config.LESION_NAME})。")
    
    model = load_model(model_path, sam_checkpoint)
    
    image_tensor, original_image, original_size = preprocess_image(image_path)
    
    pred_mask = predict_mask(model, image_tensor, original_size)
    
    visualize_results(original_image, pred_mask)

if __name__ == "__main__":
    main()