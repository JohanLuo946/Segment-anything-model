#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import json
import torch
import gradio as gr
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from test_lesion_text_report import test_single_image_with_text_report

# 配置日志
def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"gradio_app_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def load_error_image():
    """加载错误提示图像"""
    error_image_path = Path("data/error_image.jpg")
    if not error_image_path.exists():
        # 创建一个带有错误信息的图像
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        cv2.putText(img, "图像处理失败", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        cv2.imwrite(str(error_image_path), img)
    return cv2.cvtColor(cv2.imread(str(error_image_path)), cv2.COLOR_BGR2RGB)

def is_laryngoscope_image(image):
    """
    验证是否为喉镜图像
    返回: (bool, str) - (是否为喉镜图像, 错误信息)
    """
    try:
        # 1. 检查图像尺寸（喉镜图像通常有特定的尺寸范围）
        height, width = image.shape[:2]
        if width < 400 or height < 400:
            return False, "图像分辨率过低，不符合喉镜图像要求"
        
        # 2. 检查图像亮度分布（喉镜图像通常中心区域较亮）
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        center_y, center_x = height // 2, width // 2
        center_region = gray[center_y-50:center_y+50, center_x-50:center_x+50]
        if center_region.mean() < 50:  # 中心区域过暗
            return False, "图像中心区域过暗，不符合喉镜图像特征"
            
        # 3. 检查图像对比度（喉镜图像通常具有适中的对比度）
        contrast = np.std(gray)
        if contrast < 20 or contrast > 100:
            return False, "图像对比度异常，不符合喉镜图像特征"
            
        # 4. 检查色彩分布（喉镜图像通常偏红色调）
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        red_range = ((hsv[:,:,0] >= 170) | (hsv[:,:,0] <= 10)) & (hsv[:,:,1] >= 50)
        red_ratio = np.sum(red_range) / (height * width)
        if red_ratio < 0.1:  # 红色区域比例过低
            return False, "图像色彩分布不符合喉镜图像特征"
            
        # 5. 检查边缘特征（喉镜图像通常有明显的圆形边界）
        edges = cv2.Canny(gray, 100, 200)
        circles = cv2.HoughCircles(
            gray, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=100,
            param1=50,
            param2=30,
            minRadius=int(min(width, height) * 0.2),
            maxRadius=int(min(width, height) * 0.5)
        )
        if circles is None:
            return False, "未检测到喉镜图像特征的圆形边界"

        return True, "验证通过"
        
    except Exception as e:
        logger.error(f"喉镜图像验证过程出错: {str(e)}", exc_info=True)
        return False, f"图像验证过程出错: {str(e)}"

def validate_image(image):
    """验证图像是否有效且为喉镜图像"""
    if image is None:
        return False, "未上传图像"
    
    if not isinstance(image, np.ndarray):
        return False, "图像格式不正确"
        
    if image.size == 0:
        return False, "图像为空"
        
    if len(image.shape) != 3:
        return False, "图像维度不正确"
    
    # 验证是否为喉镜图像
    is_valid_scope, message = is_laryngoscope_image(image)
    if not is_valid_scope:
        return False, f"非喉镜图像: {message}"
        
    return True, "图像有效"

def process_image(image, model_path="autodl-tmp/SAM/results/models/run_6_finetune/models/checkpoint_epoch_74_best.pth"):
    """处理上传的图像并返回分析结果"""
    try:
        # 记录处理开始
        logger.info("开始处理新的图像")
        
        # 验证图像
        is_valid, message = validate_image(image)
        if not is_valid:
            logger.error(f"图像验证失败: {message}")
            return load_error_image(), f"错误: {message}"
        
        # 保存上传的图像
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / f"temp_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(str(temp_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        logger.info(f"图像已保存至: {temp_path}")
        
        # 检查模型文件
        if not Path(model_path).exists():
            logger.error(f"模型文件不存在: {model_path}")
            return load_error_image(), "错误: 模型文件不存在"
        
        # 设置输出目录
        output_dir = Path("gradio_outputs")
        output_dir.mkdir(exist_ok=True)
        
        # 运行分析
        logger.info("开始运行图像分析")
        mask, probs, report = test_single_image_with_text_report(
            model_path=model_path,
            image_path=str(temp_path),
            save_dir=str(output_dir)
        )
        
        # 读取生成的可视化结果
        result_image_path = output_dir / f"{temp_path.stem}_text_report_visualization.png"
        if not result_image_path.exists():
            logger.error("未生成结果图像")
            return load_error_image(), "错误: 未能生成分析结果图像"
            
        result_image = cv2.imread(str(result_image_path))
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        
        # 生成文本报告
        text_report = "🔍 病灶分析报告\n\n"
        
        # 基本信息
        text_report += f"图像尺寸: {report['检查基本信息']['图像尺寸']}\n"
        text_report += f"病灶总数: {report['检查基本信息']['病灶总数']}\n\n"
        
        # 病灶详情
        if report['病灶详细分析']:
            for lesion in report['病灶详细分析']:
                text_report += f"病灶 {lesion['病灶编号']}:\n"
                text_report += f"  类型: {lesion['病灶类型']['中文名称']}\n"
                text_report += f"  位置: {lesion['位置信息']['相对位置']}\n"
                text_report += f"  大小: {lesion['尺寸特征']['尺寸分级']} ({lesion['尺寸特征']['估算面积']})\n"
                text_report += f"  形态: {lesion['形态特征']['形态描述']}\n"
                text_report += f"  风险等级: {lesion['病灶类型']['风险等级']}\n"
                text_report += f"  置信度: {lesion['置信度']['检测可靠性']}\n\n"
        else:
            text_report += "未检测到病灶\n\n"
        
        # 统计信息
        text_report += "📊 统计信息:\n"
        text_report += f"总病灶面积: {report['定量统计']['总病灶面积']}\n"
        text_report += f"平均病灶大小: {report['定量统计']['平均病灶大小']}\n\n"
        
        # 临床建议
        text_report += "💡 临床建议:\n"
        if report['临床建议']:
            for advice in report['临床建议']:
                text_report += f"[{advice['优先级']}] {advice['建议']}\n"
                text_report += f"理由: {advice['理由']}\n"
        else:
            text_report += "无特殊临床建议\n"
        
        logger.info("图像处理完成")
        return result_image, text_report
        
    except Exception as e:
        logger.error(f"处理过程中出现错误: {str(e)}", exc_info=True)
        return load_error_image(), f"处理过程中出现错误:\n{str(e)}\n\n请检查图像格式是否正确，或联系管理员查看日志。"

# 创建 Gradio 界面
def create_interface():
    with gr.Blocks(title="声带病灶分析系统", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🔬 声带病灶智能分析系统")
        gr.Markdown("基于 SAM 模型的声带病灶分割与分析")
        
        with gr.Row():
            with gr.Column():
                # 输入部分
                input_image = gr.Image(label="上传图像", type="numpy")
                analyze_btn = gr.Button("开始分析", variant="primary")
                
                # 添加使用说明
                gr.Markdown("""
                ### 📝 使用说明
                1. 仅支持分析喉镜检查图像
                2. 图像必须清晰可见，无严重模糊或失焦
                3. 图像分辨率建议不低于 800x600
                4. 确保图像中心区域光照充足
                
                ### ⚠️ 注意事项
                - 只接受标准喉镜检查图像
                - 非喉镜图像将被系统自动拒绝
                - 处理时间可能需要几秒钟
                """)
            
            with gr.Column():
                # 输出部分
                output_image = gr.Image(label="分析结果")
                output_text = gr.Textbox(label="分析报告", lines=15)
        
        # 设置点击事件
        analyze_btn.click(
            fn=process_image,
            inputs=[input_image],
            outputs=[output_image, output_text]
        )
        
        # 添加示例
        gr.Examples(
            examples=[
                ["autodl-tmp/SAM/12classes_lesion/test/images/声带肉芽肿_贺群贤6011369878_20200803_030303381.jpg"],
                ["autodl-tmp/SAM/12classes_lesion/test/images/声带白斑中重_黄其观133004001726049_20210630_300216191.jpg"],
                ["autodl-tmp/SAM/12classes_lesion/test/images/淀粉样变_汪金书133004001924655_20160323_230957451.jpg"],
                ["autodl-tmp/SAM/12classes_lesion/test/images/声带囊肿 sdnz_声带囊肿_刘振龙L0909654X_20160223_230845130.jpg"],
                ["autodl-tmp/SAM/12classes_lesion/test/images/声带乳头状瘤_王天佑30094525_20211027_270214065.jpg"],
                ["autodl-tmp/SAM/12classes_lesion/test/images/任克氏水肿_顾小明993947_20220128_280156143.jpg"],
            ],
            inputs=[input_image],
        )
        
    return interface

if __name__ == "__main__":
    try:
        logger.info("启动声带病灶分析系统")
        # 创建并启动界面
        interface = create_interface()
        interface.launch(
            server_name="0.0.0.0",    # 允许外部访问
            server_port=6006,         # 使用 AutoDL 支持的端口
            share=False,              # 不需要创建额外的公共链接
            auth=None,                # 不设置访问密码
            inbrowser=True            # 自动打开浏览器
        )
    except Exception as e:
        logger.error(f"系统启动失败: {str(e)}", exc_info=True) 