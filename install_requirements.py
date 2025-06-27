#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoDL环境依赖安装脚本
为SAM声带病灶分割项目安装所有必需的依赖
"""

import os
import sys
import subprocess
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description=""):
    """运行命令并处理错误"""
    logger.info(f"执行: {description if description else command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"执行失败: {e}")
        if e.stderr:
            logger.error(f"错误信息: {e.stderr}")
        return False

def install_basic_dependencies():
    """安装基础依赖"""
    logger.info("=== 安装基础Python包 ===")
    
    basic_packages = [
        "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
        "opencv-python",
        "pillow",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "tqdm",
        "pyyaml",
    ]
    
    for package in basic_packages:
        success = run_command(f"pip install {package}", f"安装 {package.split()[0]}")
        if not success:
            logger.warning(f"安装 {package} 失败，继续...")

def install_sam():
    """安装Segment Anything"""
    logger.info("=== 安装 Segment Anything Model ===")
    
    # 尝试多种安装方式
    install_methods = [
        "pip install segment-anything",
        "pip install git+https://github.com/facebookresearch/segment-anything.git",
        "pip install 'git+https://github.com/facebookresearch/segment-anything.git@main'"
    ]
    
    for method in install_methods:
        logger.info(f"尝试方法: {method}")
        if run_command(method, "安装SAM"):
            logger.info("SAM安装成功!")
            return True
        else:
            logger.warning(f"方法失败，尝试下一个...")
    
    logger.error("所有SAM安装方法都失败了!")
    return False

def verify_installation():
    """验证安装"""
    logger.info("=== 验证安装 ===")
    
    # 测试基础包
    test_imports = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-learn"),
        ("matplotlib", "Matplotlib"),
        ("tqdm", "TQDM"),
        ("yaml", "PyYAML"),
    ]
    
    failed_imports = []
    
    for module, name in test_imports:
        try:
            __import__(module)
            logger.info(f"✓ {name} 导入成功")
        except ImportError:
            logger.error(f"✗ {name} 导入失败")
            failed_imports.append(name)
    
    # 测试SAM
    try:
        from segment_anything import sam_model_registry, SamPredictor
        logger.info("✓ Segment Anything 导入成功")
    except ImportError:
        logger.error("✗ Segment Anything 导入失败")
        failed_imports.append("Segment Anything")
    
    # 测试CUDA
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"✓ CUDA 可用: {torch.cuda.get_device_name(0)}")
            logger.info(f"✓ GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            logger.warning("✗ CUDA 不可用")
    except Exception as e:
        logger.error(f"✗ CUDA 检查失败: {e}")
    
    if failed_imports:
        logger.error(f"安装验证失败，以下包导入失败: {', '.join(failed_imports)}")
        return False
    else:
        logger.info("✓ 所有依赖安装验证成功!")
        return True

def download_sam_models():
    """下载SAM预训练模型（可选）"""
    logger.info("=== 检查SAM模型文件 ===")
    
    sam_model_path = "/root/autodl-tmp/SAM/models/sam_vit_h_4b8939.pth"
    
    if os.path.exists(sam_model_path):
        logger.info(f"✓ SAM模型已存在: {sam_model_path}")
        return True
    
    logger.info("SAM模型不存在，需要手动下载")
    logger.info("请确保以下模型文件存在:")
    logger.info(f"  {sam_model_path}")
    logger.info("下载链接: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
    
    return False

def create_directories():
    """创建必要的目录"""
    logger.info("=== 创建项目目录 ===")
    
    directories = [
        "/root/autodl-tmp/SAM/results",
        "/root/autodl-tmp/SAM/results/models",
        "/root/autodl-tmp/SAM/results/logs",
        "/root/autodl-tmp/SAM/results/checkpoints",
        "/root/autodl-tmp/SAM/results/predictions"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"✓ 创建目录: {directory}")

def main():
    """主函数"""
    logger.info("=== SAM训练环境安装脚本 - AutoDL版本 ===")
    
    # 检查Python版本
    python_version = sys.version_info
    logger.info(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        logger.error("需要Python 3.8或更高版本!")
        sys.exit(1)
    
    # 更新pip
    logger.info("=== 更新pip ===")
    run_command("pip install --upgrade pip", "更新pip")
    
    # 安装基础依赖
    install_basic_dependencies()
    
    # 安装SAM
    sam_success = install_sam()
    
    # 创建目录
    create_directories()
    
    # 验证安装
    verification_success = verify_installation()
    
    # 检查模型文件
    model_exists = download_sam_models()
    
    # 总结
    logger.info("=== 安装总结 ===")
    if verification_success:
        logger.info("✓ 环境安装成功!")
        logger.info("可以运行训练脚本:")
        logger.info("  python train_sam_autodl.py")
    else:
        logger.error("✗ 环境安装存在问题，请检查上述错误信息")
    
    if not model_exists:
        logger.warning("⚠ SAM模型文件缺失，请手动下载")
    
    logger.info("安装脚本执行完成!")

if __name__ == "__main__":
    main() 