 # 🔥 SAM声带病灶分割 - 老哥特制版 🔥

嗨！各位老哥！这是一个基于Meta SAM（Segment Anything Model）的声带病灶分割项目，老哥我花了不少时间优化，专门用来搞定声带的各种病灶！
使用时请确保数据集路径正确。

## 📢 项目简介

这个项目是老哥我基于Facebook的SAM模型搞的一个声带病灶自动分割工具。能够自动识别和分割声带的6种不同类别：

- 🎯 **背景** (Background)
- 🔴 **左声带** (Left Vocal Cord) 
- 🟢 **右声带** (Right Vocal Cord)
- 🟡 **声带小结** (Vocal Nodules)
- 🟣 **声带白斑** (Vocal Leukoplakia) 
- 🔵 **声带乳头状瘤** (Vocal Papilloma)

老哥我针对医学图像分割的特点做了大量优化，特别是解决了类别不平衡的问题！

## 🚀 项目特色 (老哥的杀手锏)

### 💪 硬核优化策略
- **🔥 Focal Loss**: 专治难分类的病灶
- **🎯 Dice Loss**: 小目标分割神器
- **⚡ 智能采样**: 病灶样本优先级训练
- **🧠 注意力机制**: 让模型主动关注病灶区域
- **🌊 动态权重调整**: 根据训练效果自动调整类别权重
- **🔍 边缘感知损失**: 提升边界分割精度

### 🎨 多版本训练脚本
老哥准备了多个版本，总有一款适合你：

1. **`train_simple.py`** - 新手友好版，简单易懂
2. **`train_optimized.py`** - 老哥优化版，各种黑科技
3. **`train_sobel_optimized.py`** - 边缘增强版，加入Sobel算子

### 🔬 完整测试套件
- **`test_optimized.py`** - 单图测试，可视化结果
- **`test_lesion.py`** - 专门测试病灶识别效果
- **`test_fold_evaluation.py`** - 科学评估，交叉验证

## 📁 项目结构

```
Segment anything model/
├── 📝 README.md                    # 你正在看的文件
├── 🛠️ install_requirements.py      # 一键安装依赖
├── 📋 requirements.txt             # 依赖列表
├── 🏋️ train_*.py                  # 各种训练脚本
├── 🔍 test_*.py                   # 测试脚本
├── 🤖 pre_models/                 # 预训练模型
│   ├── sam_vit_b_01ec64.pth       # SAM ViT-B模型
│   ├── sam_vit_h_4b8939.pth       # SAM ViT-H模型
│   ├── sam2.1_hiera_tiny.pt       # SAM2.1模型
│   └── medsam_vit_b.pth          # 医学专用SAM
└── 📊 results/                    # 训练结果和预测
    ├── predictions/               # 预测结果
    └── run_*/                     # 训练记录
```

## 🔧 环境安装

### 方法一：老哥一键安装（推荐）
```bash
python install_requirements.py
```

### 方法二：手动安装
```bash
# 基础环境
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python pillow numpy scikit-learn matplotlib tqdm

# SAM模型
pip install segment-anything

# 其他依赖
pip install -r requirements.txt
```

## 🎯 快速开始

### 1. 训练模型
```bash
# 新手推荐
python train_simple.py

# 老哥优化版（推荐）
python train_optimized.py

# 边缘增强版
python train_sobel_optimized.py
```

### 2. 测试模型
```bash
# 单图测试
python test_optimized.py

# 病灶专项测试
python test_lesion.py

# 完整评估
python test_fold_evaluation.py
```


## 🔥 核心技术亮点

### 1. 智能采样策略
```python
# 病灶样本过采样5倍！
LESION_OVERSAMPLE_factor = 5.0
# 背景样本欠采样，防止数据不平衡
BACKGROUND_UNDERSAMPLE_FACTOR = 0.3
```

### 2. 动态损失权重
```python
# 病灶类别权重爆炸式提升
INITIAL_CLASS_WEIGHTS = [0.1, 1.0, 1.0, 30.0, 35.0, 32.0]
```

### 3. 多重损失融合
- CrossEntropy Loss (基础分类)
- Focal Loss (难样本挖掘) 
- Dice Loss (小目标优化)
- Edge Loss (边界增强)

### 4. 注意力增强
```python
# 让模型主动关注病灶区域
self.attention = nn.Sequential(
    nn.Conv2d(256, 64, kernel_size=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 1, kernel_size=1),
    nn.Sigmoid()
)
```

## 📝 使用说明

### 数据准备
1. 图像文件放在 `data/train/images/` 和 `data/val/images/`
2. 标注文件放在 `data/train/masks/` 和 `data/val/masks/`
3. 确保图像和掩码文件名对应

### 类别ID映射
```python
ID_MAPPING = {
    0: 0,    # 背景
    170: 1,  # 左声带
    184: 2,  # 右声带
    105: 3,  # 声带小结
    23: 4,   # 声带白斑
    146: 5,  # 声带乳头状瘤
}
```

### 训练配置调整
老哥已经调好了大部分参数，但你可以根据需要修改：

- `BATCH_SIZE`: 显存不够就调小
- `LEARNING_RATE`: 学习率，默认2e-4
- `NUM_EPOCHS`: 训练轮数，默认150
- `CLASS_WEIGHTS`: 类别权重，病灶类别权重很高

## 🎨 可视化结果

训练完成后，老哥的脚本会自动生成：
- 🎯 分割结果可视化
- 📊 训练曲线图
- 📈 性能指标报告
- 🔥 病灶检测专项报告


*记得给个 ⭐ 支持一下！*

**愿天下无难分割的病灶！🔥**