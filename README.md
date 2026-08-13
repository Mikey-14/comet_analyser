# 🔬 Comet Analyser — 彗星实验分析工具

一个用于**彗星实验（Comet Assay）**图像定量分析的桌面应用。  
自动分割彗星的头部与尾部，计算 DNA 损伤关键指标，支持可视化标注与批量导出。

基于 **PyQt5 + OpenCV + scikit-image** 构建。

---

## 📸 功能特性

| 功能 | 描述 |
|------|------|
| 🖼️ **图像加载** | 支持单张或批量加载常见格式（PNG / JPG / BMP / TIFF） |
| 🧹 **预处理管线** | 灰度化 → 高斯/中值/双边去噪 → CLAHE 对比度增强 → 背景校正 |
| 🎯 **自动分割** | Otsu / 自适应阈值分割 → 最大连通域筛选 → 亮度峰值头部定位 → 头尾分离 |
| 📊 **指标计算** | Tail Length、Tail DNA%、Tail Moment、Olive Tail Moment、荧光强度、面积等 |
| 🎨 **可视化叠加** | 彩色标注：头部（蓝）、尾部（红）、彗星轮廓（绿）、头部中心十字、朝向线 |
| 📋 **批量分析** | 一键分析全部加载图片，结果汇总 |
| 📁 **CSV 导出** | 将分析结果导出为 UTF-8 CSV |
| 🖱️ **交互式框选** | 支持鼠标拖拽框选 ROI 区域 |

---

## 🧪 彗星实验简介

彗星实验（单细胞凝胶电泳）是检测细胞 DNA 损伤的经典方法。损伤 DNA 在电泳中迁移形成"彗星"状——头部为未损伤 DNA，尾部为断裂碎片。尾部 DNA 比例越高，损伤越严重。

### 关键指标

| 指标 | 英文 | 含义 |
|------|------|------|
| **尾长** | Tail Length | 尾部 DNA 迁移的最大距离 |
| **尾部 DNA 百分比** | Tail DNA% | 尾部荧光强度 / 总荧光强度 × 100% |
| **尾矩** | Tail Moment | TM = Tail DNA% × Tail Length |
| **Olive 尾矩** | Olive Tail Moment | OTM = (尾部质心 - 头部质心) × Tail DNA% |

---

## 📁 项目结构

```
comet_analyser/
├── main.py                    # 应用入口
├── analysis/                  # 分析算法核心
│   ├── preprocess.py          #   图像预处理管线
│   ├── segmentation.py        #   彗星分割（Otsu、头部定位、头尾分离）
│   └── metrics.py             #   指标计算（Tail DNA%, TM, OTM 等）
├── ui/                        # 用户界面
│   ├── main_window.py         #   主窗口（菜单、工具栏、文件列表、结果面板）
│   └── image_canvas.py        #   交互式图像画布（支持鼠标框选 ROI）
└── resources/                 # 静态资源（图标等）
```

### 分析管线

```
原始图像
  │
  ▼
┌─────────────────────────────┐
│  preprocess_pipeline()      │  ← analysis/preprocess.py
│  灰度化 → 去噪 → CLAHE →    │
│  背景校正                    │
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  segment_comet()            │  ← analysis/segmentation.py
│  Otsu阈值 → 最大连通域 →    │
│  峰值定位头部 → 头尾分离    │
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  compute_all_metrics()      │  ← analysis/metrics.py
│  面积 → 荧光强度 → 质心 →   │
│  Tail DNA% → TM → OTM       │
└─────────────┬───────────────┘
              ▼
        结果展示 + 可视化叠加
```

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- OpenCV
- NumPy
- PyQt5

### 安装

```bash
# 克隆仓库
git clone <your-repo-url>
cd comet_analyser

# 创建虚拟环境（推荐）
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 安装依赖
pip install opencv-python numpy PyQt5 scikit-image openpyxl
```

### 运行

```bash
# 在项目根目录 (comet-analyse/) 下执行：
python -m comet_analyser.main
# 或者
python comet_analyser/main.py
```

---

## 📖 使用指南

### 1. 加载图像

- **文件 → 打开图片** (`Ctrl+O`)：加载单张图片
- **文件 → 打开文件夹** (`Ctrl+Shift+O`)：批量加载整个文件夹

### 2. 分析图像

- **分析 → 分析当前图片** (`Ctrl+R`)：分析当前显示的彗星图像
- **分析 → 批量分析全部** (`Ctrl+B`)：分析列表中的所有图片

### 3. 查看结果

- 右侧面板显示完整指标表格（尾长、面积、荧光强度、Tail DNA%、TM、OTM 等）
- 图像下方显示关键指标摘要
- 标注叠加层：**绿色** = 彗星轮廓，**蓝色** = 头部，**红色** = 尾部，**黄色** = 头部中心 & 朝向

### 4. 切换显示

- **视图 → 切换标注叠加** (`Ctrl+L`)：在原始图像和标注图像之间切换

### 5. 导出结果

- **文件 → 导出结果为 CSV** (`Ctrl+E`)：将分析结果保存为 CSV 文件

---

## ⚙️ 可调参数

预处理和分割模块的参数均可在调用时调整，常见可调参数：

| 模块 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| 去噪 | `denoise_method` | `"gaussian"` | `gaussian` / `median` / `bilateral` |
| 去噪 | `denoise_kernel` | `5` | 滤波核大小 |
| CLAHE | `clahe_clip` | `2.0` | 对比度裁剪阈值 |
| 背景校正 | `bg_radius` | `50` | 背景估计半径 |
| 阈值 | `threshold_method` | `"otsu"` | `otsu` / `adaptive` |
| 头尾分离 | `head_radius_ratio` | `0.25` | 头部半径占比 |

---

## 📄 许可证

MIT License

---

## 👨‍🔬 参考文献

- Olive, P. L., & Banáth, J. P. (2006). The comet assay: a method to measure DNA damage in individual cells. *Nature Protocols*, 1(1), 23-29.
- Collins, A. R. (2004). The comet assay for DNA damage and repair. *Molecular Biotechnology*, 26, 249-261.