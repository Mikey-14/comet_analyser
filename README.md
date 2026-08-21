# Comet Analyser — 彗星实验分析工具

一个用于**彗星实验（Comet Assay）**图像定量分析的桌面应用。
交互式框选彗星细胞，自动分割头部与尾部，计算 DNA 损伤关键指标，支持可视化标注、交互式编辑与批量导出。

基于 **PyQt5 + OpenCV + NumPy** 构建。

---

## 功能特性

| 功能 | 描述 |
|------|------|
| 图像加载 | 支持单张 / 多张 / 整个文件夹的常见格式（PNG / JPG / BMP / TIFF） |
| 背景校正 | 框选背景区域，一组图片共享同一背景值 |
| 细胞分析 | 逐个框选彗星细胞，自动分割头部与尾部 |
| 自定义阈值 | 头部 / 尾部强度阈值独立设置（默认 0.5 / 0.05） |
| 分割模式 | 模式1：方向限制；模式2：尾部宽度限制（≤ 头部直径） |
| 指标计算 | Tail Length、Tail DNA%、Tail Moment、Olive Tail Moment、面积、荧光强度等 |
| 可视化标注 | 头部（蓝色圆）、尾部（黄色填充）叠加显示 |
| 交互式编辑 | 点击选中细胞，Delete/Backspace 或右键菜单删除 |
| Excel 导出 | 分析结果导出为 .xlsx（逐细胞明细 + 图片汇总两个工作表） |
| 高 DPI 适配 | 自动适配 1080p / 2K / 4K 显示器，字体与控件大小合适 |

---

## 彗星实验简介

彗星实验（单细胞凝胶电泳）是检测细胞 DNA 损伤的经典方法。损伤 DNA 在电泳中迁移形成"彗星"状——头部为未损伤 DNA，尾部为断裂碎片。尾部 DNA 比例越高，损伤越严重。

### 关键指标

| 指标 | 英文 | 含义 |
|------|------|------|
| 尾长 | Tail Length | 尾部 DNA 迁移的最大距离 |
| 尾部 DNA 百分比 | Tail DNA% | 尾部荧光强度 / 总荧光强度 × 100% |
| 尾矩 | Tail Moment | TM = Tail DNA% × Tail Length |
| Olive 尾矩 | Olive Tail Moment | OTM = (尾部质心 - 头部质心) × Tail DNA% |

---

## 项目结构

```
comet_analyser/                  # 应用包
├── main.py                      # 应用入口（高 DPI 适配、全局字体）
├── analysis/                    # 分析算法核心
│   ├── preprocess.py            #   图像预处理（灰度、去噪、CLAHE）
│   ├── segmentation.py          #   彗星分割（阈值分割、头尾分离、方向/宽度限制）
│   └── metrics.py               #   指标计算（Tail DNA%、TM、OTM 等）
├── ui/                          # 用户界面
│   ├── main_window.py           #   主窗口（导航栏、模式切换、结果面板）
│   └── image_canvas.py          #   交互式画布（框选 ROI、细胞选中/删除）
└── resources/                   # 静态资源
```

### 分析管线

```
原始图像
  │
  ▼
背景减除 → 去噪 → CLAHE 增强          ← preprocess
  │
  ▼
彗星分割（阈值 → 连通域锚定 → 头尾分离） ← segmentation
  │  ├─ 模式1：尾部方向限制
  │  └─ 模式2：尾部宽度限制（≤ 头部直径）
  ▼
指标计算（面积 → 荧光强度 → 质心 →      ← metrics
          Tail DNA% → TM → OTM）
  │
  ▼
结果展示 + 可视化标注 + Excel 导出
```

---

## 快速开始

### 环境要求

- Python 3.8+
- PyQt5
- OpenCV
- NumPy
- openpyxl

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
pip install opencv-python numpy PyQt5 openpyxl
```

### 运行

```bash
# 在 comet_analyse/ 目录下执行：
python comet_analyser/main.py
```

---

## 使用指南

### 1. 加载图像

- 左侧面板「加载图片...」：单张或批量选择图片
- **文件 → 打开图片** (`Ctrl+O`)：加载单张图片
- **文件 → 打开文件夹** (`Ctrl+Shift+O`)：批量加载整个文件夹

### 2. 分析流程

1. **框选背景区域**：在图片上拖拽框选一片没有细胞的背景区域（一组图片共享该背景值）
2. **框选细胞**：逐个拖拽框选彗星细胞，自动完成分割与指标计算
3. **完成当前图片**：保存当前图片结果并切换到下一张
4. 顶部导航栏「上一张 / 下一张」切换图片

### 3. 阈值与分割模式

- **使用自定义阈值**（勾选后生效）：
  - 头部阈值（默认 0.5，步长 0.1）
  - 尾部阈值（默认 0.05，步长 0.01）
- **分割模式**（顶部导航栏右侧下拉框）：
  - 模式1：仅按拖尾方向限制尾部
  - 模式2：额外限制尾部任意位置宽度不超过头部直径

### 4. 编辑细胞结果

| 操作 | 效果 |
|------|------|
| 左键单击细胞 | 选中（红色虚线框高亮，表格同步） |
| 左键单击空白 | 取消选中 |
| Delete / Backspace | 删除选中细胞 |
| 右键单击细胞 | 弹出菜单，选择「删除此细胞」 |

### 5. 导出结果

- **文件 → 导出 Excel** (`Ctrl+E`)：将结果导出为 `.xlsx`，包含两个工作表：
  - `细胞分析结果`：每个细胞的完整指标明细
  - `图片汇总`：每张图片的细胞数量与平均值

---

## 打包为 exe

使用 PyInstaller 打包为单文件可执行程序：

```bash
python -m PyInstaller --noconfirm --clean --onefile --windowed \
  --name CometAnalyser \
  --paths comet_analyser \
  --hidden-import openpyxl \
  comet_analyser/main.py
```

生成的 `CometAnalyser.exe` 约 91 MB（单文件、无控制台窗口）。

---

## 可调参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 头部阈值 | 0.5 | 头部亮度阈值（0–1），越高头部识别越严格 |
| 尾部阈值 | 0.05 | 尾部亮度阈值（0–1），越低彗星识别越宽泛 |
| 分割模式 | 模式1 | 模式1：方向限制；模式2：宽度限制 |
| 像素尺寸 | 1.0 µm/px | 设置 → 像素尺寸，用于换算尾长微米值 |

---

## 许可证

MIT License

---

## 参考文献

- Olive, P. L., & Banáth, J. P. (2006). The comet assay: a method to measure DNA damage in individual cells. *Nature Protocols*, 1(1), 23-29.
- Collins, A. R. (2004). The comet assay for DNA damage and repair. *Molecular Biotechnology*, 26, 249-261.
