# Project Structure

## 📁 目录结构

```
TD3-master/
├── design/                          # 设计文档
│   ├── README.md                    # 文档索引
│   ├── model_design_overview.md    # 模型架构设计
│   └── OPTIMIZATION_SUMMARY.md     # 优化总结
│
├── tools/                           # 分析和工具脚本
│   ├── README.md                    # 工具说明
│   ├── analyze_*.py                 # 分析脚本
│   ├── verify_*.py                  # 验证脚本
│   └── visualize_*.py               # 可视化脚本
│
├── legacy/                          # 旧版代码（已弃用）
│   ├── README.md
│   ├── DDPG.py
│   ├── OurDDPG.py
│   └── main.py
│
├── data_preprocessing/              # 数据预处理脚本
│   └── ...
│
├── embeddings/                      # 嵌入向量数据
│   ├── user_token_map.json
│   ├── recommendations.pkl
│   └── ...
│
├── processed_data/                  # 预处理后的数据
│   ├── user_average_beliefs.json
│   └── ...
│
├── results/                         # 训练结果
│   ├── current_user_beliefs.json
│   ├── training_all_episodes.json
│   └── ...
│
├── models/                          # 保存的模型
│   └── ...
│
├── logs/                            # 训练日志
│   └── ...
│
├── recommendation_environment.py    # 推荐环境实现
├── recommendation_trainer.py        # 训练器实现
├── TD3.py                          # TD3 算法实现
├── utils.py                        # 工具函数
├── run_recommendation_rl.py        # 主训练脚本
├── setup_environment.py            # 环境设置
├── config.yaml                     # 配置文件
├── requirements.txt                # 依赖包
└── README.md                       # 项目说明
```

## 🎯 核心文件

### 训练相关
- **run_recommendation_rl.py** - 主训练脚本（入口）
- **recommendation_trainer.py** - 训练循环实现
- **recommendation_environment.py** - 推荐环境
- **TD3.py** - TD3 算法
- **utils.py** - Replay Buffer 等工具

### 配置
- **config.yaml** - 所有训练参数配置

### 文档
- **README.md** - 项目说明
- **design/** - 设计文档文件夹

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置参数
编辑 `config.yaml` 设置训练参数

### 3. 运行训练
```bash
python run_recommendation_rl.py
```

### 4. 分析结果
```bash
python tools/analyze_replay_buffer_quality.py
python tools/visualize_buffer_quality.py
```

## 📊 数据流程

```
原始数据 (embeddings/, processed_data/)
    ↓
recommendation_environment.py (加载和处理)
    ↓
recommendation_trainer.py (训练循环)
    ↓
TD3.py (策略学习)
    ↓
results/ (保存结果)
```

## 🔧 开发指南

### 修改训练参数
→ 编辑 `config.yaml`

### 修改环境逻辑
→ 编辑 `recommendation_environment.py`

### 修改训练流程
→ 编辑 `recommendation_trainer.py`

### 修改算法
→ 编辑 `TD3.py`

### 添加分析工具
→ 在 `tools/` 文件夹添加脚本

### 更新文档
→ 在 `design/` 文件夹更新文档

## 📝 文件命名规范

### Python 文件
- 核心模块: `{module_name}.py`
- 工具脚本: `{action}_{target}.py`
- 示例: `analyze_buffer_quality.py`

### 文档文件
- 设计文档: `{component}_design.md`
- 总结文档: `{TOPIC}_SUMMARY.md`
- 指南文档: `{topic}_GUIDE.md`

### 数据文件
- JSON: `{description}.json`
- Pickle: `{description}.pkl`
- CSV: `{description}.csv`

## 🗂️ 文件夹用途

| 文件夹 | 用途 | 是否提交 |
|--------|------|----------|
| design/ | 设计文档 | ✅ 是 |
| tools/ | 分析工具 | ✅ 是 |
| legacy/ | 旧代码 | ✅ 是 |
| data_preprocessing/ | 预处理脚本 | ✅ 是 |
| embeddings/ | 嵌入数据 | ⚠️ 视大小 |
| processed_data/ | 处理后数据 | ⚠️ 视大小 |
| results/ | 训练结果 | ❌ 否 |
| models/ | 保存的模型 | ❌ 否 |
| logs/ | 日志 | ❌ 否 |
| __pycache__/ | Python 缓存 | ❌ 否 |

## 🔍 查找文件

### 我想...
- **运行训练** → `run_recommendation_rl.py`
- **修改配置** → `config.yaml`
- **了解架构** → `design/model_design_overview.md`
- **查看优化** → `design/OPTIMIZATION_SUMMARY.md`
- **分析结果** → `tools/analyze_*.py`
- **查看结果** → `results/`

---

**最后更新**: 2024-11-09
**项目状态**: 已优化，可用于训练
