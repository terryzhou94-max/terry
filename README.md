# 嘉陵江流域物理流量预测项目骨架

该项目提供了一个模块化的物理型流量预测框架，可根据配置文件自动加载嘉陵江流域DEM、坡度、土壤、气象强迫和各子流域数据，计算潜在蒸散发（PET），并对每个子流域分别进行率定与模拟。

## 目录结构

```
src/
  jialing_model/
    config.py                # 配置解析
    pipeline.py              # 主流程控制
    data/
      loader.py              # 数据加载
      preprocess.py          # PET计算与时间序列对齐
    model/
      hydrology.py           # 线性水库模型与率定
      parameterization.py    # 基于子流域属性的参数约束
      metrics.py             # NSE/KGE指标
    utils/
      progress.py            # 进度条
  main.py                    # 命令行入口
config/
  example_config.yaml        # 配置示例
```

## 快速开始

1. 准备配置文件，参考 `config/example_config.yaml`，填写DEM、坡度、土壤、气象、流量及各子流域数据路径和字段。
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 执行：
   ```bash
   python -m src.main config/your_config.yaml --iterations 5000
   ```

执行过程中会在 PyCharm 控制台动态打印率定进度和NSE/KGE最优值。结果默认输出至配置文件中的 `output_dir`。

## 依赖

- pandas
- numpy
- pyyaml
- geopandas（用于矢量数据，可选）
- rasterio（用于栅格数据，可选）

根据需要补充scipy、numba等进一步优化运行效率和指标表现。

