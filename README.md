# 项目说明

本仓库用于语音转换/合成相关的训练与推理工作，当前目录已包含训练数据、模型产物与推理入口。

## 1. 训练结果（本次训练产物）

- 训练后的模型权重通常存放在 [trained/](trained/)（例如最新 checkpoint）。
- 训练日志与曲线数据通常在 [logs/](logs/)，便于回看收敛情况。
- 本次训练使用的配置保存在 [configs/](configs/)，对应的数据清单在 [filelists/](filelists/)。
- 训练数据与处理后的数据分别在 [dataset_raw/](dataset_raw/) 与 [dataset/](dataset/)。

可按需补充本次训练的关键说明：

- 模型名称/版本：
- 训练轮数与关键指标：
- 最佳 checkpoint：

## 2. 测试流程（输入到输出的 Pipeline）

- 输入：将待转换音频放入 [pipeline_input/](pipeline_input/)，或通过 [webUI.py](webUI.py) / [flask_api.py](flask_api.py) 上传。
- 预处理：切分与重采样（[inference/slicer.py](inference/slicer.py)、[resample.py](resample.py)）。
- 特征提取：内容特征与 F0（[preprocess_hubert_f0.py](preprocess_hubert_f0.py)）。
- 模型推理：加载训练好的模型进行转换（[inference/infer_tool.py](inference/infer_tool.py)、[inference_main.py](inference_main.py)）。
- 输出：结果写入 [pipeline_output/](pipeline_output/) 或 [results/](results/)（部分流程也会输出到 [converted/](converted/)）。

## 3. 开发环境（简述）

- 当前在 Linux 环境中进行开发与调试（不展开配置细节）。
- 核心以 Python 脚本为主，训练入口在 [train.py](train.py)，推理入口在 [inference_main.py](inference_main.py)。
- 提供可选的 Web 交互与 API：见 [webUI.py](webUI.py)、[flask_api.py](flask_api.py)。
- 如需 Notebook 形式的实验记录，可参考 [sovits4_for_colab.ipynb](sovits4_for_colab.ipynb)。
