# 项目使用说明

本项目包含两个主要模块：

- `fdr`：人脸检测、人脸特征提取、人脸相似度/置信度排序。
- `svc`：歌曲/视频音频分离、SVC 人声转换、背景音乐与合成人声混音。

默认按 Windows + CPU 环境使用。

## 第 1 部分：环境配置

### 0. 环境要求

建议环境：

- 系统：Windows 10/11 64 位。
- 计算设备：CPU 环境即可，不要求显卡。
- Python：Python 3.10。
- 内存：建议 16 GB 起步，处理较长音频时建议 32 GB。
- 存储：建议预留 20 GB 以上，模型、缓存、中间 wav 和输出文件会占用空间。

CPU 运行会比较慢，尤其是 `svc` 的 Demucs 分离和 SVC 推理，属于正常现象。

### 1. 安装 Python 3.10

Python 官网：

https://www.python.org/downloads/

安装完成后，在 PowerShell 中检查：

```powershell
py -3.10 --version
```

### 2. 创建并启动虚拟环境

在项目根目录运行：

```powershell
cd E:\test
py -3.10 -m venv myvenv
.\myvenv\Scripts\Activate.ps1
python --version
```

如果 PowerShell 不允许激活虚拟环境，可以先执行：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

然后重新运行：

```powershell
.\myvenv\Scripts\Activate.ps1
```

### 3. 安装依赖

#### 3.1 安装 MS Visual Studio 编译工具

安装 Microsoft C++ Build Tools：

https://visualstudio.microsoft.com/visual-cpp-build-tools/

安装时建议勾选 C++ build tools / MSVC / Windows SDK。安装后打开 "Developer PowerShell for VS" 或重新打开 PowerShell，检查：

```powershell
where cl
cl
```

能找到 `cl.exe`，并且 `cl` 能输出 Microsoft C/C++ 编译器信息即可。

#### 3.2 调整 pip 版本

```powershell
python -m pip install pip==24.0
python -m pip --version
```

#### 3.3 安装 torch CPU 版本

PyTorch CPU wheel 地址：

https://download.pytorch.org/whl/cpu

安装命令：

```powershell
python -m pip install torch==1.13.1+cpu torchaudio==0.13.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

检验：

```powershell
python -c "import torch, torchaudio; print(torch.__version__); print(torchaudio.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

CPU 环境下应看到类似：

```text
1.13.1+cpu
0.13.1+cpu
False
None
```

#### 3.4 安装其余 pip 依赖

先安装编译相关基础包：

```powershell
python -m pip install "setuptools<70" wheel "cython<3"
```

再安装项目依赖。当前仓库的依赖文件在根目录 `requirements.txt`：

```powershell
python -m pip install --no-build-isolation -r .\requirements.txt
```

安装后检查依赖冲突：

```powershell
python -m pip check
```

## 第 2 部分：fdr 人脸检测与识别

### 1. 模型和功能

`fdr` 使用 InsightFace 的 `buffalo_l` 模型做：

- 人脸检测。
- 人脸 embedding 特征提取。
- 查询人脸与图片库人脸的 cosine similarity 对比。
- 使用 `fdr/calibration_results.json` 把 cosine 分数校准成更直观的 confidence 百分比。
- 输出带人脸框和置信度标注的可视化图片。

模型目录要求：

```text
fdr/models/buffalo_l/
```

该目录下需要包含：

```text
1k3d68.onnx
2d106det.onnx
det_10g.onnx
genderage.onnx
w600k_r50.onnx
```

### 2. 两种运行模式

#### compare 模式

compare 模式会读取 `fdr/pictures/xin/` 下的图片，使用排序后的第一张图片作为基准图，然后和目录内其他图片逐一比较。

运行：

```powershell
python .\fdr\main.py --mode compare
```

适合已经准备好一组图片，想快速比较谁最像基准图。

#### capture 模式

capture 模式会打开摄像头，按空格拍摄一张查询图，然后与 `fdr/pictures/xin/` 下的图片库比较。

运行：

```powershell
python .\fdr\main.py --mode capture
```

摄像头窗口中：

- 按空格拍照。
- 按 `q` 取消。

### 3. 数据放在哪里，怎么操作

日常比对图片放在：

```text
fdr/pictures/xin/
```

支持格式：

```text
.jpg
.jpeg
.png
```

compare 模式会把 `fdr/pictures/xin/` 中排序后的第一张图片当作基准图。想指定基准图时，可以把它命名成更靠前的名字，例如：

```text
000_query.jpg
001_person_a.jpg
002_person_b.jpg
```

输出结果在：

```text
fdr/outputs/YYYYMMDD_HHMMSS/
```

主要查看：

```text
fdr/outputs/YYYYMMDD_HHMMSS/run_config.json
fdr/outputs/YYYYMMDD_HHMMSS/bbox_confidence/
```

`run_config.json` 保存完整排序结果，`bbox_confidence/` 保存带框和置信度的图片。

如果要重新生成校准数据，训练图片放在：

```text
fdr/aligned_images/
```

列表文件放在：

```text
fdr/train.txt
```

生成训练 pair 分数：

```powershell
python .\fdr\scripts\data_generate.py --train_txt .\fdr\train.txt --image_root .\fdr\aligned_images --output_csv .\fdr\train_pairs_with_score.csv --providers CPUExecutionProvider
```

重新训练置信度校准文件：

```powershell
python .\fdr\train_face_score_calibrators.py --train_csv .\fdr\train_pairs_with_score.csv --output_json .\fdr\calibration_results.json
```

一般日常使用不需要重新训练，只要 `fdr/calibration_results.json` 已存在即可。

## 第 3 部分：svc 人声转换与混音

### 1. 下载模型

权重文件建议都放到：

```text
svc/logs/44k_denoise/
```

与默认命令保持一致。

- `G_163200.pth`，推理常用，推荐：
  https://drive.google.com/file/d/1PzVt5En6hGgeodGd1SaeU8a94ZSLirbO/view?usp=sharing
- `G_0.pth`，可选，一般不用：
  https://drive.google.com/file/d/1VdWt7L-LCasvaoE3GPf4D9N9IHprrF7W/view?usp=sharing
- `D_163200.pth`，判别器权重，训练用，推理不需要：
  https://drive.google.com/file/d/15Vx1tS5gmuOLozbZeAEG1NC27lrI3cDW/view?usp=sharing
- `D_0.pth`，判别器权重，训练用，推理不需要：
  https://drive.google.com/file/d/1dudSLd6fSH3zF-yZgyJMWVzabiys9SXd/view?usp=sharing

日常推理只需要确认这个文件存在：

```text
svc/logs/44k_denoise/G_163200.pth
```

默认配置文件使用：

```text
svc/configs/config.json
```

### 2. 数据放在哪里

默认歌曲目录是：

```text
svc/wav/test02/
```

把要处理的音频或视频文件放进去即可。支持常见格式：

```text
.wav .mp3 .mp4 .m4a .flac .aac .ogg .opus .webm .mkv .mov .m4v
```

建议每个歌曲目录里只放一个原始输入文件。脚本会自动忽略这些生成文件：

```text
wav_voice.wav
wav_bgmusic.wav
my_voice.wav
output.wav
```

如果目录里有多个原始输入文件，需要用 `--input` 指定。

### 3. 怎么运行和调参

默认运行完整流程：

```powershell
python .\svc\main.py
```

默认流程包括：

1. 用 Demucs 分离人声和背景。
2. 用 SVC 模型把分离出来的人声转换成目标音色。
3. 把转换后的人声和背景音乐混合成最终 wav。

默认是 CPU：

```text
--device cpu
--demucs-device cpu
```

默认模型和配置：

```text
--model-path logs/44k_denoise/G_163200.pth
--config-path configs/config.json
--speaker myvoice_denoise_mono
--transpose -2
--f0-predictor rmvpe
--clip 15
```

注意：`svc/main.py` 中的相对路径都是相对于 `svc/` 目录解析的。

#### 调整音量大小

默认混音参数：

```text
--voice-gain 1.8
--background-gain 0.85
```

如果合成人声还小，可以继续加大人声或降低背景：

```powershell
python .\svc\main.py --voice-gain 2.2 --background-gain 0.75
```

如果人声太大或破音，可以降低：

```powershell
python .\svc\main.py --voice-gain 1.4 --background-gain 0.9
```

脚本会在混音时自动限制峰值，避免最终 wav 超过安全峰值。

#### 设置歌曲目录、输入和输出

使用其他歌曲目录：

```powershell
python .\svc\main.py --song-dir wav/my_song
```

指定输入文件：

```powershell
python .\svc\main.py --song-dir wav/my_song --input input.mp4
```

指定最终输出：

```powershell
python .\svc\main.py --song-dir wav/my_song --output wav/my_song/output_louder.wav
```

如果已经有分离结果和转换结果，只想重新混音：

```powershell
python .\svc\main.py --skip-separation --skip-conversion
```

重新混音并调整音量：

```powershell
python .\svc\main.py --skip-separation --skip-conversion --voice-gain 2.2 --background-gain 0.75
```

强制重新生成所有步骤：

```powershell
python .\svc\main.py --force
```

#### 结果在哪里看

默认输出目录：

```text
svc/wav/test02/
```

主要文件：

```text
wav_voice.wav      Demucs 分离出来的人声
wav_bgmusic.wav    Demucs 分离出来的背景音乐
my_voice.wav       SVC 转换后的人声
output.wav         最终混音结果
```

通常直接试听：

```text
svc/wav/test02/output.wav
```

如果处理 MP4/M4A/MP3 等非 wav 输入，脚本会自动使用 ffmpeg 解码到临时 wav。项目依赖中包含 `imageio-ffmpeg`，通常不需要额外安装 ffmpeg。
