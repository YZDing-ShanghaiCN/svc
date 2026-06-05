
# SVC 推理使用说明（本仓库）

本仓库包含 SoVITS4 / SVC 的推理脚本 `inference_main.py`，以及人声分离（Demucs）等辅助工具。
本文档主要覆盖：

1. 环境安装配置
2. 推理 pipeline 流程与指令（数据放哪里、结果在哪看）
3. 需要 ignore 的文件清单（含云盘权重链接与放置说明）

---

## 1. 环境安装与配置

### 1.1 系统依赖（Linux）

建议先安装这些系统包：

- `ffmpeg`：Demucs/音频处理常用
- `libsndfile1`：Python 包 `soundfile` 依赖
- `build-essential`：编译 `fairseq` 等依赖时需要

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libsndfile1 build-essential
```

### 1.2 Python 环境（建议 Python 3.10）

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

### 1.3 安装 Python 依赖

仓库提供的 `requirement.txt` 以 **Windows CPU 部署 / Python 3.10** 为目标写的，但在 Linux 下用于 **CPU 推理** 通常也可用。

```bash
pip install -r requirement.txt
```

按功能选择的可选依赖（你用到哪个再装哪个）：

- `-f0p pm`：需要 `praat-parselmouth`
- `-f0p dio/harvest`：需要 `pyworld`
- `-f0p crepe`：需要 `torchcrepe`
- 训练聚类模型（可选）：`scikit-learn`、`tqdm`

```bash
pip install praat-parselmouth pyworld torchcrepe scikit-learn tqdm
```

> GPU 推理说明：
> - `requirement.txt` 里固定安装了 `torch==...+cpu` / `torchaudio==...+cpu`。
> - 如果你想用 GPU，请先按 PyTorch 官方命令安装与你 CUDA 版本匹配的 `torch/torchaudio`，再安装其它依赖（需要的话，把 `requirement.txt` 里这两行删掉或单独安装剩余包）。

### 1.4 必要模型文件检查

最小可跑通推理一般需要这些文件（本仓库多数已自带，缺的你需要补齐）：

- 推理模型（示例）：`logs/44k_denoise/G_163200.pth`
- 推理配置（示例）：`configs/config.json`（必须与模型匹配）
- Speech encoder（ContentVec，默认会读）：`pretrain/checkpoint_best_legacy_500.pt`
- F0 模型（当你用 `-f0p rmvpe`）：`pretrain/rmvpe.pt`

说明：本仓库默认通过 `.gitignore` 忽略大权重文件；如果你 clone 下来发现 `logs/` 或 `pretrain/` 里缺少权重，请先看第 3 节的“云盘权重与 ignore 规则”。

可选功能：

- 若开启 `--enhance` 或浅扩散（`--shallow_diffusion/--only_diffusion`），通常还需要把 NSF-HiFiGAN vocoder 权重放到 `pretrain/nsf_hifigan/model`（当前目录只有占位文件 `put_nsf_hifigan_ckpt_here`）。

---

## 2. Pipeline 流程与指令（数据放哪里 / 结果在哪看）

### 2.1 推荐目录约定（示例）

仓库自带了示例目录 `wav/test/`，你可以按这个约定放数据：

```text
wav/test/
	input.wav          # 原始音频（可以是歌曲/录音）
	wav_voice.wav       # Demucs 分离出来的人声（步骤A生成）
	wav_bgmusic.wav     # Demucs 分离出来的伴奏（步骤A生成）
	my_voice.wav        # 推理后的目标音色人声（步骤B输出）
```

### 2.2 步骤 A（可选）：人声分离（Demucs）

如果你的 `input.wav` 是带伴奏的歌曲，建议先把人声/伴奏分离。

Linux/macOS（bash）示例（同 `cmd.txt`）：

```bash
SONG_DIR="wav/test"

demucs --two-stems=vocals -o "$SONG_DIR/_demucs" "$SONG_DIR/input.wav"

mv -f "$SONG_DIR/_demucs/htdemucs/input/vocals.wav" "$SONG_DIR/wav_voice.wav"
mv -f "$SONG_DIR/_demucs/htdemucs/input/no_vocals.wav" "$SONG_DIR/wav_bgmusic.wav"

rm -rf "$SONG_DIR/_demucs"
```

Windows（PowerShell）CPU 示例（同 `cmd_windows_cpu.ps1`）：

```powershell
$SongDir = "wav/test"

demucs --two-stems=vocals -o "$SongDir/_demucs" "$SongDir/input.wav"

Move-Item -Force "$SongDir/_demucs/htdemucs/input/vocals.wav" "$SongDir/wav_voice.wav"
Move-Item -Force "$SongDir/_demucs/htdemucs/input/no_vocals.wav" "$SongDir/wav_bgmusic.wav"

Remove-Item -Recurse -Force "$SongDir/_demucs"
```

### 2.3 步骤 B：SVC 推理（音色转换）

最常用的方式是给 `inference_main.py` 指定：模型、配置、输入音频、输出路径、目标角色名。

Linux 示例（同 `cmd.txt`）：

```bash
SONG_DIR="wav/test"

python inference_main.py \
	-m logs/44k_denoise/G_163200.pth \
	-c configs/config.json \
	-ip "$SONG_DIR/wav_voice.wav" \
	-op "$SONG_DIR/my_voice.wav" \
	-t -2 \
	-s myvoice_denoise_mono \
	-f0p rmvpe \
	-cl 15 \
	-wf wav
```

Windows CPU 示例（同 `cmd_windows_cpu.ps1`，多了 `-d cpu`）：

```powershell
python inference_main.py `
	-m logs/44k_denoise/G_163200.pth `
	-c configs/config.json `
	-ip "$SongDir/wav_voice.wav" `
	-op "$SongDir/my_voice.wav" `
	-t -2 `
	-s myvoice_denoise_mono `
	-f0p rmvpe `
	-cl 15 `
	-wf wav `
	-d cpu
```

#### 输入/输出在哪里看？

- 输入：你通过 `-ip` 指定的文件（示例为 `wav/test/wav_voice.wav`）
- 输出：
	- 如果你指定了 `-op`：输出就是 `-op` 指向的路径（示例为 `wav/test/my_voice.wav`）
	- 如果不指定 `-op`：默认输出到 `results/`，文件名会自动拼上音高、角色名、f0 方案等信息

#### 关键参数速查

- `-m / --model_path`：推理模型 `.pth`
- `-c / --config_path`：配置 `config.json`
- `-s / --spk_list`：目标角色名（必须存在于 config 的 `spk` 字段里；例如 `configs/config.json` 里有 `myvoice_denoise_mono`）
- `-t / --trans`：移调（半音），可为负
- `-f0p / --f0_predictor`：F0 预测器（`rmvpe`/`pm`/`dio`/`harvest`/`crepe`/`fcpe`）
- `-cl / --clip`：强制切片长度（秒）。`0` 表示自动切片；大音频建议给一个值（例如 15）
- `-d / --device`：设备（例如 `cpu` / `cuda`）。不填会自动选择
- 查看完整参数：`python inference_main.py -h`

### 2.4 批量推理（不写 -ip/-op）

当你不传 `-ip` 时，会从 `raw/` 目录读取 `-n` 指定的文件名；不传 `-op` 时，会写入 `results/`。
注意：脚本不会自动创建目录，第一次用请手动创建。

```bash
mkdir -p raw results

# 把待处理音频放到 raw/
cp /path/to/a.wav raw/a.wav

python inference_main.py \
	-m logs/44k_denoise/G_163200.pth \
	-c configs/config.json \
	-n a.wav \
	-s myvoice_denoise_mono \
	-t 0 \
	-f0p rmvpe \
	-wf wav

# 输出会在 results/ 下
```

---

## 3. 需要 ignore 的文件有哪些（以及云盘权重放哪里）

本仓库的策略是：**代码/配置进 Git**，**大文件（模型权重/训练日志/音频数据与输出）不进 Git**。

### 3.1 云盘权重（下载后放置到哪里）

你提供的权重文件如下（建议都放到 `logs/44k_denoise/`，与示例命令保持一致）：

- `G_163200.pth`（推理常用，推荐）
	- https://drive.google.com/file/d/1PzVt5En6hGgeodGd1SaeU8a94ZSLirbO/view?usp=sharing
- `G_0.pth`（可选，一般不用）
	- https://drive.google.com/file/d/1VdWt7L-LCasvaoE3GPf4D9N9IHprrF7W/view?usp=sharing
- `D_163200.pth`（判别器权重，**训练用**，推理不需要）
	- https://drive.google.com/file/d/15Vx1tS5gmuOLozbZeAEG1NC27lrI3cDW/view?usp=sharing
- `D_0.pth`（判别器权重，**训练用**，推理不需要）
	- https://drive.google.com/file/d/1dudSLd6fSH3zF-yZgyJMWVzabiys9SXd/view?usp=sharing

放置后的目录示例：

```text
logs/44k_denoise/
	G_163200.pth
	D_163200.pth
	(其它训练日志文件可有可无)
```

推理时只要把 `-m` 指向 `G_*.pth` 即可（例如 `-m logs/44k_denoise/G_163200.pth`）。

可选：用命令行下载（Linux/macOS，需先安装 `gdown`）：

```bash
pip install -U gdown
mkdir -p logs/44k_denoise

gdown --fuzzy "https://drive.google.com/file/d/1PzVt5En6hGgeodGd1SaeU8a94ZSLirbO/view?usp=sharing" -O logs/44k_denoise/G_163200.pth
gdown --fuzzy "https://drive.google.com/file/d/1VdWt7L-LCasvaoE3GPf4D9N9IHprrF7W/view?usp=sharing" -O logs/44k_denoise/G_0.pth
gdown --fuzzy "https://drive.google.com/file/d/15Vx1tS5gmuOLozbZeAEG1NC27lrI3cDW/view?usp=sharing" -O logs/44k_denoise/D_163200.pth
gdown --fuzzy "https://drive.google.com/file/d/1dudSLd6fSH3zF-yZgyJMWVzabiys9SXd/view?usp=sharing" -O logs/44k_denoise/D_0.pth
```

### 3.2 当前已忽略（不会提交到 Git）的文件

仓库根目录的 `.gitignore` 主要包含这些规则（核心目的：避免把大文件和生成物提交到 Git）：

- **模型权重/导出文件**：`*.pth`、`*.pt`、`*.ckpt`、`*.onnx`
- **音频文件**：`*.wav`（以及常见输出格式如 `*.flac` 等）
- **训练日志目录**：`logs/`
- **推理/数据中间目录**：`raw/`、`results/`、`**/_demucs/`
- **运行时临时文件**：`inference/chunks_temp.json`
- **Python 缓存**：`__pycache__/`、`*.pyc`、`*.pyo`、`*.pyd`

因此：

- `logs/44k_denoise/G_163200.pth` 等权重应通过云盘分发（见 3.1），而不是进 Git
- `wav/test/*.wav`、推理输出音频也默认不会进 Git


