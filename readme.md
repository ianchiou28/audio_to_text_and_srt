# Audio to SRT/TXT

使用 OpenAI 的 Whisper 模型将音频文件转换为字幕文件（.srt）或文本文件（.txt）。

## 功能特点

- 支持 .wav、.mp3、.m4a 等常见音频格式
- 自动检测 GPU 可用性，支持 CUDA 加速
- 提供三种转换脚本：
  - `atb.py`: 同时生成 .srt 和 .txt 文件
  - `ats.py`: 仅生成 .srt 字幕文件
  - `att.py`: 仅生成 .txt 文本文件
- 支持多种 Whisper 模型尺寸：
  - tiny（最快，精度最低）
  - base
  - small
  - medium
  - large
  - large-v2
  - large-v3（最慢，精度最高）
- 自动语言检测
- 完整的 GPU 环境检查和提示

## 环境要求

- Python 3.x
- PyTorch
- Whisper
- CUDA 环境（可选，但强烈建议）

## 安装说明

1. 安装 Python 依赖：
```bash
# 安装支持 CUDA 的 PyTorch（推荐）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装 Whisper
pip install -U openai-whisper

# 安装 FFmpeg（Windows 用户）
choco install ffmpeg
```

2. 如果遇到安装问题，可尝试：
```bash
pip install setuptools-rust
pip install git+https://github.com/openai/whisper.git
```

## 使用方法

1. 将音频文件放在脚本同一目录下

2. 编辑对应脚本文件底部的设置区域：
   - 修改 `AUDIO_FILE` 为你的音频文件名
   - 选择合适的 `MODEL_SIZE`

3. 运行对应的脚本：

```bash
# 同时生成 .srt 和 .txt 文件
python atb.py

# 仅生成 .srt 字幕文件
python ats.py

# 仅生成 .txt 文本文件
python att.py
```

## 配置选项

在每个脚本文件的底部都有设置区域，您可以修改：

- `AUDIO_FILE`: 音频文件名称（支持相对路径和绝对路径）
- `MODEL_SIZE`: Whisper 模型大小，建议配置：
  - CPU 模式：使用 "tiny" 或 "base" 模型
  - GPU 模式：可以使用 "large-v3" 获得最佳效果

## 注意事项

1. 首次运行时会自动下载选定的 Whisper 模型
2. 程序会自动检测 GPU 可用性并给出相应提示
3. 建议使用 GPU 运行以获得更快的转录速度
4. 使用 large 系列模型时需要较大的 GPU 内存
5. 如果音频文件不存在，atb.py 会尝试创建演示文件

## 输出示例

运行脚本后将生成（以音频文件名为基础）：

- `.srt` 文件: 包含时间轴的字幕文件，格式如下：
```
1
00:00:00,000 --> 00:00:05,000
转录的文本内容

2
00:00:05,000 --> 00:00:10,000
下一段文本内容
```

- `.txt` 文件: 纯文本转录结果

程序还会在控制台显示：
- GPU 检查结果
- 检测到的语言
- 完整的转录文本内容

