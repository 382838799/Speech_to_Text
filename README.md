# Whisper Speech Recognition Project

这是一个基于 OpenAI Whisper 模型的语音识别项目，使用 Transformer 架构进行语音到文本的转换。

## 文件结构

.
├── Datasets # 数据集目录
├── Models # 模型保存目录
├── Utils
│ └── dataset.py # 数据集处理
│ └── evaluation.py # 模型评估
│ └── saveaudio.py # 保存音频
├── Whisper
│ ├── decoder.py # 解码器模块
│ ├── encoder.py # 编码器模块
│ ├── forward_generate.py # 文本生成
│ └── Model.py # 模型定义
├── test.py # 测试脚本
├── test2.py # 使用自己录制的音频测试
├── train.py # 训练脚本
├── requirements.txt # 需求环境
└── README.md # 项目说明文件


## 功能说明
本系统的主要功能是：给定一段音频，能够将其音频内容准确识别并转换成对应的文本。这一功能将在多种实际应用场景中发挥重要作用，例如会议记录、视频字幕生成、客服对话转录等。


## 运行方法
### 安装依赖
pip install -r requirements.txt

### 训练模型
直接运行train.py,运行后程序会自动下载数据集到Datasets文件夹中，然后开始进行训练，训练模型会保存至Models文件夹中

### 测试模型
1.运行test.py，运行后程序会将用于测试的音频保存至Audio文件夹中，然后输出真实结果和识别结果，测试结束计算词错误率和字符错误率
2.使用自己录制的wav文件进行测试：将录音文件放置testdata文件夹中，运行test2.py，程序会直接识别test2中的音频文件
