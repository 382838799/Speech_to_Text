from transformers import WhisperProcessor
import torch
from IPython.display import Audio, display
import os
import torchaudio
from Whisper.Model import Model

# 初始化 WhisperProcessor
processor = WhisperProcessor.from_pretrained('openai/whisper-small',
                                             language='Chinese',
                                             task='transcribe')

def load_wav(file_path):
    # 使用 torchaudio 加载 WAV 文件
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000:
        # 重采样到 16000 Hz
        transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = transform(waveform)
    return waveform.squeeze(0)  # 删除多余的维度

@torch.no_grad()
def test():
    model = Model()
    model.load_state_dict(torch.load('./Models/model.pth'))
    # 将自己的 WAV 文件处理为 generate 函数的输入
    wav_file_path = "./TestData/test1.wav"
    audio_tensor = load_wav(wav_file_path)
    # 使用 WhisperProcessor 处理音频数据
    input_features = processor.feature_extractor(audio_tensor, sampling_rate=16000, return_tensors='pt').input_features[0]
    print(model.generate(input_features))

test()
