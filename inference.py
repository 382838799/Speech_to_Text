from transformers import WhisperProcessor
from Whisper.Model import Model
import torch
import numpy as np


@torch.no_grad()
def infer(audio_array):
    model = Model()
    model.load_state_dict(torch.load('./Models/model_state_epoch_2.pth'))

    prediction = model.generate(audio_array)
    return prediction


# 假设我们有一个新的音频数据作为输入
new_audio_array = np.load('path_to_new_audio_file.npy')

# 进行推理
predicted_sentence = infer(new_audio_array)
print("识别结果: " + predicted_sentence)