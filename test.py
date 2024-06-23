from transformers import WhisperProcessor
from Utils.dataset import Dataset
from Whisper.Model import Model
import Whisper.forward_generate
import os
import torch
from scipy.io import wavfile
import numpy as np
from Utils.evaluation import evaluate_model
from Utils.saveaudio import save_Audio

@torch.no_grad()
def test():

    model = Model()
    model.load_state_dict(torch.load('./Models.pth'))
    dataset_test = Dataset('test')

    # 存储真实结果和识别结果
    ground_truths = []
    predictions = []

    for i in range(100):
        # 保存测试音频为wav文件
        audio_array = dataset_test.dataset[i]['audio']['array']
        rate = 16000
        save_Audio(audio_array,rate,i)

        ground_truth = dataset_test.dataset[i]['sentence']
        prediction = model.generate(dataset_test[i]['speech'])
        ground_truths.append(ground_truth)
        predictions.append(prediction)
        print("真实结果: " + ground_truth + "   " + "识别结果: " + prediction)
        print()

    # 计算评估指标
    evaluate_model(ground_truths,predictions)

if __name__ == "__main__":
    test()