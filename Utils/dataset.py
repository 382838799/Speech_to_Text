import torch
from transformers import WhisperProcessor
from datasets import load_dataset, Audio
import os

processor = WhisperProcessor.from_pretrained('openai/whisper-small',
                                             language='Chinese',
                                             task='transcribe')

class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):

        cache_dir = './Datasets'
        os.makedirs(cache_dir, exist_ok=True)
        #使用common_voice_11_0里的中文语音数据集
        dataset = load_dataset(path='mozilla-foundation/common_voice_11_0',
                               name='zh-CN',
                               split=split)
        #训练数据集大小设置为5000，测试数据及大小设置为100
        size = 5000 if split == 'train' else 100 
        dataset = dataset.shuffle(seed=0).select(range(size))
        dataset = dataset.cast_column('audio', Audio(sampling_rate=16000))
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        data = self.dataset[i]
        speech = processor.feature_extractor(
            data['audio']['array'], sampling_rate=16000,
            return_tensors='pt').input_features[0]
        text = processor.tokenizer(data['sentence'],
                                   return_tensors='pt').input_ids[0]
        return {'speech': speech, 'text': text}
