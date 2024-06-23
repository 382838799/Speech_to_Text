from scipy.io import wavfile
import os
import numpy as np
def save_Audio(audio_array,rate,i):
    #将测试集保存为音频文件
    audio_dir = './Audio'  # 音频保存路径
    os.makedirs(audio_dir, exist_ok=True)
    if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
            audio_array = np.int16(audio_array * 32767)
    output_path = os.path.join(audio_dir, f'test_audio{i+1}.wav')
    wavfile.write(output_path, rate, np.array(audio_array, dtype=np.int16))