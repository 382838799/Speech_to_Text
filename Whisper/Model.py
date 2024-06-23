from transformers import WhisperProcessor
import Whisper.forward_generate
import Whisper.encoder
import Whisper.decoder
import torch
import re
processor = WhisperProcessor.from_pretrained('openai/whisper-small',
                                             language='Chinese',
                                             task='transcribe')
class Model(torch.nn.Module):
    #Whisper模型构建
    def __init__(self):
        super().__init__()

        from transformers import WhisperForConditionalGeneration
        pretrained = WhisperForConditionalGeneration.from_pretrained(
            'openai/whisper-small')

        self.encoder = Whisper.encoder.load_encoder(pretrained.model.encoder)
        
        self.decoder = Whisper.decoder.load_decoder(pretrained.model.decoder)

        self.fc_out = torch.nn.Linear(768, 51865, bias=False)
        self.fc_out.load_state_dict(pretrained.proj_out.state_dict())

    def forward(self, speech, text):
        #向右偏移一位
        text = torch.cat([text[:, :1], text], dim=1)[:, :-1]
        kv = self.encoder(speech)
        out = self.decoder(x=text, kv=kv)

        return self.fc_out(out)
    
    def generate(self,speech):
        #推理，生成识别结果
        text = torch.LongTensor([[50258]])
        cache_kv = None
        kv = self.encoder(speech.unsqueeze(0))
        generate = [text.item()]

        for _ in range(100):
            text, cache_kv = Whisper.forward_generate.forward_decoder(self.decoder,
                                            x=text,
                                            kv=kv,
                                            cache_kv=cache_kv)

            text = self.fc_out(text).argmax(dim=2)
            generate.append(text.item())

            if text.item() == 50257:
                break
        #提取识别结果
        text=processor.decode(generate)
        pattern = re.compile(r'[\u4e00-\u9fa5]+')
        chinese_characters = pattern.findall(text)
        result = ''.join(chinese_characters)
        return result