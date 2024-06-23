from Utils.dataset import Dataset
from transformers import WhisperProcessor
from Whisper.Model import Model
import torch
import Whisper.forward_generate
import os

processor = WhisperProcessor.from_pretrained('openai/whisper-small',#加载预训练模型
                                             language='Chinese',
                                             task='transcribe')

def collate_fn(data):
    speech = [{'input_features': i['speech']} for i in data]
    speech = processor.feature_extractor.pad(
        speech, return_tensors='pt').input_features
    text = [{'input_ids': i['text']} for i in data]
    text = processor.tokenizer.pad(text, return_tensors='pt').input_ids
    return {'speech': speech, 'text': text}


def train():
    model_dir = './Models'#模型保存路径
    os.makedirs(model_dir, exist_ok=True)
    dataset = Dataset(split='train')
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=4,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)
    
    model=Model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train()
    model.to(device)
    start_epoch = 0
    checkpoint_path = 'checkpoint.pth'
    if os.path.exists(checkpoint_path):#加载checkpoint，继续上次训练
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    print("start train")
    for epoch in range(start_epoch,1):
        print('第',epoch,'轮训练开始：')
        for i, data in enumerate(loader):
            #if(i%10==0):
            print(i)
            for k in data.keys():
                data[k] = data[k].to(device)
            out = model(**data)

            loss = loss_fn(out.flatten(end_dim=-2), data['text'].flatten()) / 4
            loss.backward()
            if (i + 1) % 4 == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                model.zero_grad()
                optimizer.zero_grad()

            if (i + 1) % 100 == 0:
                print(epoch, i, loss.item())

        model_save_path = os.path.join(model_dir, f'model_state_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), model_save_path)#保存模型
        torch.save({#保存当前训练状态
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item()
        }, checkpoint_path)

if __name__ == "__main__":
    train()