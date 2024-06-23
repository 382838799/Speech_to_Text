# import torch

# class Atten(torch.nn.Module):

#     def __init__(self):
#         super().__init__()

#         self.q = torch.nn.Linear(768, 768, bias=True)  # 查询向量线性变换
#         self.k = torch.nn.Linear(768, 768, bias=False)  # 键向量线性变换
#         self.v = torch.nn.Linear(768, 768, bias=True)  # 值向量线性变换
#         self.out = torch.nn.Linear(768, 768, bias=True)  # 输出线性变换

#     def forward(self, x, mask):
#         b, lens, _ = x.size()

#         q = self.q(x) * 0.125
#         k = self.k(x)
#         v = self.v(x)

#         q = q.reshape(b, lens, 12, 64).transpose(1,
#                                                  2).reshape(b * 12, lens, 64)
#         k = k.reshape(b, lens, 12, 64).transpose(1,
#                                                  2).reshape(b * 12, lens, 64)
#         v = v.reshape(b, lens, 12, 64).transpose(1,
#                                                  2).reshape(b * 12, lens, 64)


#         atten = q.bmm(k.transpose(1, 2)).reshape(b, 12, lens, lens) + mask
#         atten = atten.reshape(b * 12, lens, lens).softmax(dim=-1).bmm(v)
#         atten = atten.reshape(b, 12, lens,
#                               64).transpose(1, 2).reshape(b, lens, 768)

#         return self.out(atten)



# class CrossAtten(torch.nn.Module):

#     def __init__(self):
#         super().__init__()

#         self.q = torch.nn.Linear(768, 768, bias=True)
#         self.k = torch.nn.Linear(768, 768, bias=False)
#         self.v = torch.nn.Linear(768, 768, bias=True)

#         self.out = torch.nn.Linear(768, 768, bias=True)

#     def forward(self, x, kv):
#         b, lens, _ = x.size()

#         q = self.q(x) * 0.125
#         k = self.k(kv)
#         v = self.v(kv)

#         q = q.reshape(b, lens, 12, 64).transpose(1,
#                                                  2).reshape(b * 12, lens, 64)
#         k = k.reshape(b, 1500, 12, 64).transpose(1,
#                                                  2).reshape(b * 12, 1500, 64)
#         v = v.reshape(b, 1500, 12, 64).transpose(1,
#                                                  2).reshape(b * 12, 1500, 64)

#         atten = q.bmm(k.transpose(1, 2)).softmax(dim=-1).bmm(v)
#         atten = atten.reshape(b, 12, lens,
#                               64).transpose(1, 2).reshape(b, lens, 768)

#         return self.out(atten)


# class Layer(torch.nn.Module):

#     def __init__(self):
#         super().__init__()

#         self.norm1 = torch.nn.LayerNorm(768)
#         self.atten = Atten()

#         self.norm2 = torch.nn.LayerNorm(768)
#         self.cross_atten = CrossAtten()

#         self.s = torch.nn.Sequential(
#             torch.nn.LayerNorm(768),
#             torch.nn.Linear(768, 3072),
#             torch.torch.nn.GELU(),
#             torch.nn.Linear(3072, 768),
#         )

#     def forward(self, x, mask, kv):
#         x = self.atten(self.norm1(x), mask=mask) + x
#         x = self.cross_atten(self.norm2(x), kv=kv) + x

#         return self.s(x) + x

# def get_mask(b, lens):
#     mask = torch.full((lens, lens), -float('inf'))

#     t = torch.arange(lens)
#     t = t < (t + 1).reshape(lens, 1)
#     mask.masked_fill_(t, 0.0)

#     return mask.reshape(1, 1, lens, lens).repeat(b, 1, 1, 1)


# class Decoder(torch.nn.Module):

#     def __init__(self):
#         super().__init__()
#         self.embed = torch.nn.Embedding(51865, 768, 50257)
#         self.embed_pos = torch.nn.Embedding(448, 768)

#         self.layer = torch.nn.ModuleList([Layer() for _ in range(12)])
#         self.norm = torch.nn.LayerNorm(768)

#     def forward(self, x, kv):
#         mask = get_mask(*x.shape).to(x.device)
        
#         x = self.embed(x) + self.embed_pos.weight[:x.shape[1]]

#         for i in self.layer:
#             x = i(x, mask=mask, kv=kv)

#         return self.norm(x)


# def load_decoder(pretrained):
#     decoder = Decoder()
#     # 加载预训练模型的权重
#     decoder.embed.load_state_dict(pretrained.embed_tokens.state_dict())
#     decoder.embed_pos.load_state_dict(pretrained.embed_positions.state_dict())
#     decoder.norm.load_state_dict(pretrained.layer_norm.state_dict())

#     # 加载每层的自注意力、交叉注意力和前馈神经网络的权重
#     for i in range(12):
#         decoder.layer[i].norm1.load_state_dict(
#             pretrained.layers[i].self_attn_layer_norm.state_dict())

#         decoder.layer[i].atten.q.load_state_dict(
#             pretrained.layers[i].self_attn.q_proj.state_dict())
#         decoder.layer[i].atten.k.load_state_dict(
#             pretrained.layers[i].self_attn.k_proj.state_dict())
#         decoder.layer[i].atten.v.load_state_dict(
#             pretrained.layers[i].self_attn.v_proj.state_dict())
#         decoder.layer[i].atten.out.load_state_dict(
#             pretrained.layers[i].self_attn.out_proj.state_dict())

#         decoder.layer[i].norm2.load_state_dict(
#             pretrained.layers[i].encoder_attn_layer_norm.state_dict())

#         decoder.layer[i].cross_atten.q.load_state_dict(
#             pretrained.layers[i].encoder_attn.q_proj.state_dict())
#         decoder.layer[i].cross_atten.k.load_state_dict(
#             pretrained.layers[i].encoder_attn.k_proj.state_dict())
#         decoder.layer[i].cross_atten.v.load_state_dict(
#             pretrained.layers[i].encoder_attn.v_proj.state_dict())
#         decoder.layer[i].cross_atten.out.load_state_dict(
#             pretrained.layers[i].encoder_attn.out_proj.state_dict())

#         decoder.layer[i].s[0].load_state_dict(
#             pretrained.layers[i].final_layer_norm.state_dict())

#         decoder.layer[i].s[1].load_state_dict(
#             pretrained.layers[i].fc1.state_dict())
#         decoder.layer[i].s[3].load_state_dict(
#             pretrained.layers[i].fc2.state_dict())

#     return decoder

import torch

class Atten(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(768, 768, bias=True)  # 查询向量线性变换
        self.k = torch.nn.Linear(768, 768, bias=False)  # 键向量线性变换
        self.v = torch.nn.Linear(768, 768, bias=True)  # 值向量线性变换
        self.out = torch.nn.Linear(768, 768, bias=True)  # 输出线性变换

    def forward(self, x, mask):
        b, lens, _ = x.size()

        # 计算查询、键和值向量
        q = self.q(x) * 0.125
        k = self.k(x)
        v = self.v(x)

        # 重新塑造和转置查询、键和值向量
        q = q.reshape(b, lens, 12, 64).transpose(1, 2).reshape(b * 12, lens, 64)
        k = k.reshape(b, lens, 12, 64).transpose(1, 2).reshape(b * 12, lens, 64)
        v = v.reshape(b, lens, 12, 64).transpose(1, 2).reshape(b * 12, lens, 64)

        # 计算注意力得分并加上掩码
        atten = q.bmm(k.transpose(1, 2)).reshape(b, 12, lens, lens) + mask
        # 对注意力得分进行softmax归一化并计算注意力输出
        atten = atten.reshape(b * 12, lens, lens).softmax(dim=-1).bmm(v)
        atten = atten.reshape(b, 12, lens, 64).transpose(1, 2).reshape(b, lens, 768)

        return self.out(atten)  # 返回经过线性变换的注意力输出

class CrossAtten(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(768, 768, bias=True)  # 查询向量线性变换
        self.k = torch.nn.Linear(768, 768, bias=False)  # 键向量线性变换
        self.v = torch.nn.Linear(768, 768, bias=True)  # 值向量线性变换
        self.out = torch.nn.Linear(768, 768, bias=True)  # 输出线性变换

    def forward(self, x, kv):
        b, lens, _ = x.size()

        # 计算查询、键和值向量
        q = self.q(x) * 0.125
        k = self.k(kv)
        v = self.v(kv)

        # 重新塑造和转置查询、键和值向量
        q = q.reshape(b, lens, 12, 64).transpose(1, 2).reshape(b * 12, lens, 64)
        k = k.reshape(b, 1500, 12, 64).transpose(1, 2).reshape(b * 12, 1500, 64)
        v = v.reshape(b, 1500, 12, 64).transpose(1, 2).reshape(b * 12, 1500, 64)

        # 计算交叉注意力得分并进行softmax归一化
        atten = q.bmm(k.transpose(1, 2)).softmax(dim=-1).bmm(v)
        atten = atten.reshape(b, 12, lens, 64).transpose(1, 2).reshape(b, lens, 768)

        return self.out(atten)  # 返回经过线性变换的注意力输出

class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(768)  # 层归一化
        self.atten = Atten()  # 自注意力模块
        self.norm2 = torch.nn.LayerNorm(768)  # 层归一化
        self.cross_atten = CrossAtten()  # 交叉注意力模块
        self.s = torch.nn.Sequential(
            torch.nn.LayerNorm(768),
            torch.nn.Linear(768, 3072),
            torch.nn.GELU(),
            torch.nn.Linear(3072, 768),
        )  # 前馈神经网络

    def forward(self, x, mask, kv):
        x = self.atten(self.norm1(x), mask=mask) + x  # 自注意力层
        x = self.cross_atten(self.norm2(x), kv=kv) + x  # 交叉注意力层
        return self.s(x) + x  # 前馈神经网络层

def get_mask(b, lens):
    mask = torch.full((lens, lens), -float('inf'))  # 创建全负无穷掩码
    t = torch.arange(lens)
    t = t < (t + 1).reshape(lens, 1)
    mask.masked_fill_(t, 0.0)  # 将上三角部分填充为0
    return mask.reshape(1, 1, lens, lens).repeat(b, 1, 1, 1)  # 扩展掩码维度以匹配批次大小

class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(51865, 768, 50257)  # 嵌入层
        self.embed_pos = torch.nn.Embedding(448, 768)  # 位置嵌入层
        self.layer = torch.nn.ModuleList([Layer() for _ in range(12)])  # 12层解码器层
        self.norm = torch.nn.LayerNorm(768)  # 层归一化

    def forward(self, x, kv):
        mask = get_mask(*x.shape).to(x.device)  # 获取掩码并转移到与输入相同的设备上
        x = self.embed(x) + self.embed_pos.weight[:x.shape[1]]  # 嵌入输入和位置
        for i in self.layer:
            x = i(x, mask=mask, kv=kv)  # 通过每一层进行前向传播
        return self.norm(x)  # 返回标准化的输出

def load_decoder(pretrained):
    decoder = Decoder()
    # 加载预训练模型的权重
    decoder.embed.load_state_dict(pretrained.embed_tokens.state_dict())
    decoder.embed_pos.load_state_dict(pretrained.embed_positions.state_dict())
    decoder.norm.load_state_dict(pretrained.layer_norm.state_dict())

    for i in range(12):
        # 加载每层的自注意力、交叉注意力和前馈神经网络的权重
        decoder.layer[i].norm1.load_state_dict(pretrained.layers[i].self_attn_layer_norm.state_dict())
        decoder.layer[i].atten.q.load_state_dict(pretrained.layers[i].self_attn.q_proj.state_dict())
        decoder.layer[i].atten.k.load_state_dict(pretrained.layers[i].self_attn.k_proj.state_dict())
        decoder.layer[i].atten.v.load_state_dict(pretrained.layers[i].self_attn.v_proj.state_dict())
        decoder.layer[i].atten.out.load_state_dict(pretrained.layers[i].self_attn.out_proj.state_dict())
        decoder.layer[i].norm2.load_state_dict(pretrained.layers[i].encoder_attn_layer_norm.state_dict())
        decoder.layer[i].cross_atten.q.load_state_dict(pretrained.layers[i].encoder_attn.q_proj.state_dict())
        decoder.layer[i].cross_atten.k.load_state_dict(pretrained.layers[i].encoder_attn.k_proj.state_dict())
        decoder.layer[i].cross_atten.v.load_state_dict(pretrained.layers[i].encoder_attn.v_proj.state_dict())
        decoder.layer[i].cross_atten.out.load_state_dict(pretrained.layers[i].encoder_attn.out_proj.state_dict())
        decoder.layer[i].s[0].load_state_dict(pretrained.layers[i].final_layer_norm.state_dict())
        decoder.layer[i].s[1].load_state_dict(pretrained.layers[i].fc1.state_dict())
        decoder.layer[i].s[3].load_state_dict(pretrained.layers[i].fc2.state_dict())

    return decoder  # 返回加载了预训练权重的解码器
