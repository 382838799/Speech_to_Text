# import torch


# class Atten(torch.nn.Module):

#     def __init__(self):
#         super().__init__()

#         self.q = torch.nn.Linear(768, 768, bias=True)
#         self.k = torch.nn.Linear(768, 768, bias=False)
#         self.v = torch.nn.Linear(768, 768, bias=True)
#         self.out = torch.nn.Linear(768, 768, bias=True)

#     def forward(self, x):
#         q = self.q(x) * 0.125
#         k = self.k(x)
#         v = self.v(x)

#         q = q.reshape(-1, 1500, 12, 64).transpose(1, 2).reshape(-1, 1500, 64)
#         k = k.reshape(-1, 1500, 12, 64).transpose(1, 2).reshape(-1, 1500, 64)
#         v = v.reshape(-1, 1500, 12, 64).transpose(1, 2).reshape(-1, 1500, 64)


#         atten = q.bmm(k.transpose(1, 2)).softmax(dim=-1).bmm(v)
#         atten = atten.reshape(-1, 12, 1500,
#                               64).transpose(1, 2).reshape(-1, 1500, 768)
#         atten = self.out(atten)

#         return atten


# class Layer(torch.nn.Module):

#     def __init__(self):
#         super().__init__()
#         self.s1 = torch.nn.Sequential(
#             torch.nn.LayerNorm(768),
#             Atten(),
#         )

#         self.s2 = torch.nn.Sequential(
#             torch.nn.LayerNorm(768),
#             torch.nn.Linear(768, 3072),
#             torch.nn.GELU(),
#             torch.nn.Linear(3072, 768),
#         )

#     def forward(self, x):
#         x = self.s1(x) + x
#         return self.s2(x) + x



# class Encoder(torch.nn.Module):

#     def __init__(self):
#         super().__init__()
#         self.s1 = torch.nn.Sequential(
#             torch.nn.Conv1d(80, 768, kernel_size=3, stride=1, padding=1),
#             torch.nn.GELU(),
#             torch.nn.Conv1d(768, 768, kernel_size=3, stride=2, padding=1),
#             torch.nn.GELU(),
#         )

#         self.embed = torch.nn.Embedding(1500, 768)

#         s2 = [Layer() for _ in range(12)]
#         s2.append(torch.nn.LayerNorm(768))
#         self.s2 = torch.nn.Sequential(*s2)

#     def forward(self, x):
#         x = self.s1(x).permute(0, 2, 1) + self.embed.weight

#         return self.s2(x)


# def load_encoder(pretrained):
#     encoder = Encoder()

#     encoder.s1[0].load_state_dict(pretrained.conv1.state_dict())
#     encoder.s1[2].load_state_dict(pretrained.conv2.state_dict())
#     encoder.embed.load_state_dict(pretrained.embed_positions.state_dict())

#     for i in range(12):
#         encoder.s2[i].s1[1].q.load_state_dict(
#             pretrained.layers[i].self_attn.q_proj.state_dict())
#         encoder.s2[i].s1[1].k.load_state_dict(
#             pretrained.layers[i].self_attn.k_proj.state_dict())
#         encoder.s2[i].s1[1].v.load_state_dict(
#             pretrained.layers[i].self_attn.v_proj.state_dict())
#         encoder.s2[i].s1[1].out.load_state_dict(
#             pretrained.layers[i].self_attn.out_proj.state_dict())

#         encoder.s2[i].s1[0].load_state_dict(
#             pretrained.layers[i].self_attn_layer_norm.state_dict())
#         encoder.s2[i].s2[0].load_state_dict(
#             pretrained.layers[i].final_layer_norm.state_dict())
#         encoder.s2[i].s2[1].load_state_dict(
#             pretrained.layers[i].fc1.state_dict())
#         encoder.s2[i].s2[3].load_state_dict(
#             pretrained.layers[i].fc2.state_dict())

#     encoder.s2[12].load_state_dict(pretrained.layer_norm.state_dict())

#     return encoder

import torch

class Atten(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 定义查询（query）、键（key）和值（value）的线性变换层
        self.q = torch.nn.Linear(768, 768, bias=True)
        self.k = torch.nn.Linear(768, 768, bias=False)
        self.v = torch.nn.Linear(768, 768, bias=True)
        self.out = torch.nn.Linear(768, 768, bias=True)

    def forward(self, x):
        q = self.q(x) * 0.125  # 查询向量线性变换并缩放
        k = self.k(x)  # 键向量线性变换
        v = self.v(x)  # 值向量线性变换

        # 重塑并转置查询、键和值向量以适应多头注意力机制
        q = q.reshape(-1, 1500, 12, 64).transpose(1, 2).reshape(-1, 1500, 64)
        k = k.reshape(-1, 1500, 12, 64).transpose(1, 2).reshape(-1, 1500, 64)
        v = v.reshape(-1, 1500, 12, 64).transpose(1, 2).reshape(-1, 1500, 64)

        # 计算注意力分数并应用softmax进行归一化，再计算注意力加权值向量
        atten = q.bmm(k.transpose(1, 2)).softmax(dim=-1).bmm(v)
        
        # 重塑并转置回原始形状
        atten = atten.reshape(-1, 12, 1500, 64).transpose(1, 2).reshape(-1, 1500, 768)
        
        # 输出线性变换
        atten = self.out(atten)

        return atten

class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 定义自注意力层和前馈层
        self.s1 = torch.nn.Sequential(
            torch.nn.LayerNorm(768),
            Atten(),
        )

        self.s2 = torch.nn.Sequential(
            torch.nn.LayerNorm(768),
            torch.nn.Linear(768, 3072),
            torch.nn.GELU(),
            torch.nn.Linear(3072, 768),
        )

    def forward(self, x):
        # 自注意力层和残差连接
        x = self.s1(x) + x
        # 前馈层和残差连接
        return self.s2(x) + x

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 定义卷积层和嵌入层
        self.s1 = torch.nn.Sequential(
            torch.nn.Conv1d(80, 768, kernel_size=3, stride=1, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv1d(768, 768, kernel_size=3, stride=2, padding=1),
            torch.nn.GELU(),
        )

        self.embed = torch.nn.Embedding(1500, 768)

        # 定义多层编码器层和最终的层归一化
        s2 = [Layer() for _ in range(12)]
        s2.append(torch.nn.LayerNorm(768))
        self.s2 = torch.nn.Sequential(*s2)

    def forward(self, x):
        # 输入卷积变换并加上嵌入权重
        x = self.s1(x).permute(0, 2, 1) + self.embed.weight
        # 通过多层编码器层
        return self.s2(x)

def load_encoder(pretrained):
    encoder = Encoder()

    # 加载预训练权重
    encoder.s1[0].load_state_dict(pretrained.conv1.state_dict())
    encoder.s1[2].load_state_dict(pretrained.conv2.state_dict())
    encoder.embed.load_state_dict(pretrained.embed_positions.state_dict())

    for i in range(12):
        encoder.s2[i].s1[1].q.load_state_dict(pretrained.layers[i].self_attn.q_proj.state_dict())
        encoder.s2[i].s1[1].k.load_state_dict(pretrained.layers[i].self_attn.k_proj.state_dict())
        encoder.s2[i].s1[1].v.load_state_dict(pretrained.layers[i].self_attn.v_proj.state_dict())
        encoder.s2[i].s1[1].out.load_state_dict(pretrained.layers[i].self_attn.out_proj.state_dict())

        encoder.s2[i].s1[0].load_state_dict(pretrained.layers[i].self_attn_layer_norm.state_dict())
        encoder.s2[i].s2[0].load_state_dict(pretrained.layers[i].final_layer_norm.state_dict())
        encoder.s2[i].s2[1].load_state_dict(pretrained.layers[i].fc1.state_dict())
        encoder.s2[i].s2[3].load_state_dict(pretrained.layers[i].fc2.state_dict())

    encoder.s2[12].load_state_dict(pretrained.layer_norm.state_dict())

    return encoder
