# import Whisper.decoder
# import torch
# from transformers import WhisperForConditionalGeneration

# pretrained = WhisperForConditionalGeneration.from_pretrained(
#     'openai/whisper-small').model.decoder
# decoder = Whisper.decoder.load_decoder(pretrained)
# def forward_atten(self, x, cache_kv):
#     #前向注意力机制
#     q = self.q(x).reshape(1, 1, 12, 64).transpose(1, 2) * 0.125
#     k = self.k(x).reshape(1, -1, 12, 64).transpose(1, 2)
#     v = self.v(x).reshape(1, -1, 12, 64).transpose(1, 2)

#     if cache_kv:
#         k = torch.cat([cache_kv[0], k], dim=2)
#         v = torch.cat([cache_kv[1], v], dim=2)

#     cache_kv = k, v

#     q = q.reshape(12, -1, 64)
#     k = k.reshape(12, -1, 64)
#     v = v.reshape(12, -1, 64)

#     atten = q.bmm(k.transpose(1, 2)).softmax(dim=-1).bmm(v)
#     atten = atten.reshape(1, 12, 1, 64).transpose(1, 2).reshape(1, 1, 768)
#     atten = self.out(atten)
#     return atten, cache_kv


# def forward_cross_atten(self, x, kv, cache_kv):
#     #前向交叉注意力机制
#     q = self.q(x).reshape(1, 1, 12, 64).transpose(1, 2) * 0.125
#     if cache_kv:
#         k, v = cache_kv
#     else:
#         k = self.k(kv).reshape(1, -1, 12, 64).transpose(1, 2)
#         v = self.v(kv).reshape(1, -1, 12, 64).transpose(1, 2)
#         cache_kv = k, v

#     q = q.reshape(12, -1, 64)
#     k = k.reshape(12, -1, 64)
#     v = v.reshape(12, -1, 64)

#     atten = q.bmm(k.transpose(1, 2)).softmax(dim=-1).bmm(v).reshape(
#         1, 12, 1, 64).transpose(1, 2).reshape(1, 1, 768)

#     atten = self.out(atten)

#     return atten, cache_kv


# def forward_layer(self, x, kv, cache_kv):#
#     #前向传播层
#     next_cache_kv = ()

#     res = x
#     x, _cache = forward_atten(self.atten,
#                               self.norm1(x),
#                               cache_kv=cache_kv[:2] if cache_kv else None)
#     x = x + res
#     next_cache_kv += _cache

#     res = x
#     x, _cache = forward_cross_atten(
#         self.cross_atten,
#         self.norm2(x),
#         kv,
#         cache_kv=cache_kv[2:] if cache_kv else None)
#     x = x + res
#     next_cache_kv += _cache

#     return self.s(x) + x, next_cache_kv


# def forward_decoder(self, x, kv, cache_kv):
#     #解码器前向传播
#     pos_offset = cache_kv[0][0].shape[2] if cache_kv else 0

#     x = self.embed(x) + self.embed_pos.weight[pos_offset:pos_offset +
#                                               x.shape[1]]

#     next_cache_kv = []
#     for i, layer in enumerate(self.layer):
#         x, _cache = forward_layer(layer, x, kv,
#                                   cache_kv[i] if cache_kv else None)
#         next_cache_kv.append(_cache)

#     return self.norm(x), tuple(next_cache_kv)

import Whisper.decoder
import torch
from transformers import WhisperForConditionalGeneration

# 加载预训练的Whisper模型并获取其解码器部分
pretrained = WhisperForConditionalGeneration.from_pretrained('openai/whisper-small').model.decoder
decoder = Whisper.decoder.load_decoder(pretrained)

def forward_atten(self, x, cache_kv):
    # 前向注意力机制
    q = self.q(x).reshape(1, 1, 12, 64).transpose(1, 2) * 0.125  # 计算查询向量并进行重塑和转置
    k = self.k(x).reshape(1, -1, 12, 64).transpose(1, 2)  # 计算键向量并进行重塑和转置
    v = self.v(x).reshape(1, -1, 12, 64).transpose(1, 2)  # 计算值向量并进行重塑和转置

    # 如果缓存键值存在，将其与当前键值向量拼接
    if cache_kv:
        k = torch.cat([cache_kv[0], k], dim=2)
        v = torch.cat([cache_kv[1], v], dim=2)

    cache_kv = k, v  # 更新缓存键值对

    # 重塑查询、键和值向量
    q = q.reshape(12, -1, 64)
    k = k.reshape(12, -1, 64)
    v = v.reshape(12, -1, 64)

    # 计算注意力得分并对值向量进行加权求和
    atten = q.bmm(k.transpose(1, 2)).softmax(dim=-1).bmm(v)
    atten = atten.reshape(1, 12, 1, 64).transpose(1, 2).reshape(1, 1, 768)
    atten = self.out(atten)  # 通过线性层

    return atten, cache_kv  # 返回注意力输出和缓存键值对

def forward_cross_atten(self, x, kv, cache_kv):
    # 前向交叉注意力机制
    q = self.q(x).reshape(1, 1, 12, 64).transpose(1, 2) * 0.125  # 计算查询向量并进行重塑和转置
    if cache_kv:
        k, v = cache_kv  # 使用缓存的键和值向量
    else:
        k = self.k(kv).reshape(1, -1, 12, 64).transpose(1, 2)  # 计算键向量并进行重塑和转置
        v = self.v(kv).reshape(1, -1, 12, 64).transpose(1, 2)  # 计算值向量并进行重塑和转置
        cache_kv = k, v  # 更新缓存键值对

    # 重塑查询、键和值向量
    q = q.reshape(12, -1, 64)
    k = k.reshape(12, -1, 64)
    v = v.reshape(12, -1, 64)

    # 计算注意力得分并对值向量进行加权求和
    atten = q.bmm(k.transpose(1, 2)).softmax(dim=-1).bmm(v).reshape(1, 12, 1, 64).transpose(1, 2).reshape(1, 1, 768)
    atten = self.out(atten)  # 通过线性层

    return atten, cache_kv  # 返回注意力输出和缓存键值对

def forward_layer(self, x, kv, cache_kv):
    # 前向传播一层
    next_cache_kv = ()

    res = x  # 残差连接
    x, _cache = forward_atten(self.atten, self.norm1(x), cache_kv=cache_kv[:2] if cache_kv else None)
    x = x + res  # 加上残差
    next_cache_kv += _cache  # 更新缓存键值对

    res = x  # 残差连接
    x, _cache = forward_cross_atten(self.cross_atten, self.norm2(x), kv, cache_kv=cache_kv[2:] if cache_kv else None)
    x = x + res  # 加上残差
    next_cache_kv += _cache  # 更新缓存键值对

    return self.s(x) + x, next_cache_kv  # 返回层输出和缓存键值对

def forward_decoder(self, x, kv, cache_kv):
    # 解码器前向传播
    pos_offset = cache_kv[0][0].shape[2] if cache_kv else 0

    # 嵌入输入和位置
    x = self.embed(x) + self.embed_pos.weight[pos_offset:pos_offset + x.shape[1]]

    next_cache_kv = []
    for i, layer in enumerate(self.layer):
        x, _cache = forward_layer(layer, x, kv, cache_kv[i] if cache_kv else None)
        next_cache_kv.append(_cache)  # 更新缓存键值对

    return self.norm(x), tuple(next_cache_kv)  # 返回标准化的输出和缓存键值对
