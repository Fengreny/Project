import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


# 注意力机制实现 - 这是你可以修改的关键部分
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 创建查询(Q)、键(K)、值(V)和输出的线性层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        计算缩放点积注意力
        
        参数:
            Q: 查询张量 [batch_size, num_heads, seq_len, d_k]
            K: 键张量 [batch_size, num_heads, seq_len, d_k]
            V: 值张量 [batch_size, num_heads, seq_len, d_k]
            mask: 可选掩码张量 [batch_size, 1, 1, seq_len]
            
        返回:
            注意力输出和注意力权重
        """
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用掩码(如果提供)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 应用softmax获取注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 计算输出
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性投影
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # 将张量分割成多头
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 应用注意力机制
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 最终线性层
        output = self.W_o(attn_output)
        
        return output, attn_weights


# 前馈神经网络
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # 注册为缓冲区(不作为模型参数)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# 编码器层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # 自注意力机制
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈神经网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


# 解码器层
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        # 自注意力机制
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 交叉注意力机制
        attn_output, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # 前馈神经网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


# 编码器
class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Encoder, self).__init__()
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


# 解码器
class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Decoder, self).__init__()
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x


# 完整的Transformer模型
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 d_ff=2048, num_layers=6, dropout=0.1, max_seq_length=5000):
        super(Transformer, self).__init__()
        
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout)
        
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)
        
        # 初始化参数
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_mask(self, src, tgt):
        # 创建源序列掩码
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        
        # 创建目标序列掩码
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask.to(tgt.device)
        
        return src_mask, tgt_mask
        
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        # 编码器部分
        src_embedded = self.positional_encoding(self.encoder_embedding(src))
        enc_output = self.encoder(src_embedded, src_mask)
        
        # 解码器部分
        tgt_embedded = self.positional_encoding(self.decoder_embedding(tgt))
        dec_output = self.decoder(tgt_embedded, enc_output, src_mask, tgt_mask)
        
        # 输出层
        output = self.output_linear(dec_output)
        
        return output


# 使用示例
def transformer_example():
    # 定义超参数
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6
    dropout = 0.1
    
    # 创建模型
    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_layers, dropout)
    
    # 示例输入
    src = torch.randint(1, src_vocab_size, (64, 20))  # [batch_size, src_seq_len]
    tgt = torch.randint(1, tgt_vocab_size, (64, 15))  # [batch_size, tgt_seq_len]
    
    # 前向传播
    output = transformer(src, tgt)
    print(f"输出形状: {output.shape}")  # 应该是 [64, 15, tgt_vocab_size]
    
    return transformer


# 如果作为主程序运行
if __name__ == "__main__":
    model = transformer_example()


#关键部分解析：注意力机制
#注意力机制是你可以重点修改的部分，它位于 MultiHeadAttention 类中的 scaled_dot_product_attention 方法。这个方法实现了标准的缩放点积注意力计算：
'''
def scaled_dot_product_attention(self, Q, K, V, mask=None):
    # 计算注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
    
    # 应用掩码(如果提供)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 应用softmax获取注意力权重
    attn_weights = F.softmax(scores, dim=-1)
    attn_weights = self.dropout(attn_weights)
    
    # 计算输出
    output = torch.matmul(attn_weights, V)
    
    return output, attn_weights
    '''


# ###如何修改注意力机制
# 你可以尝试以下几种修改方式：

# 相对位置编码：修改注意力计算，加入相对位置信息

# 稀疏注意力：实现局部注意力或者块稀疏注意力

# 线性注意力：使用核方法近似点积注意力，如 Linformer 或 Performer 中的方法

# 高斯核注意力：使用高斯核函数替代点积

# 多头注意力变体：修改多头的结构或组合方式



# 修改示例：高斯核注意力
# 以下是一个使用高斯核函数替代点积的示例：
'''
def gaussian_kernel_attention(self, Q, K, V, mask=None, sigma=1.0):
    # 计算查询和键之间的欧氏距离
    q_norm = Q.norm(dim=-1, keepdim=True)
    k_norm = K.norm(dim=-1, keepdim=True)
    
    # 计算高斯核 exp(-||q-k||^2 / (2*sigma^2))
    qk_dist = -2 * torch.matmul(Q, K.transpose(-2, -1))
    qk_dist += q_norm.pow(2)
    qk_dist += k_norm.pow(2).transpose(-2, -1)
    scores = torch.exp(-qk_dist / (2 * sigma * sigma))
    
    # 应用掩码(如果提供)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, 0)
    
    # 归一化
    scores_sum = scores.sum(dim=-1, keepdim=True)
    attn_weights = scores / (scores_sum + 1e-9)
    
    # 计算输出
    output = torch.matmul(attn_weights, V)
    
    return output, attn_weights
'''

# 要使用这个修改后的注意力机制，只需在 MultiHeadAttention 类中替换 scaled_dot_product_attention 方法，或者添加一个新方法并在 forward 中调用它。

# 如何测试你的修改
# 创建一个简单的序列到序列任务（如机器翻译或文本摘要）
# 分别使用原始 Transformer 和修改后的版本训练模型
# 比较两个模型在验证集上的性能
# 分析注意力权重的分布和模式变化
# 通过这种方式，你可以直观地了解你的修改对模型性能和行为的影响。

