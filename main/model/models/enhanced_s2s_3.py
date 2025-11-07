import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class PositionalEncoding(nn.Module):
    """
    位置编码层：为时间序列添加位置信息
    
    设计原理：
    - 时间序列中，每个时间步的位置信息非常重要
    - LSTM虽然能处理序列，但位置编码能帮助模型更好地理解时间关系
    - 使用正弦和余弦函数生成位置编码，这样不同位置的编码具有数学规律性
    
    作用：
    1. 让模型明确知道每个时间步在序列中的位置
    2. 帮助注意力机制更好地关注不同时间步的关系
    3. 提供额外的时间信息，补充原始特征
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        初始化位置编码层
        
        参数:
            d_model: 特征维度（与隐藏层维度一致）
            max_len: 序列最大长度
            dropout: Dropout比例
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算除法项: 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 偶数维度用sin，奇数维度用cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 增加一个batch维度，并注册为buffer
        # buffer是模型的一部分，但不被视为参数，不会被优化器更新
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        将位置编码添加到输入特征中
        
        操作：x + pos_encoding
        为什么是相加而不是拼接？
        1. 相加保持了原有的特征维度
        2. 让模型能同时学习原始特征和位置信息
        3. 计算更高效
        """
        # x 的形状: [batch_size, seq_len, d_model]
        # 截取与输入序列长度匹配的位置编码
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class EnhancedMultiHeadAttention(nn.Module):
    """
    增强的多头注意力机制
    
    设计原理：
    - 多头注意力允许模型同时关注不同类型的信息
    - 每个头关注不同的特征子空间，然后合并结果
    - 比单头注意力更能捕捉复杂的时间依赖关系
    
    作用：
    1. 让模型能够同时关注多个时间步的信息
    2. 不同的头可以学习不同的注意力模式（如短期vs长期）
    3. 提高模型的表达能力和泛化性能
    """
    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        """
        初始化多头注意力层
        
        参数:
            d_model: 特征维度（必须能被num_heads整除）
            num_heads: 注意力头的数量
            dropout_rate: Dropout比例，防止过拟合
        """
        super(EnhancedMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads # 每个头的维度

        # 定义Query、Key、Value的投影矩阵
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        # 最终的输出投影层
        self.dense = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def split_heads(self, x, batch_size):
        """
        将特征维度分割成多个头
        
        形状变化：[batch_size, seq_len, d_model] -> [batch_size, num_heads, seq_len, depth]
        """
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, v, k, q, mask=None):
        """
        多头注意力的前向传播
        
        参数:
            q: Query张量 [batch_size, seq_len_q, d_model]
            k: Key张量 [batch_size, seq_len_k, d_model]  
            v: Value张量 [batch_size, seq_len_v, d_model]
            mask: 掩码，用于遮蔽某些位置
        """
        batch_size = q.size(0)

        # 步骤1：通过投影矩阵生成Q、K、V
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # 步骤2：分割成多个头
        q = self.split_heads(q, batch_size) # [batch_size, num_heads, seq_len_q, depth]
        k = self.split_heads(k, batch_size) # [batch_size, num_heads, seq_len_k, depth]
        v = self.split_heads(v, batch_size) # [batch_size, num_heads, seq_len_v, depth]

        # 步骤3：计算缩放点积注意力
        # matmul_qk: [batch_size, num_heads, seq_len_q, seq_len_k]
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        dk = torch.tensor(k.size(-1), dtype=torch.float32)
        scaled_attention_logits = matmul_qk / torch.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = F.softmax(scaled_attention_logits, dim=-1) # [batch_size, num_heads, seq_len_q, seq_len_k]
        
        output = torch.matmul(attention_weights, v) # [batch_size, num_heads, seq_len_q, depth]
        
        # 步骤4：重新组合多个头的结果
        output = output.permute(0, 2, 1, 3).contiguous()
        concat_attention = output.view(batch_size, -1, self.d_model)

        # 步骤5：最终投影和dropout
        output = self.dense(concat_attention)
        output = self.dropout(output)

        return output, attention_weights

class BahdanauAttention(nn.Module):
    """
    改进的Bahdanau注意力机制 (Additive Attention)
    
    设计原理：
    - Bahdanau注意力使用加法而不是点积
    - 通过学习一个前馈网络来计算注意力权重
    - 特别适合处理编码器-解码器之间的对齐关系
    """
    def __init__(self, units, dropout_rate=0.1):
        """
        初始化Bahdanau注意力层
        
        参数:
            units: 注意力机制的隐藏单元数
            dropout_rate: Dropout比例
        """
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(units, units) # 用于编码器隐藏状态的投影
        self.W2 = nn.Linear(units, units) # 用于解码器隐藏状态的投影
        self.V = nn.Linear(units, 1)      # 将tanh激活后的结果投影到标量
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, values):
        """
        计算Bahdanau注意力
        
        参数:
            query: 解码器的当前隐藏状态 [batch_size, hidden_size]
            values: 编码器的所有隐藏状态 [batch_size, seq_len, hidden_size]
        """
        # 扩展query维度以便与values广播
        # query: [batch_size, hidden_size] -> [batch_size, 1, hidden_size]
        query_with_time_axis = query.unsqueeze(1)
        
        # score = V * tanh(W1*encoder + W2*decoder)
        score = self.V(torch.tanh(self.W1(values) + self.W2(query_with_time_axis)))
        
        # attention_weights: [batch_size, seq_len, 1]
        attention_weights = F.softmax(score, dim=1)
        attention_weights = self.dropout(attention_weights)
        
        # context_vector: [batch_size, hidden_size]
        context_vector = torch.sum(attention_weights * values, dim=1)
        
        return context_vector, attention_weights.squeeze(-1)

class Encoder(nn.Module):
    """编码器模块"""
    def __init__(self, n_features, n_hiddensize, num_encoder_layers,
                 lstm_dropout, use_layer_norm, use_residual, use_positional_encoding):
        super(Encoder, self).__init__()
        self.n_hiddensize = n_hiddensize
        self.num_encoder_layers = num_encoder_layers
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.use_positional_encoding = use_positional_encoding

        self.input_projection = nn.Linear(n_features, n_hiddensize)
        if self.use_positional_encoding:
            self.pos_encoding = PositionalEncoding(n_hiddensize)
        
        self.lstm_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        for i in range(num_encoder_layers):
            self.lstm_layers.append(nn.LSTM(
                input_size=n_hiddensize,
                hidden_size=n_hiddensize,
                num_layers=1, # 每个层独立创建
                batch_first=True,
                dropout=lstm_dropout if num_encoder_layers > 1 else 0
            ))
            if use_layer_norm:
                self.norm_layers.append(nn.LayerNorm(n_hiddensize))
            self.dropout_layers.append(nn.Dropout(lstm_dropout))

    def forward(self, x):
        """
        编码器前向传播
        参数:
            x: 原始输入序列 [batch_size, input_len, n_features]
        返回:
            encoder_output: 编码器输出序列 [batch_size, input_len, n_hiddensize]
            encoder_states: 最终隐藏状态元组 (h_n, c_n)
        """
        x = self.input_projection(x)
        if self.use_positional_encoding:
            x = self.pos_encoding(x)

        hidden_states = None
        
        for i in range(self.num_encoder_layers):
            residual = x
            x, (h, c) = self.lstm_layers[i](x)
            
            if self.use_residual and i > 0:
                x = x + residual
            
            if self.use_layer_norm:
                x = self.norm_layers[i](x)
                
            x = self.dropout_layers[i](x)
            
            # 最后一层的状态作为解码器初始状态
            if i == self.num_encoder_layers - 1:
                hidden_states = (h, c)

        return x, hidden_states

class Decoder(nn.Module):
    """解码器模块"""
    def __init__(self, n_labels, n_hiddensize, num_decoder_layers,
                 num_attention_heads, attention_dropout, lstm_dropout,
                 use_layer_norm, use_residual):
        super(Decoder, self).__init__()
        self.n_hiddensize = n_hiddensize
        self.n_labels = n_labels
        self.num_decoder_layers = num_decoder_layers
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual

        self.lstm_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for i in range(num_decoder_layers):
            # 第一层输入是 (n_labels + n_hiddensize)
            input_dim = n_hiddensize if i > 0 else (n_labels + n_hiddensize)
            self.lstm_layers.append(nn.LSTM(
                input_size=input_dim,
                hidden_size=n_hiddensize,
                num_layers=1,
                batch_first=True
            ))
            if use_layer_norm:
                self.norm_layers.append(nn.LayerNorm(n_hiddensize))

        self.attention = BahdanauAttention(n_hiddensize, attention_dropout)
        if num_attention_heads > 1:
            self.multi_head_attention = EnhancedMultiHeadAttention(
                n_hiddensize, num_attention_heads, attention_dropout)
        else:
            self.multi_head_attention = None

        self.output_layer = nn.Linear(n_hiddensize * 2, n_labels) # LSTM输出 + context
        self.dropout = nn.Dropout(lstm_dropout)

    def forward(self, x, hidden_state, encoder_output):
        """
        解码器单步前向传播
        参数:
            x: 当前时间步输入 [batch_size, 1, n_labels]
            hidden_state: 上一时间步的隐藏状态元组 (h, c)
            encoder_output: 编码器所有输出 [batch_size, input_len, n_hiddensize]
        返回:
            output: 当前时间步预测 [batch_size, n_labels]
            hidden_state: 当前时间步隐藏状态
            attention_weights: 注意力权重
        """
        # 注意力机制
        # query是上一层的隐藏状态 h_state
        # 在多层LSTM中，通常使用最后一层的隐藏状态
        last_hidden = hidden_state[0][-1] # [batch_size, n_hiddensize]
        context_vector, attention_weights = self.attention(last_hidden, encoder_output)
        
        if self.multi_head_attention:
            multi_head_context, _ = self.multi_head_attention(
                v=encoder_output,
                k=encoder_output,
                q=last_hidden.unsqueeze(1)
            )
            context_vector = (context_vector + multi_head_context.squeeze(1)) / 2

        # context_vector: [batch_size, n_hiddensize]
        # x: [batch_size, 1, n_labels]
        # 拼接输入和上下文向量
        rnn_input = torch.cat([x.squeeze(1), context_vector], dim=1) # [batch_size, n_labels + n_hiddensize]
        
        # 逐层通过LSTM
        current_h, current_c = [], []
        
        # PyTorch LSTM的hidden state需要是 [num_layers, batch_size, hidden_size]
        for i in range(self.num_decoder_layers):
            # 准备每层的输入和状态
            input_for_layer = rnn_input if i == 0 else self.dropout(lstm_output)
            state_for_layer = (hidden_state[0][i:i+1], hidden_state[1][i:i+1])
            
            lstm_output, (h, c) = self.lstm_layers[i](input_for_layer.unsqueeze(1), state_for_layer)
            lstm_output = lstm_output.squeeze(1)

            if self.use_layer_norm:
                lstm_output = self.norm_layers[i](lstm_output)
            
            current_h.append(h)
            current_c.append(c)

        hidden_state = (torch.cat(current_h, dim=0), torch.cat(current_c, dim=0))
        
        # 最终输出
        output = self.output_layer(torch.cat([lstm_output, context_vector], dim=1))
        
        return output, hidden_state, attention_weights

class EnhancedSeq2Seq(nn.Module):
    """
    增强版Seq2Seq模型，包含多项改进
    """
    def __init__(self, n_features, n_labels, input_len, output_len,
                 n_hiddensize=256, num_encoder_layers=3, num_decoder_layers=3,
                 num_attention_heads=8, attention_dropout=0.1,
                 lstm_dropout=0.2, dense_dropout=0.3, # dense_dropout is not used in this structure
                 use_layer_norm=True, use_residual=True, use_positional_encoding=True,
                 teacher_forcing_ratio=0.5):
        super(EnhancedSeq2Seq, self).__init__()
        
        self.encoder = Encoder(n_features, n_hiddensize, num_encoder_layers, lstm_dropout, 
                               use_layer_norm, use_residual, use_positional_encoding)
        
        self.decoder = Decoder(n_labels, n_hiddensize, num_decoder_layers, num_attention_heads, 
                               attention_dropout, lstm_dropout, use_layer_norm, use_residual)
                               
        self.output_len = output_len
        self.n_labels = n_labels
        self.teacher_forcing_ratio = teacher_forcing_ratio

        print(f"创建增强版Seq2Seq模型 (PyTorch):")
        print(f"  - 编码器层数: {num_encoder_layers}")
        print(f"  - 解码器层数: {num_decoder_layers}")
        print(f"  - 注意力头数: {num_attention_heads}")
        print(f"  - 隐藏层大小: {n_hiddensize}")
        print(f"  - 层归一化: {use_layer_norm}")
        print(f"  - 残差连接: {use_residual}")
        print(f"  - 位置编码: {use_positional_encoding}")

    def forward(self, encoder_input, decoder_input=None):
        """
        模型前向传播
        参数:
            encoder_input: [batch_size, input_len, n_features]
            decoder_input: [batch_size, output_len, n_labels] (用于teacher forcing)
        """
        batch_size = encoder_input.size(0)
        device = encoder_input.device

        encoder_output, hidden_state = self.encoder(encoder_input)

        outputs = torch.zeros(batch_size, self.output_len, self.n_labels).to(device)
        
        # 解码器初始输入
        current_input = torch.zeros(batch_size, 1, self.n_labels).to(device)

        for t in range(self.output_len):
            output, hidden_state, _ = self.decoder(current_input, hidden_state, encoder_output)
            outputs[:, t, :] = output

            # 决定是否使用teacher forcing
            use_teacher_forcing = torch.rand(1).item() < self.teacher_forcing_ratio
            
            if self.training and use_teacher_forcing and decoder_input is not None:
                current_input = decoder_input[:, t, :].unsqueeze(1)
            else:
                # 使用自己的预测作为下一步的输入
                current_input = output.unsqueeze(1)
        
        return outputs

    def predict_autoregressive(self, encoder_input):
        """
        自回归预测
        """
        self.eval() # 设置为评估模式
        with torch.no_grad():
            batch_size = encoder_input.size(0)
            device = encoder_input.device

            encoder_output, hidden_state = self.encoder(encoder_input)

            outputs = torch.zeros(batch_size, self.output_len, self.n_labels).to(device)
            current_input = torch.zeros(batch_size, 1, self.n_labels).to(device)

            for t in range(self.output_len):
                output, hidden_state, _ = self.decoder(current_input, hidden_state, encoder_output)
                outputs[:, t, :] = output
                current_input = output.unsqueeze(1) # 使用预测作为下一步输入
        
        self.train() # 恢复训练模式
        return outputs

# === 测试代码 ===
if __name__ == "__main__":
    print("测试增强版Seq2Seq模型 (PyTorch)...")
    
    # 测试参数
    n_features = 9
    n_labels = 1
    input_len = 144
    output_len = 6
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    try:
        # 创建模型
        model = EnhancedSeq2Seq(
            n_features=n_features,
            n_labels=n_labels,
            input_len=input_len,
            output_len=output_len,
            n_hiddensize=128,
            num_encoder_layers=3,
            num_decoder_layers=2,
            num_attention_heads=4,
            use_layer_norm=True,
            use_residual=True,
            teacher_forcing_ratio=0.5
        ).to(device)

        # 打印模型参数量
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型总参数量: {total_params:,}")
        
        # 创建测试数据
        test_encoder_input = torch.rand(batch_size, input_len, n_features).to(device)
        test_decoder_input = torch.rand(batch_size, output_len, n_labels).to(device)
        
        # --- 测试训练模式 (Teacher Forcing) ---
        print("\n测试训练模式...")
        model.train()
        output_train = model(test_encoder_input, test_decoder_input)
        print(f"训练模式输出形状: {output_train.shape}")
        
        # --- 测试预测模式 (自回归) ---
        print("\n测试预测模式...")
        predictions = model.predict_autoregressive(test_encoder_input)
        print(f"预测输出形状: {predictions.shape}")

        # --- 模拟一个训练步骤 ---
        print("\n模拟一个训练步骤...")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        optimizer.zero_grad() # 清空梯度
        
        # 前向传播
        output = model(test_encoder_input, test_decoder_input)
        
        # 计算损失
        loss = criterion(output, test_decoder_input)
        print(f"模拟损失: {loss.item():.6f}")
        
        # 反向传播
        loss.backward()
        
        # 更新权重
        optimizer.step()
        print("优化器步进完成。")
        
        print("\n增强版Seq2Seq模型 (PyTorch) 测试完成!")
        
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")
        raise
