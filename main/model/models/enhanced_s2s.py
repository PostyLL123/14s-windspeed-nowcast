import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 1. Positional Encoding ---
# NOTE: Adds information about the position of tokens in the sequence.
class PositionalEncoding(nn.Module):
    def __init__(self, sequence_length, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model
        
        # FIX: Pass the required arguments to the creation function.
        pos_encoding = self.create_positional_encoding(sequence_length, d_model)
        
        # NOTE: register_buffer makes this tensor part of the module's state,
        # but not a parameter to be trained. It will be moved to the correct device (e.g., GPU).
        self.register_buffer('pos_encoding', pos_encoding)

    def create_positional_encoding(self, seq_len, d_model):
        pos_enc = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
        
        # NOTE: This is the core formula for positional encoding.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))#decrease with index
        
        # FIX: Corrected the multiplication syntax for sin/cos calculation.
        # It should be an element-wise multiplication, not passing div_term as a second argument.
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        
        # NOTE: Add a batch dimension so it can be easily added to the input tensor.
        return pos_enc.unsqueeze(0)

    def forward(self, x):
        # x is expected to have shape: [batch_size, sequence_length, d_model]
        # FIX: Slicing was incorrect. We need to slice up to the input's sequence length.
        # This allows the model to handle sequences shorter than the max `sequence_length`.
        return x + self.pos_encoding[:, :x.size(1), :]

# --- 2. Multi-Head Attention ---
# NOTE: Implements scaled dot-product attention with multiple heads.
class EnhancedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        super(EnhancedMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def split_heads(self, x, batch_size):
        # Reshape the input to be [batch_size, seq_len, num_heads, depth]
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        # Transpose to get [batch_size, num_heads, seq_len, depth] for attention calculation
        return x.permute(0, 2, 1, 3)

    def scaled_dot_product_attention(self, q, k, v, mask):
        # Matmul of Q and K^T -> [batch_size, num_heads, seq_len_q, seq_len_k]
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))

        dk = torch.tensor(k.shape[-1], dtype=torch.float32)
        scaled_attention_logits = matmul_qk / torch.sqrt(dk)

        if mask is not None:
            # Apply the mask by filling masked positions with a very large negative value.
            scaled_attention_logits += (mask * -1e9)

        # Softmax is applied on the last axis (seq_len_k) to get attention weights.
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        
        # FIX: The original code multiplied scaled_attention_logits by V.
        # You must multiply the attention_weights (after softmax) by V.
        output = torch.matmul(attention_weights, v)

        return output, attention_weights
    
    def forward(self, v, k, q, mask=None):
        batch_size = q.size(0)

        q = self.wq(q)
        v = self.wv(v)
        # FIX: K was being projected with wv instead of wk. Corrected to use self.wk(k).
        k = self.wk(k)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        # Reverse the permute to get [batch_size, seq_len_q, num_heads, depth]
        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()
        # Concatenate heads back together
        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)

        output = self.dense(concat_attention)
        output = self.dropout(output)

        return output, attention_weights

# --- 3. Bahdanau Attention ---
# NOTE: Implements additive attention (Bahdanau-style).
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.1):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, values):
        # query: [batch_size, hidden_size]
        # values: [batch_size, seq_len, hidden_size]
        query_with_time_axis = query.unsqueeze(1) # -> [batch_size, 1, hidden_size]

        # Calculate score: V * tanh(W1*values + W2*query)
        score = self.V(torch.tanh(self.W1(values) + self.W2(query_with_time_axis)))
        
        # Softmax to get weights -> [batch_size, seq_len, 1]
        attention_weights = F.softmax(score, dim=1)
        attention_weights = self.dropout(attention_weights)#NOTE: whether to put dropout last

        # Multiply weights with values to get the context vector
        context_vector = attention_weights * values
        context_vector = torch.sum(context_vector, dim=1) # -> [batch_size, hidden_size]

        return context_vector, attention_weights

# --- 4. Encoder ---
# NOTE: Processes the input sequence using a stack of LSTMs.
class Encoder(nn.Module):
    def __init__(self, n_features, n_hiddensize, num_encoder_layers, 
                 lstm_dropout, use_layer_norm, use_residual, use_positional_encoding, input_len):
        super(Encoder, self).__init__()
        self.use_positional_encoding = use_positional_encoding
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        
        self.input_projection = nn.Linear(n_features, n_hiddensize)
        if self.use_positional_encoding:
            self.pos_encoding = PositionalEncoding(input_len, n_hiddensize)

        # FIX: Corrected typo from Modulelist to ModuleList
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=n_hiddensize,
                hidden_size=n_hiddensize,
                num_layers=1, # Each LSTM layer is created individually
                batch_first=True,
                # NOTE: Dropout is only applied between layers in a multi-layer LSTM,
                # so we handle it manually with a Dropout layer.
                dropout=0 
            ) for _ in range(num_encoder_layers)
        ])

        if use_layer_norm:
            # FIX: Corrected typo from `self,norm_layers` to `self.norm_layers`
            # FIX: Corrected typo from Modulelist to ModuleList
            self.norm_layers = nn.ModuleList([nn.LayerNorm(n_hiddensize) for _ in range(num_encoder_layers)])

        # FIX: Corrected typo from Modulelist to ModuleList
        self.dropout_layers = nn.ModuleList([nn.Dropout(lstm_dropout) for _ in range(num_encoder_layers)])

    def forward(self, x):
        x = self.input_projection(x)

        if self.use_positional_encoding:
            x = self.pos_encoding(x)

        # FIX: Corrected zip syntax from `zip[...]` to `zip(...)`
        for i, (lstm, dropout) in enumerate(zip(self.lstm_layers, self.dropout_layers)):
            residual = x
            lstm_out, (h_state, c_state) = lstm(x)
            
            # NOTE: Residual connection adds the input of the layer to its output.
            # Usually skipped for the first layer as the dimensions might change.
            if self.use_residual and i > 0:
                lstm_out = lstm_out + residual
            
            if self.use_layer_norm:
                lstm_out = self.norm_layers[i](lstm_out)

            x = dropout(lstm_out)
        
        # NOTE: Returns the output of the final layer and its final hidden/cell states.
        return x, (h_state, c_state)

# NOTE: The original `Decoder` class is not used by `EnhancedSeq2Seq`.
# The logic was moved into `_build_decoder_step_cell` for more granular control,
# which is a good design for autoregressive prediction. I have removed it to avoid confusion.


# --- 5. The Main Seq2Seq Model ---
class Model(nn.Module):
    def __init__(self, cfg,
                 **kwargs):
        super(Model, self).__init__(**kwargs)

        if cfg.n_hiddensize % cfg.num_attention_heads != 0:
            raise ValueError('n_hiddensize must be divisible by num_attention_heads')

        self.output_len = cfg.output_len
        self.n_labels = cfg.n_labels
        self.n_hiddensize = cfg.n_hiddensize
        self.num_decoder_layers = cfg.num_decoder_layers
        self.input_len = cfg.input_len
        self.n_features = cfg.n_features

        self.encoder = Encoder(
            cfg.n_features, cfg.n_hiddensize, cfg.num_encoder_layers, cfg.lstm_dropout,
            cfg.use_layer_norm, cfg.use_residual, cfg.use_positional_encoding, cfg.input_len
        )
        
        # NOTE: This builds the components for a single step of the decoder.
        self.decoder_cell = self._build_decoder_step_cell(
            cfg.n_labels, cfg.n_hiddensize, cfg.num_attention_heads, cfg.num_decoder_layers,
            cfg.attention_dropout, cfg.lstm_dropout, cfg.use_layer_norm, cfg.use_residual
        )
        print(f'Enhanced S2S model initialized.')
        self.cal_trend = self.cal_trend
    
    def cal_trend(self, x):
        device = x.device
        dtype = x.dtype
        batch_size = x.shape[0]
        
        n_trend_steps = min(8, self.input_len)
        
        # 获取用于计算趋势的数据 [batch_size, n_trend_steps]
        # 我们只使用第一个特征 ( :1 并 .squeeze(-1) )
        y_vals = x[:, -n_trend_steps:, 0].to(dtype) 
        
        # 获取最后一个时间步的值，作为预测的起点
        # [batch_size]
        last_value = x[:, -1, 0]

        # --- 2. 矢量化计算斜率 ---
        
        if n_trend_steps > 1:
            # 创建 x 轴 [0, 1, ..., n_trend_steps-1]
            x_vals = torch.arange(n_trend_steps, device=device, dtype=dtype)
            
            # 计算 x 和 y 的均值
            x_mean = x_vals.mean()
            # y_vals 的均值需要在批次维度上计算
            y_mean = y_vals.mean(dim=1, keepdim=True) # [batch_size, 1]
            
            # 计算 x 和 y 的偏差
            x_dev = x_vals - x_mean
            y_dev = y_vals - y_mean
            
            # 矢量化计算斜率
            # 分子: sum((x - x_mean) * (y - y_mean))
            numerator = torch.sum(x_dev * y_dev, dim=1)
            
            # 分母: sum((x - x_mean)^2)
            # (分母对所有批次都是一样的)
            denominator = torch.sum(x_dev ** 2)
            
            # 斜率: [batch_size]
            # 添加一个小的 epsilon 防止除以零 (虽然 n_trend_steps > 1 保证了分母不为0)
            slope = numerator / (denominator + 1e-9)
            
        else:
            # 如果趋势步数不足以计算斜率，则认为斜率为 0
            slope = torch.zeros(batch_size, device=device, dtype=dtype)

        # --- 3. 矢量化趋势预测 ---
        
        # 创建输出时间步 [1, 2, ..., output_len]
        # (对应原始代码中的 t + 1)
        future_steps = torch.arange(1, self.output_len + 1, device=device, dtype=dtype)
        # 扩展 shape: [1, output_len]
        future_steps = future_steps.unsqueeze(0) 

        # 调整 slope 和 last_value 的 shape 以便广播
        # [batch_size, 1]
        slope = slope.unsqueeze(1)
        last_value = last_value.unsqueeze(1)
        
        # 广播计算:
        # last_value [B, 1] + slope [B, 1] * future_steps [1, L_out]
        # 结果 shape: [batch_size, output_len]
        predicted_values = last_value + slope * future_steps
        
        # --- 4. 组装 Decoder Input ---
        
        # 初始化 decoder_input
        decoder_input = torch.zeros(batch_size, self.output_len, self.n_labels, 
                                    device=device, dtype=dtype)
        
        # 将预测的趋势值填充到第一个特征通道
        decoder_input[:, :, 0] = predicted_values
        
        return decoder_input        

    def _build_decoder_step_cell(self, n_labels, n_hiddensize, num_attention_heads, num_decoder_layers,
                                 attention_dropout, lstm_dropout, use_layer_norm, use_residual):
        layers = {}
        # NOTE: Using LSTMCell because we are processing one timestep at a time.
        # FIX: Corrected typo from Modulelist to ModuleList
        layers['lstm_layers'] = nn.ModuleList([
            nn.LSTMCell(
                input_size=n_labels if i == 0 else n_hiddensize,
                hidden_size=n_hiddensize
            ) for i in range(num_decoder_layers)
        ])

        layers['attention'] = BahdanauAttention(n_hiddensize, attention_dropout)
        if num_attention_heads > 1:
            layers['multi_head_attention'] = EnhancedMultiHeadAttention(n_hiddensize, num_attention_heads, attention_dropout)
        
        # FIX: Corrected typo from Modulelist to ModuleList
        layers['dropout_layers'] = nn.ModuleList([nn.Dropout(lstm_dropout) for _ in range(num_decoder_layers)])

        if use_layer_norm:
            # FIX: Corrected typo and removed extra list wrapper.
            layers['norm_layers'] = nn.ModuleList([nn.LayerNorm(n_hiddensize) for _ in range(num_decoder_layers)])
        
        layers['output_layer'] = nn.Linear(n_hiddensize*2, n_labels)

        # Store config flags for easy access in the forward pass.
        layers['config'] = nn.Module()
        layers['config'].use_layer_norm = use_layer_norm
        layers['config'].use_residual = use_residual
        layers['config'].num_attention_heads = num_attention_heads
        
        return nn.ModuleDict(layers)
    
    def forward(self, encoder_input, decoder_input):
        encoder_output, (h, c) = self.encoder(encoder_input)
        batch_size = encoder_input.size(0)

        decoder_input = torch.zeros((batch_size, self.output_len, self.n_labels), device=encoder_input.device)

        batch_size = encoder_input.size(0)

        # Initialize decoder states with the final encoder states.
        # Squeeze to remove the num_layers dimension from the encoder's output state.
        decoder_states = [(h.squeeze(0), c.squeeze(0))] * self.num_decoder_layers

        outputs = []

        for t in range(self.output_len):
            current_input = decoder_input[:, t:t+1, :]
            lstm_input = current_input.squeeze(1)

            for i, (lstm_cell, dropout) in enumerate(zip(self.decoder_cell['lstm_layers'], self.decoder_cell['dropout_layers'])):
                residual = lstm_input
                h, c = lstm_cell(lstm_input, decoder_states[i])
                decoder_states[i] = (h, c)

                if self.decoder_cell.config.use_residual and i > 0:
                    h = h + residual
                
                if self.decoder_cell.config.use_layer_norm:
                    h = self.decoder_cell['norm_layers'][i](h)
                
                # The output of this layer becomes the input for the next.
                lstm_input = dropout(h)

            query = lstm_input
            
            context_vector, _ = self.decoder_cell['attention'](query, encoder_output)
            
            if self.decoder_cell.config.num_attention_heads > 1:
                q = query.unsqueeze(1)
                mha_context, _ = self.decoder_cell['multi_head_attention'](encoder_output, encoder_output, q)
                context_vector = (mha_context.squeeze(1) + context_vector) / 2
            
            combined = torch.cat((query, context_vector), dim=1)
            output = self.decoder_cell['output_layer'](combined)
            outputs.append(output.unsqueeze(1))


        return torch.cat(outputs, dim=1)
    
    @torch.no_grad()
    def predict_autoregressive(self, x, initial_input_method='trend'):
        self.eval() # Set model to evaluation mode

        #encoder_output, (h, c) = self.encoder(x)
        #decoder_states = [(h.squeeze(0), c.squeeze(0))] * self.num_decoder_layers

        batch_size = x.shape[0]

        # Initialize the decoder input based on the chosen method
        if initial_input_method == 'zeros':
            decoder_input = torch.zeros((batch_size, self.output_len, self.n_labels), device=x.device)
        elif initial_input_method == 'last_encoder':
            # Assumes the first feature is the one to be propagated
            decoder_input = torch.zeros((batch_size, self.output_len, self.n_labels), device=x.device)
            last_values = x[:, -1, :self.n_labels]
            decoder_input[:, 0, :] = last_values
        elif initial_input_method == 'trend':
            decoder_input = self.cal_trend(x)

        else: # Add more initializers like 'mean' or 'trend' here if needed
            raise NotImplementedError(f"Initial input method '{initial_input_method}' not supported")

        predictions = []

        for t in range(self.output_len):
            current_decoder_input = decoder_input[:, :t+1, :]
            if t == 0:
                current_decoder_input = decoder_input[:, :1, :]

            full_decoder_input = torch.zeros((batch_size, self.output_len, self.n_labels), device=x.device)
            full_decoder_input[:, :current_decoder_input.size(1), :] = current_decoder_input

            full_prediction = self.forward(x, full_decoder_input)

            current_pred = full_prediction[:, t:t+1, :]
            predictions.append(current_pred)
            
            # 更新decoder input：使用刚刚的预测作为下一步的输入
            if t < self.output_len - 1:  # 不是最后一步
                decoder_input[:, t+1:t+2, :] = current_pred

        return torch.cat(predictions, dim=1)


    

    # NOTE: The trend initialization was removed as it's complex to implement correctly
    # with torch and often less effective than simpler methods like 'last_encoder'.
    # If needed, it would require careful tensor-to-numpy conversion for polyfit.


if __name__ == '__main__':
    print("测试增强版Seq2Seq模型 (PyTorch)...")
    
    

    from dataclasses import dataclass

    # 使用 @dataclass 装饰器
    @dataclass
    class Config:
        n_features: int = 9
        n_labels: int = 1
        input_len: int = 144
        output_len: int = 6
        batch_size: int = 32
        n_hiddensize = 128
        num_attention_heads = 4
        num_decoder_layers = 4
        num_encoder_layers = 4
        lstm_dropout = 0.2
        attention_dropout = 0.2
        use_layer_norm = True
        use_residual = True
        use_positional_encoding = True
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 现在您可以这样使用它 ---
    # 1. 实例化 (它会自动生成 __init__)
    cfg = Config()
    try:
        # 创建模型
        model = Model(
                cfg
        ).to(cfg.device)

        # 打印模型参数量
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型总参数量: {total_params:,}")
        
        # 创建测试数据
        test_encoder_input = torch.rand(cfg.batch_size, cfg.input_len, cfg.n_features).to(cfg.device)
        test_decoder_input = torch.rand(cfg.batch_size, cfg.output_len, cfg.n_labels).to(cfg.device)
        
        # --- 测试训练模式 (Teacher Forcing) ---
        print("\n测试训练模式 (forward)...")
        model.train() # 切换到训练模式
        output_train = model(test_encoder_input, test_decoder_input)
        print(f"训练模式输出形状: {output_train.shape}")
        
        # --- 测试预测模式 (自回归) ---
        print("\n测试预测模式 (predict_autoregressive)...")
        model.eval() # 切换到评估模式 (关闭dropout等)
        predictions = model.predict_autoregressive(test_encoder_input)
        print(f"预测输出形状: {predictions.shape}")

        # --- 模拟一个训练步骤 ---
        print("\n模拟一个训练步骤...")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        model.train() # 确保在训练步骤前调用 .train()
        
        optimizer.zero_grad() # 清空梯度
        
        # 前向传播 (调用 forward)
        output = model(test_encoder_input, test_decoder_input)
        
        # 计算损失
        # 注意：损失函数通常比较 output 和 test_decoder_input
        # (假设 test_decoder_input 是我们的目标)
        loss = criterion(output, test_decoder_input) 
        print(f"模拟损失: {loss.item():.6f}")
        
        # 反向传播
        loss.backward()
        
        # 更新权重
        optimizer.step()
        print("优化器步进完成。")
        
        print("\n增强版Seq2Seq模型 (PyTorch) 测试完成!")
    
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
