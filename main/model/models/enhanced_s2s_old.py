import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

#torch
class PositionalEncoding(nn.Module):
    def __init__(self, sequence_length, d_model, **kwargs):
        super(PositionalEncoding,self).__init__(**kwargs)
        self.sequence_length=sequence_length
        self.d_model=d_model
        pos_encoding=self.create_postional_encoding(self.sequence_length, self.d_model)#
        self.register_buffer('pos_encoding', pos_encoding)#

    def create_postional_encoding(self, seq_len, d_model):
        pos_enc = torch.zeros(seq_len, d_model)
        position = torch.arange(0,seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000)/d_model))#
        pos_enc[:,0::2]=torch.sin(position*div_term)
        pos_enc[:,1::2]=torch.cos(position*div_term)
        return pos_enc.unsqueeze(0)
    def forward(self,x):
        return x+self.pos_encoding[:,x.size(1),:]#

#torch 
class EnhancedMultHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        super(EnhancedMultHeadAttention,self).__init__()
        assert d_model % num_heads == 0

        self.d_model=d_model
        self.num_heads = num_heads
        self.depth=d_model//num_heads

        self.wq = nn.Linear(d_model,d_model)
        self.wk = nn.Linear(d_model,d_model)
        self.wv = nn.Linear(d_model,d_model)

        self.dense = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)

        return x.permute(0,2,1,3)
    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = torch.matmul(q, k.transpose(-2,-1))

        dk = torch.tensor(k.shape[-1], dtype=torch.float32)
        scaled_attention_logits = matmul_qk/ torch.sqrt(dk)

        if mask is not None:
            scaled_attention_logits = scaled_attention_logits.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scaled_attention_logits, dim=-1)#
        output = torch.matmul(scaled_attention_logits, v)

        return output, attention_weights
    
    def forward(self,v, k, q, mask=None):
        batch_size=q.size(0)

        q = self.wq(q)
        v = self.wv(v)
        k = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = scaled_attention.permute(0,2,1,3).contiguous()
        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)

        output = self.dense(concat_attention)
        output = self.dropout(output)

        return output, attention_weights

    
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.1):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, values, training = None):
        query_with_time_axis = query.unsqueeze(1)

        score = self.V(torch.tanh(self.W1(values) + self.W2(query_with_time_axis)))
        attention_weights = F.softmax(score, dim=1)
        attention_weights = self.dropout(attention_weights)

        context_vector = attention_weights * values
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector, attention_weights

#torch
#相比tf， torch将encoder和decoder部分封装在了外部
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

        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size = n_hiddensize,
                hidden_size = n_hiddensize,
                num_layers = 1,
                batch_first = True,
                dropout = lstm_dropout if num_encoder_layers > 1 else 0
            ) for _ in range(num_encoder_layers)
        ])

        if use_layer_norm:
            self.norm_layers = nn.ModuleList([nn.LayerNorm(n_hiddensize) for _ in range(num_encoder_layers)])

        self.dropout_layers = nn.ModuleList([nn.Dropout(lstm_dropout) for _ in range(num_encoder_layers)])


    def forward(self, x):
        x = self.input_projection(x)

        if self.use_positional_encoding:
            x = self.pos_encoding(x)

        for i, (lstm, dropout) in enumerate(zip(self.lstm_layers, self.dropout_layers)):
            residual = x
            lstm_out , (h_state, c_state) = lstm(x)

            if self.use_residual and i>0:
                lstm_out = lstm_out+residual
            
            if self.use_layer_norm:
                lstm_out = self.norm_layers[i](lstm_out)

            x = dropout(lstm_out)

        return x, (h_state, c_state)
    
class Decoder(nn.Module):
    def __init__(self, n_labels, n_hiddensize, num_decoder_layers, num_attention_heads,
                 attention_dropout, lstm_dropout, use_layer_norm, use_residual):
        super(Decoder, self).__init__()
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.num_attention_heads = num_attention_heads

        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size = (n_labels+n_hiddensize) if i == 0 else n_hiddensize,
                n_hiddensize = n_hiddensize,
                num_layers = 1,
                batch_first = True,
                dropout = lstm_dropout if num_decoder_layers >1 else 0
            ) for i in range (num_decoder_layers)
        ])

        self.attention = BahdanauAttention(n_hiddensize, attention_dropout)
        if num_attention_heads > 1:
            self.multi_head_attention = EnhancedMultHeadAttention(
                n_hiddensize, num_decoder_layers, attention_dropout
            )

        if self.use_layer_norm:
            self.norm_layers = nn.ModuleList([
                nn.LayerNorm(n_hiddensize) 
                for _ in range (num_decoder_layers)
            ])

        self.output_layer = nn.Linear(n_hiddensize, n_labels)

    def forward(self, decoder_input,encoder_output, hidden, cell):
        output_len = decoder_input.size(1)
        batch_size = decoder_input.size(0)

        current_input = decoder_input[:,0,:]

        decoder_states = [(hidden, cell)] * len(self.lstm_layers)#

        outputs = []

        for t in range(output_len):
            last_layer_hidden = decoder_states[-1][0].squeeze(0)
            context_vector, _ = self.attention(last_layer_hidden, encoder_output)

            if self.num_attention_heads > 1:
                q = last_layer_hidden.unsqueeze(1)
                multi_head_context, _ = self.multi_head_attention(encoder_output, encoder_output, q)
                context_vector = (context_vector + multi_head_context.squeeze(1))/2

            rnn_input = torch.cat((current_input, context_vector), dim=1)

            x = rnn_input.unsqueeze(1)
            for i, lstm in enumerate(self.lstm_layers):
                residual = x
                x, (h,c) = lstm(x, decoder_states[i])#
                decoder_states[i] = (h,c)
                if self.use_residual:
                    x = x + residual
                if self.use_layer_norm:
                    x = self.norm_layers[i](x)

            output = self.output_layer(x.squeeze(1))
            outputs.append(output)
            if t < output_len - 1:
                current_input = decoder_input[:, t+1, :]#??????
            return torch.cat(outputs, dim=1)#dim and axis

        
class EnhancedSeq2Seq(nn.Module):
    def __init__(self, n_features, n_labels, input_len, output_len,
                 n_hiddensize=256, num_encoder_layers=3, num_decoder_layers=3,
                 num_attention_heads=4, attention_dropout=0.1,
                 lstm_dropout=0.2, dense_dropout=0.3,
                 use_layer_norm=True, use_residual=True, use_positional_encoding = True,
                 **kwargs):
        super(EnhancedSeq2Seq, self).__init__()

        if n_hiddensize % num_attention_heads != 0:
            print('n_hiddensize must be divisible by num_attention_heads')

        self.output_len = output_len
        self.n_labels = n_labels
        #self.input_len = input_len don't have this, self. only have two lines above

        self.encoder = Encoder(
            n_features, n_hiddensize, num_encoder_layers, lstm_dropout,
            use_layer_norm, use_residual, use_positional_encoding, input_len
        )
        self.decoder_cell = self._build_decoder_step_cell(
            n_labels, n_hiddensize, num_attention_heads, num_decoder_layers,
            attention_dropout, lstm_dropout, use_layer_norm, use_residual
        )
        print(f'enhanced S2S model')
    def _build_decoder_step_cell(self, n_labels, n_hiddensize, num_attention_heads, num_decoder_layers,
                                 attention_dropout, lstm_dropout, use_layer_norm, use_residual):
        layers = {}
        layers['lstm_layers'] = nn.ModuleList([
            nn.LSTMCell(
                input_size = (n_hiddensize) if i > 0 else (n_hiddensize + n_labels),
                hidden_size = n_hiddensize
            ) for i in range(num_decoder_layers)
        ])

        layers['attention'] = BahdanauAttention(n_hiddensize, attention_dropout)
        if num_attention_heads > 1:
            layers['multi_head_attention'] = EnhancedMultHeadAttention(n_hiddensize, num_attention_heads, attention_dropout)

        if use_layer_norm:
            layers['norm_layers'] = nn.ModuleList([nn.LayerNorm(n_hiddensize) for _ in range(num_decoder_layers)])
        layers['output_layer'] = nn.Linear(n_hiddensize, n_labels)

        # 将配置也存起来方便在forward中访问 
        layers['config'] = nn.Module() # 用一个空的Module来挂载非nn.Module的配置
        layers['config'].use_layer_norm = use_layer_norm
        layers['config'].use_residual = use_residual
        layers['config'].num_attention_heads = num_attention_heads
        
        return nn.ModuleDict(layers)
    
    def forward(self, encoder_input, decoder_input):
        encoder_output, (h, c) = self.encoder(encoder_input)

        batch_size = encoder_input.size(0)

        decoder_states = [(h.squeeze(0), c.squeeze(0))] * len(self.decoder_cell['lstm_layers'])

        outputs = []
        for t in range(self.output_len):
            current_input_label = decoder_input[:, t, :]
            query = current_input_label#decoder_states[-1][0] 为什么不是current_input_label
            context_vector, _ = self.decoder_cell['attention'](query, encoder_output)#

            if self.decoder_cell.config.num_attention_heads > 1:
                q = query.unsqueeze(1)
                mha_context, _ = self.decoder_cell['multi_head_attention'](encoder_output, encoder_output, q)
                context_vector = (mha_context.squeeze(1) + context_vector) / 2

            lstm_input = torch.cat((current_input_label, context_vector), dim=1)

            for i, lstm_cell in enumerate(self.decoder_cell['lstm_layers']):
                residual = lstm_input
                h,c = lstm_cell(lstm_input, decoder_states[i])
                decoder_states[i] = (h,c)

                if self.decoder_cell.config.use_residual:
                    h = h + residual
                if self.decoder_cell.config.use_layer_norm:
                    h = self.decoder_cell['norm_layers'][i](h)

                lstm_input = h

            output = self.decoder_cell['output_layer'](lstm_input)
            outputs.append(output.unsqueeze(1))


            return torch.cat(outputs, dim=1)
        
    @torch.no_grad()
    def predict_autoregressive(self, x, initial_input = 'last_encoder', verbose = 0):
        self.eval()

        encoder_output, (h, c) = self.encoder(x)
        decoder_states = [(h.squeeze(0), c.squeeze(0))] * len(self._build_decoder_step_cell['lstm_layers'])

        batch_size = x.shape[0]

        if initial_input == 'zeros':
            decoder_input = torch.zeros((batch_size, self.output_len, self.n_labels))
        elif initial_input == 'last_encoder':
            last_values = x[:,-1:,:1]
            decoder_input = torch.repeat(last_values, self.output_len, axis=1)

        elif initial_input == 'mean':
            mean_values = torch.mean(x[:,:,:1], axis=1, keepdims=True)
            decoder_input = torch.repeat(mean_values, self.output_len, axis=-1)
        elif initial_input == 'trend':
            decoder_input = self._init_with_trend(x)

        else:
            print('initial method not supported')

        

        predictions = []

        for t in range(self.output_lenut):
            current_input_label = decoder_input[:, t, :]

            query = decoder_states[-1][0]
            context_vector = self.decoder_cell['attention'](query, encoder_output)
            if self.decoder_cell.config.num_attention_heads > 1:
                q = query.unsqueeze(1)
                mha_context, _ = self.decoder_cell['multi_head_attention'](encoder_output,
                                                                           encoder_output,
                                                                           q)
                context_vector = (context_vector + mha_context.squeeze(1)) / 2

            lstm_input = torch.cat((current_input_label, context_vector), dim=1)

            for i, lstm_cell in enumerate(self.decoder_cell['lstm_layers']):
                residual = lstm_input

                h,c = lstm_cell(lstm_input, decoder_states[i])

                decoder_states[i] = (h, c)
                if self.decoder_cell.config.use_residual and i > 0: h = h + residual
                if self.decoder_cell.config.use['use_layer_norm']: h = self.decoder_cell['norm_layers'][i](h)
                lstm_input = h

            current_pred = self.decoder_cell['output_layer'](lstm_input)
            predictions.append(current_pred.unsqueeze(1))

            if t < self.output_len - 1:
                decoder_input[:, t+1, :] = current_pred

            return torch.cat(predictions, dim=1)
                                                                            



    def _init_with_trend(self,x):
        #import numpy as np
        batch_size = x.shape[0]

        n_trend_steps = min(6, self.input_len)

        trend_data = x[:,:n_trend_steps,:1]

        decoder_input = torch.zeros((batch_size, self.output_len, self.n_labels))

        for b in range(batch_size):
            y_vals = trend_data[b,:,0]
            x_vals = torch.arange(n_trend_steps)

            if len(y_vals) > 1:
                slope = torch.polyfit(x_vals, y_vals, 1)[0]
                last_value = y_vals[-1]

                for t in range(self.output_len):
                    pridicted_value = last_value+slope*(t+1)
                    decoder_input[b,t,0] = pridicted_value
            else:
                decoder_input[b,:,0] = y_vals[0]

            return decoder_input    

if __name__ == '__main__':
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


