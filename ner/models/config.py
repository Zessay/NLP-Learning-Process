# 设置LSTM训练参数
class TrainingConfig:
    batch_size = 64
    # 设置学习率
    lr = 0.001
    epochs = 5 
    print_step = 15
    
class LSTMConfig:
    emb_size = 128  # 词向量的维度
    hidden_size = 128   # LSTM隐向量的维度