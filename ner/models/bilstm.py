import torch
import torch.nn as nn 
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size, tag2id):
        '''
        vocab_size: 所有词汇的数量
        emb_size: 词向量的维度
        hidden_size: 隐向量的维数
        out_size: 标注的类别数
        '''
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.bilstm = nn.LSTM(emb_size, hidden_size, batch_first=True,
                             bidirectional=True)
        self.lin = nn.Linear(2*hidden_size, out_size)
        self.tag2id = tag2id
        
    def forward(self, sents_tensor, lengths):
        # [B, L, emb_size]
        emb = self.embedding(sents_tensor)
        packed = pack(emb, lengths, batch_first=True, enforce_sorted=True)
        # 输出的shape: [B, L, hidden_size*2]
        rnn_out, _ = self.bilstm(packed)
        rnn_out, _ = unpack(rnn_out, batch_first=True)
        
        scores = self.lin(rnn_out)  ## [B, L, out_size]
        return scores
    
    def test(self, sents_tensor, lengths):
        '''
        第三个参数不会用到，只是为了和BiLSTM_CRF保持一样的接口
        '''
        logits = self.forward(sents_tensor, lengths)  
        _, batch_tagids = torch.max(logits, dim=2)
        return batch_tagids