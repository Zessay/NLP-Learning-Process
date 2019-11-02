import torch
import torch.nn as nn 
from itertools import zip_longest

from .bilstm import BiLSTM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size, tag2id):
        '''
        vocab_size: 词表大小
        emb_size: 词向量大小
        hidden_size: 隐向量的维度
        out_size: 标注的种类
        '''
        super(BiLSTM_CRF, self).__init__()
        self.bilstm = BiLSTM(vocab_size, emb_size, hidden_size, out_size, tag2id)
        
        # CRF实际上就是多学习了一个转移矩阵
        # [out_size, out_size]初始化为均匀分布
        self.transition = nn.Parameter(torch.ones(out_size, out_size) * 1/out_size)
        self.tag2id = tag2id
        
        
    def forward(self, sents_tensor, lengths):
        # 把biLSTM的输出当做是发射概率矩阵，[B, L, out_size]
        ## 这个步骤打破了最大熵马尔科夫模型观测独立性假设
        emission = self.bilstm(sents_tensor, lengths)
        
        # 计算CRF scores，这个scores的大小为[B, L, out_size, out_size]
        # 也就是每个字对应一个[out_size, out_size]的矩阵
        # 这个矩阵第i行第j列的元素的含义是：
        # 上一时刻tag为i，这一时刻tag为j的分数  
        ## 这里需要仔细理解，因为这个步骤打破了hmm在条件独立性的假设
        batch_size, max_len, out_size = emission.size()
        crf_scores = emission.unsqueeze(
            2).expand(-1, -1, out_size, -1) + self.transition.unsqueeze(0)
        return crf_scores
    
    def test(self, test_sents_tensor, lengths):
        '''使用维特比算法解码'''
        start_id = self.tag2id['<start>']
        end_id = self.tag2id['<end>']
        pad = self.tag2id['<pad>']
        tagset_size = len(self.tag2id)
        
        crf_scores = self.forward(test_sents_tensor, lengths)
        device = crf_scores.device
        # B表示batch_size, L表示max_len，T表示tagset_size
        B, L, T, _ = crf_scores.size()
        # viterbi[i,j,k]表示第i个句子，第j个字对应第k个标记的最大分数
        viterbi = torch.zeros(B, L, T).to(device)
        # backpointer[i,j,k]表示第i个句子，第j个字对应的第k个标记时前一个标记的id
        backpointer = (torch.zeros(B,L,T).long() * end_id).to(device)
        lengths = torch.LongTensor(lengths).to(device)
        
        # 向前递推
        for step in range(L):
            ## 由于batch是按照长度从长到短排列的
            ## 这里通过batch_size_t表示接下来计算的batch大小
            batch_size_t = (lengths>step).sum().item()
            if step == 0:
                # 第一个字的前一个标记只能是start_id
                viterbi[:batch_size_t, step, :] = crf_scores[:batch_size_t, step, start_id, :]
                backpointer[:batch_size_t, step, :] = start_id
            else:
                max_scores, prev_tags = torch.max(
                    viterbi[:batch_size_t, step-1, :].unsqueeze(2) + 
                    crf_scores[:batch_size_t, step, :, :],  # [B, T, T]
                    dim=1
                )
                viterbi[:batch_size_t, step, :] = max_scores
                backpointer[:batch_size_t, step, :] = prev_tags
        
        # 在回溯的时候我们只需要用到backpointer矩阵
        ## 这里将backpointer矩阵进行了展开
        backpointer = backpointer.view(B, -1)   # [B, L*T]
        tagids = []
        tags_t = None
        for step in range(L-1, 0, -1):
            ## 获取需要解码的矩阵的实际长度
            batch_size_t = (lengths>step).sum().item()
            if step == L-1:
                ## 这里将index定位到当前时间步对应的第一个tag所在的索引
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                ## 由于是最后一个step，所以只需要获取end_id上记录的对应值即可
                index += end_id
            else:
                ## 记录上一个batch有效值大小
                prev_batch_size_t = len(tags_t)
                ## 表示新的batch中第一次出现的数量，
                ## 对于在batch中第一次出现的来说，只需要获取end_id位置记录的索引即可
                ## 对于之前已经出现过的，则需要获取上一个step最佳位置上记录的位置
                new_in_batch = torch.LongTensor(
                    [end_id] * (batch_size_t - prev_batch_size_t)
                ).to(device)
                ## 这里表示相对于第一个索引的偏移量
                offset = torch.cat(
                    [tags_t, new_in_batch],
                    dim=0
                )  # 这个offset实际上就是前一时刻的
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += offset.long()
            try:
                tags_t = backpointer[:batch_size_t].gather(
                    dim=1, index=index.unsqueeze(1).long())
            except RuntimeError:
                import pdb
                pdb.set_trace()
            tags_t = tags_t.squeeze(1)
            tagids.append(tags_t.tolist())
        
        # tagids: 是[L-1]大小的列表
        # 其列表中的元素表示该batch在该时刻的标记
        # 下面修正其顺序，并将维度转换为[B, L]
        tagids = list(zip_longest(*reversed(tagids), fillvalue=pad))
        tagids = torch.Tensor(tagids).long()
        
        return tagids