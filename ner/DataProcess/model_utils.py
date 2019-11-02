import torch
import torch.nn.functional as F

# **********CRF工具函数**************
def word2features(sent, i):
    '''抽取单个字的特征'''
    word = sent[i]
    prev_word = "<s>" if i == 0 else sent[i-1]
    next_word = "</s>" if i == (len(sent) - 1) else sent[i+1]
    # 使用的特征
    # 前一个词，当前词，后一个词
    # 前一个词+当前词，当前词+后一个词
    features = {
        'w': word,
        'w-1': prev_word,
        'w+1': next_word, 
        'w-1:w': prev_word+word,
        'w:w+1': word+next_word,
        'bias': 1
    }
    return features


def sent2features(sent):
    '''抽取序列的特征'''
    return [word2features(sent, i) for i in range(len(sent))]

# ************LSTM工具函数***********
def tensorized(batch, maps):
    '''将单词映射为对应的索引'''
    # batch是按照原始的长度排序的
    PAD = maps.get('<pad>')
    UNK = maps.get('<unk>')
    
    max_len = len(batch[0])
    batch_size = len(batch)
    
    batch_tensor = torch.ones(batch_size, max_len).long() * PAD
    for i, l in enumerate(batch):
        for j, e in enumerate(l):
            batch_tensor[i][j] = maps.get(e, UNK)
    # batch中各个句子的长度
    lengths = [len(l) for l in batch]
    return batch_tensor, lengths

def sort_by_lengths(word_lists, tag_lists):
    '''按照长度对整个数据进行排序'''
    pairs = list(zip(word_lists, tag_lists))
    # 按照从长到短的顺序排序
    indices = sorted(range(len(pairs)), 
                    key=lambda k: len(pairs[k][0]),
                    reverse=True)
    pairs = [pairs[i] for i in indices]
    word_lists, tag_lists = list(zip(*pairs))
    # indices保存的是排序之前的每个元素的原索引位置
    return word_lists, tag_lists, indices

def cal_loss(logits, targets, tag2id):
    '''计算损失
    logits: [B, L, out_size]
    targets: [B, L]
    lengths: [B]
    '''
    PAD = tag2id.get('<pad>')
    assert PAD is not None
    
    mask = (targets != PAD)
    targets = targets[mask]
    out_size = logits.size(2)
    logits = logits.masked_select(
        mask.unsqueeze(2).expand(-1, -1, out_size)
    ).contiguous().view(-1, out_size)
    
    assert logits.size(0) == targets.size(0)
    loss = F.cross_entropy(logits, targets)
    return loss

def indexed(targets, target_size, start_id):
    '''
    将targets中的数转化为在[T*T]大小序列中的索引，T是标注的种类
    '''
    batch_size, max_len = targets.size()
    for col in range(max_len-1, 0, -1):
        targets[:, col] += (targets[:, col-1]*target_size)
    targets[:, 0] += (start_id * target_size)
    return targets


def cal_lstm_crf_loss(crf_scores, targets, tag2id):
    """计算双向LSTM-CRF模型的损失
    该损失函数的计算可以参考:https://arxiv.org/pdf/1603.01360.pdf
    """
    pad_id = tag2id.get('<pad>')
    start_id = tag2id.get('<start>')
    end_id = tag2id.get('<end>')
    
    device = crf_scores.device
    
    # targets: [B, L], crf_scores: [B, L, T, T]
    batch_size, max_len = targets.size()
    target_size = len(tag2id)
    
    # 选择非pad的单词
    mask = (targets != pad_id)
    lengths = mask.sum(dim=1)
    targets = indexed(targets, target_size, start_id)
    
    targets = targets.masked_select(mask)  # [real_L]
    flatten_scores = crf_scores.masked_select(
        mask.view(batch_size, max_len, 1, 1).expand_as(crf_scores)
    ).view(-1, target_size*target_size).contiguous()
    
    golden_scores = flatten_scores.gather(dim=1, index=targets.unsqueeze(1)).sum()
    
    # 计算path scores
    # scores_upto_t[i, j]表示第i个句子中的第t个词被标注为j标记的所有t时刻之前的所有子路径的分数之和
    scores_upto_t = torch.zeros(batch_size, target_size).to(device)
    
    for t in range(max_len):
        # 当前时刻有效的batch_size
        batch_size_t = (lengths > t).sum().item()
        if t == 0:
            scores_upto_t[:batch_size_t] = crf_scores[:batch_size_t, t, start_id, :]
        else:
            scores_upto_t[:batch_size_t] = torch.logsumexp(
                crf_scores[:batch_size_t, t, :, :] + 
                scores_upto_t[:batch_size_t].unsqueeze(2),
                dim=1
            )
    all_path_scores = scores_upto_t[:, end_id].sum()
    
    loss = (all_path_scores - golden_scores) / batch_size
    return loss