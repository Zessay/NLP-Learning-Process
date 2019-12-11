
QA任务是指**给定一个问题，自动在许多回复中寻找最相关的回复（Answer Selection），或者所有相关问题来重用它们的回答（Question Retrieval）**。

---

> <font color=blue>**HCAN**</font>：[Bridging the Gap Between Relevance Matching and Semantic Matching for Short Text Similarity Modeling](https://cs.uwaterloo.ca/~jimmylin/publications/Rao_etal_EMNLP2019.pdf) [2019]

（1）定义输入序列query的长度为$n$，context的长度为$m$

（2）模型结果

- 通过Embedding层将query和context转化为词向量矩阵，使用3种不同的编码器：
    - **Deep Encoder**：由多层CNN堆叠而成，来得到高级别的`k-gram`特征，对于不同的CNN层，query和context参数是共享的；
    - **Wide Encoder**：并联的CNN组成，使用不同的窗口大小，得到不同的`k-gram`信息；
    - **Contextual Encoder**：使用BiLSTM捕获长距离的上下文依赖。

&emsp; 相较而言，Deep Encoder更加节省参数。

- **Relevance Matching**：基于编码器得到的query和context表征，计算相似度矩阵；基于列归一化得到相似度矩阵$\tilde{S}$，分别进行最大池化和平均池化得到判别向量$\text{Max}(S)$和$\text{Mean}(S)$；引入权重作为先验衡量不同query terms和phrases的重要性，这里使用**IDF作为重要性的值**；得到$O_{\text{RM}}$。

$$
O_{\text{RM}} = \{\text{wgt}(q) \odot \text{Max}(S), \text{wgt}(q) \odot \text{Mean}(S)\}

O_{\text{RM}} \in 2 \cdot \mathbb{R}^{n}
$$

- **Semantic Matching**：基于编码器得到的query和context表征，使用co-attention：
$$
A = \text{REP}(U_q W_q) + \text{REP}(U_c W_c) + U_qW_bU_c^T 

A = \text{softmax}_{\text{col}} (A)

A \in \mathbb{R}^{n \times m}
$$

&emsp;$\text{REP}$表示将输入向量转化为$\mathbb{R}^{n \times m}$；在两个方向上使用co-attention，即query-to-context和context-to-query：

$$
\tilde{U}_q = A^T U_q

\tilde{U}_c = \text{REP}(\text{max}_{\text{col}}(A)U_c)

\tilde{U}_q \in \mathbb{R}^{m \times F}, \; \tilde{U}_c \in \mathbb{R}^{m \times F}
$$
&emsp;接着将两个矩阵拼接，并计算交叉特征，最后经过一个BiLSTM，提取最后一层的输出作为两个语句的语义匹配输出：

$$
H = [U_c; \tilde{U}_q; U_c \otimes \tilde{U}_q; \tilde{U}_c \otimes \tilde{U}_q]

O_{\text{SM}} = \text{BiLSTM}(H)

H \in \mathbb{R}^{m \times 4F}, \quad  O_{SM} \in R^{4F} 
$$
- 将每一个编码层得到的RM向量和SM向量进行拼接，经过两层全连接层，使用ReLU激活函数生成最终的预测结果。

<div align=center><img src="http://ww1.sinaimg.cn/large/006Ejijoly1g9t2akwg7wj30l50cijss.jpg"/></div>
---

> <font color=blue>**HAR**</font>：[A Hierarchical Attention Retrieval Model for Healthcare Question Answering](http://dmkd.cs.vt.edu/papers/WWW19.pdf) [2018]

（1）基于传统模型并没有特别关注query和document文本长度在匹配过程中的问题，本文提出针对长文本的建模方法。使用$q$表示query文本，使用$\{1d, \cdots, ld\}$表示document中的每一个句子。

（2）模型结构：

- 首先，使用Embedding层将输入query和document转化为词向量矩阵；
- 经过BiGRU分别对query和document进行加强编码，每个单词得到对应的上下文表征向量；
- 使用双向注意力机制，对document中的每一个sentence进行交叉特征编码;

$$
s_{xy} = w_c^T \cdot [u_x^{id}; u_y^q; u_x^{id} \odot u_y^q] \\

\bar{S}_{D2Q} = \text{softmax}_{\text{row}}(S) \\ 

\bar{S}_{Q2D} = \text{softmax}_{\text{col}}(S) \\

A_{D2Q} = \bar{S}_{D2Q} \cdot U^q \\ 

A_{Q2D} = \bar{S}_{D2Q} \cdot \bar{S}_{Q2T}^T \cdot U^{id} \\

V^{id} = [U^{id}; A_{D2Q}; U^{id} \odot A_{D2Q}; U^{id} \odot A_{Q2D}] \in \mathbb{R}^{n \times 4H}
$$

- 对query进行自注意力编码，得到query的表征；
- 对第3步得到的每一个document的交叉注意力编码进行两层的self-attention，得到document的表征向量；
- 将document的表征向量经过FFN得到和query维度相同的向量；
- 将query和document向量按元素相乘，并将最终的结果经过MLP得到最后的匹配分数。

<div align=center><img src="http://ww1.sinaimg.cn/large/006Ejijoly1g9qqpfhj5tj30kw0hbjts.jpg"/></div>
---

> <font color=blue>**MIX**</font>：[MIX: Multi-Channel Information Crossing for Text Matching](https://sites.ualberta.ca/~dniu/Homepage/Publications_files/hchen-kdd18.pdf) [2018]

（1）模型结构

- 输入question和answer，转化成词向量，计算unigram的交互矩阵；
- 使用CNN分别得到question和answer的bigram向量和trigram向量，计算bigram和trigram的交互矩阵；
- 根据question和answer对应的每个词的IDF和POS得到IDF Attention矩阵和POS Attention矩阵，同时定义一个参数化矩阵表征位置的重要性Position Attention；
- 将3个Attention矩阵和上面得到的gram交互矩阵计算，得到27层矩阵并进行Stack；
- 经过多层的CNN矩阵，得到表征向量输入MLP中得到最终的匹配结果。

<div align=center><img src="http://ww1.sinaimg.cn/large/006Ejijoly1g9qjzz1p6aj30rk0hf0vb.jpg"/></div>
---

> <font color=blue>**MCAN**</font>：[Multi-Cast Attention Networks for Retrieval-based Question Answering and Response Prediction](https://arxiv.org/pdf/1806.00778.pdf) [2018]


（1）模型结构：

- 输入query和document，使用一层HighWay网络对单词进行强化编码；
- 使用4种不同的Attention策略，分别是基于query和document交互矩阵的max-pooling Attention，mean-pooling Attention，Alignment Attention以及基于词向量矩阵的Self Attention；对于每一个单词向量，分别得到4个不同的Attention向量；
- 将每种Attention得到的向量，和原始向量使用3种交互方式（concat，按元素乘积，按元素相减），得到3种向量；
- 使用向量到标量的映射函数，将每个向量映射成一个变量（Sum或NN或FM）；
- 这样，针对每个单词得到12个标量，将这些标量进行拼接，再拼接到原始词向量的后面；
- 将上面得到的向量经过BiLSTM，query和document共享权重，使用MeanMax池化的策略，将得到每个序列对应的向量；
- 将query和document的向量经过（concat，按元素乘积，按元素相减）得到的向量，输出两层HighWay网络；
- 最后将HighWay网络的输出经过线性层，得到输出。

<div align=center><img src="http://ww1.sinaimg.cn/large/006Ejijoly1g9qc5lytqsj30g60g9q4k.jpg"/></div>
---

> <font color=blue>**aNMM**</font>：[aNMM: Ranking Short Answer Texts with Attention-Based Neural Matching Model](https://arxiv.org/pdf/1801.01641.pdf) [2016]

（1）aNMM-1模型结构

- 输入Question和Answer，将单词映射为词向量矩阵，根据词向量计算相似度匹配矩阵；
- 将匹配矩阵中元素的取值范围`[-1, 1]`等间距的划分成多个bins；
- 对于Q中的每一个单词，计算位置每一个bin中的元素的和，通过乘以参数矩阵再经过函数激活，得到对应的节点；
- 对于每一个单词，基于单词词向量和参数向量得到每一个单词的Attention权重，基于窗口M计算加权大小；
- 得到向量经过MLP得到最终分数。

（2）aNMM-2模型结构

- 输入Question和Answer，将单词映射为词向量矩阵，根据词向量计算相似度匹配矩阵；
- 将匹配矩阵中元素的取值范围`[-1, 1]`等间距的划分成多个bins；
- 对于Q中的每一个单词，计算位置每一个bin中的元素的和，**通过乘以多个不同的参数矩阵**，每一个单词得到多个不同的节点；
- 将每个单词得到的$T$个不同节点，再乘以一个参数向量，经过激活函数，每个单词得到一个节点；
- 对于每一个单词，基于单词词向量和参数向量得到每一个单词的Attention权重，基于窗口M计算加权大小；
- 得到向量经过MLP得到最终分数。

<div align=center><img src="http://ww1.sinaimg.cn/large/006Ejijoly1g9of0pkw8rj30pa0dcn0u.jpg"/></div>
