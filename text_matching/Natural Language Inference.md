&emsp;自然语言推理任务**给定一个premise，判断hypothesis是true（entailment），false（contradiction）还是undetermined（neutral）**。

---

> <font color=blue>**AF-DMN**</font>：[Attention-Fused Deep Matching Network for Natural Language Inference](https://www.ijcai.org/proceedings/2018/0561.pdf) [2018]

（1）模型结构

- **Encoder层**：通过Embedding首先将两个输入转化为向量矩阵的形式；通常会使用BiLSTM网络分别对两个句子进行强化编码。
- **Cross Attention**：
&emsp;给定之前模块输出的两个句子表征$\mathbf{H}^{t-1}_p$和$\mathbf{H}^{t-1}_q$，计算联合注意力矩阵$\mathbf{A}^t \in R^{m \times n}$，其中每个元素表明两个单词之间的相关度。

$$
\mathbf{A}_{i,j}^t = h^{{t-1}^T}_{p_i} W^t h_{q_j}^{t-1} + <\mathbf{U}^t_l, h_{p_i}^{t-1}> + <\mathbf{U}_r^t, h^{t-1}_{q_j}>
$$

&emsp; 其中$W^t \in \mathbb{R}^{2h \times 2h}, \; U^t_l, U^t_r \in \mathbb{R}^{2h}$，$<.,.>$表示内积。

&emsp;然后，可以计算得到每个单词$p_i$和$q_j$的注意力表征$\tilde{h}_{p_i}^t \in \mathbb{R}^{2h}, \; \tilde{h}^t_{q_j} \in \mathbb{R}^{2h}$：

$$
\mathbf{a}^t_{p_i} = \text{softmax} (\mathbf{A}^t_{i:}), \; \mathbf{a}^t_{q_j} = \text{softmax}(\mathbf{A}^t_{:j}) \\ 

\tilde{\mathbf{h}}^t_{p_i} = \mathbf{H}^{t-1}_1 \cdot \mathbf{a}_{p_i}^t , \;  \tilde{\mathbf{h}}^{t}_{q_j} = \mathbf{H}_p^{t-1} \cdot \mathbf{a}^t_{q_j}
$$
- **Fusion for Cross Attention**：

&emsp;为了进一步强化交互的结果，基于cross attention的结果计算融合

$$
\bar{\mathbf{f}}^t_{p_i} = [\mathbf{h}^t_{p_i}; \tilde{\mathbf{h}}^t_{p_i}; \mathbf{h}^t_{p_i} - \tilde{\mathbf{h}}^t_{p_i}; \mathbf{h}^t_{p_i} \odot \tilde{\mathbf{h}}^t_{p_i}] \\ 

\tilde{\mathbf{f}}^t_{p_i} = \text{ReLU}(W^t_f \bar{\mathbf{f}}^t_{p_i} + \mathbf{b}^t_f) \\ 

\mathbf{f}^t_{p_i} = \text{Bi-LSTM}(\tilde{\mathbf{f}}^t_{p_i}, \mathbf{f}^t_{p_{i-1}}, \mathbf{f}^t_{p_{i+1}})
$$
- **Self Attention**：

&emsp;为了引入长句子的长期依赖，使用了self-attention的策略。对于一个句子，首先计算self-attention矩阵$\mathbf{S}^t \in \mathbf{R}^{m \times m}$：

$$
\mathbf{S}^t_{i, j} = <\mathbf{f}^t_{p_i}, \mathbf{f}^t_{p_j}>
$$
&emsp;然后计算每个单词的attention向量：

$$
s^t_{p_i} = \text{softmax}(\mathbf{S}^t_i), \quad \bar{\mathbf{h}}^t_{p_i} = \mathbf{F}^t_p \cdot s^t_{p_i}
$$
- **Fusion for Self Attention**：

&emsp;句子的融合表征会进入下一个block

$$
\bar{\mathbf{h}}^t_{p_i} = [\mathbf{f}^t_{p_i}; \bar{\mathbf{h}}^t_{p_i}; \mathbf{f}^t_{p_i} - \bar{\mathbf{h}}^t_{p_i}; \mathbf{f}^t_{p_i} \odot \bar{\mathbf{h}}^t_{p_i}] \\ 

\tilde{\mathbf{h}}^t_{p_i} = \text{ReLU}(W^t_h \bar{\mathbf{h}}^t_{p_{i-1}} + \mathbf{b}^t_h) \\ 

\mathbf{h}^t_{p_i} =  \text{Bi-LSTM}(\tilde{\mathbf{h}}^t_{p_i}, \mathbf{h}^t_{p_{i-1}}, \mathbf{h}^t_{p_{i+1}})
$$
- **预测层**：使用Pooling的方式将两个句子的向量矩阵转化为固定长度的向量，然后输入一个2层的MLP中。为了能够捕获所有的信息以及突出两个句子的关键特性，我们在每个句子中都采用了mean pooling和max pooling，并进行concat。

<div align=center><img src="http://ww1.sinaimg.cn/large/006Ejijoly1g9slhnmpj1j30sa0glq5d.jpg"/></div>
---

> <font color=blue>**HCRN**</font>：[Hermitian Co-Attention Networks for Text Matching in Asymmetrical Domains](https://www.ijcai.org/proceedings/2018/0615.pdf) [2018]

（1）模型结构

- **Input Encoding**：输入两个语句$(a, b)$，经过Embedding层将单词转化为词向量，每个单词经过一个投影层，使用ReLU激活；然后每个序列经过一个BiLSTM结果，得到单词的强化想来那个。
- **Hermitian Co-Attention**：

&emsp;复数值的点积为：

$$
<a_i, b_j> = \bar{a}_i^T b_j
$$
这里$\bar{a}_i$表示$a_i$的共轭值。

&emsp;定义相似度矩阵计算方式为：

$$
s_{i,j} = \text{Re}(<a_i + iF_{\text{proj}}(a_i), b_j + i F_{\text{proj}}(b_j)>)
$$

这里$<.,.>$表示**Hermitian内积**，而$Re(.)$表示取复数的实数部分。

&emsp;本文还探索了稍复杂一点的bilinear积：

$$
s_{i, j} = \text{Re}(a^T_i M b_j)
$$

- **Hermitian Intra-Attention**：

&emsp;通过自注意力层，可以得到强化的上下文表征：

$$
x_i' = \sum_{j=1}^l \frac{\exp(\hat{s}_{ij})}{\sum_{k=1}^l \exp(\hat{s}_{ik})} x_j
$$
 其中，$\hat{s}_{ij}$表示自注意力，也是基于Hermitian计算得到的。

- **聚合和预测**：通过基于Attention加权得到两个句子的句向量表征，concat之后经过MLP得到最终结果。

<div align=center><img src="http://ww1.sinaimg.cn/large/006Ejijoly1g9ryahz3nlj30dr0erdgm.jpg"/></div>
---

> <font color=blue>**SAN**</font>：[Stochastic Answer Networks for Natural Language Inference](https://arxiv.org/pdf/1804.07888.pdf) [2018]

（1）模型结构

- **Lexicon Encoding层**：首先，将word embeddings和character embeddings拼接处理OOV问题，对于$P$和$H$，我们使用两个独立的前向网络对单词进行强化编码，得到$E^p \in \mathbb{R}^{d \times m}, \; E^h \in \mathbb{R}^{d \times n}$；
- **Contextual Encoding层**：使用两个堆叠的BiLSTM层对$P$和$H$编码上下文信息，由于是双向的，所以每个单词向量的维度会加倍；使用maxout层将BiLSTM的输出缩减到原始的大小；将两个BiLSTM的输出拼接，得到$C^p \in \mathbb{R}^{2d \times m}, \; C^h \in \mathbb{R}^{2d \times n}$分别作为$P, \; H$的表征；
- **Memory层**：首先使用点积计算$P$和$H$中tokens的相似度，不同于Transformer中使用标量对分数进行归一化，我们使用一个线性层将$C^p, C^h$转化为$\hat{C}^p, \hat{C}^h$：
$$
A = \text{dropout} (f_{\text{attention}}(\hat{C}^p, \hat{C}^h)) \in \mathbb{R}^{m \times n}
$$
&emsp;其中$A$是attention矩阵，dropout用于平滑；于是可以通过交叉计算和拼接得到聚合了对方信息后的premise和hypothesis$U^p = [C^p; C^h A] \in \mathbb{R}^{4d \times m}$和$U^h = [C^h; C^p A^T] \in \mathbb{R}^{4d \times n}$。最后，得到premise和hypothesis的最终表征$M^p = \text{BiLSTM}([U^p; C^p])$和$M^h = \text{BiLSTM}([U^h; C^h])$。
- **Answer层**：首先，初始状态$s_0$是hypothesis中的Attention向量，即：
$$
s_0 = \sum_j \alpha_j M^h_j \\ 

\alpha_j = \frac{\exp(\theta_2 \cdot M^h_j)}{\sum_{j'} \exp(\theta_2 \cdot M^h_{j'})}
$$

&emsp;使用GRU进行状态传递的计算，即$s_t = \text{GRU}(s_{t-1}, x_t)$，其中$x_t$是上一个状态$s_{t-1}$和记忆矩阵$M^p$的结果：

$$
x_t = \sum_j \beta_j M^p_j \\ 

\beta_j = \text{softmax} (s_{t-1} \theta_3 M^p)
$$

&emsp;在每个时间步$t \in \{0,1,\cdots, T-1\}$，使用一层分类层得到最终的关系：

$$
P_t^r = \text{softmax} (\theta_4 [s_t; x_t; |s_t - x_t|; s_t \odot x_t])
$$
&emsp;最后，使用所有$T$个时间步的输出平均值作为最终的输出分数：
$$
P^r = \text{avg} ([P_0^r, P_1^r, \cdots, P^r_{T-1}])
$$
&emsp;训练期间，在平均概率之前使用随机概率dropout；在解码期间，平均所有输出改善模型性能。**随机概率dropout是指在平均之间随机dropout掉某一个step输出的概率，防止模型对某一个step的输出过于依赖**。

<div align=center><img src="http://ww1.sinaimg.cn/large/006Ejijoly1g9rvjfi3lmj30kd0je0vm.jpg"/></div>
---

> <font color=blue>**DIIN**</font>：[Natural Language Inference Over Interaction Space](https://arxiv.org/pdf/1709.04348.pdf) [2018]

（1）网络结构

- **Embedding层**：将word embedding，经过CNN的char embedding，以及POS tag的onehot向量和精确匹配二值化特征进行拼接，得到每个单词的表征；
- **Encoding层**：将premise和hypothesis经过两层HighWay网络，得到强化表征，新的表征经过self-attention，考虑单词的顺序和上下文信息；
$$
\mathbf{A}_{ij} = \alpha(\hat{\mathbf{P}}_i) \\

\bar{\mathbf{P}}_i = \sum_{j=1}^p \frac{\exp(\mathbf{A}_{ij})}{\sum_{k=1}^p \exp(\mathbf{A}_{ki})} \hat{\mathbf{P}}_j
$$

&emsp;其中$\bar{\mathbf{P}}_i$是根据$\hat{\mathbf{P}}$加权的结果，使用$\alpha(\mathbf{a}, \mathbf{b}) = w_a^T [a; b; a \odot b]$；然后将$\hat{\mathbf{P}}$和$\bar{\mathbf{P}}$拼接之后传入语义混合门。混合门的实现形式为：

$$
z_i = \tanh(W^{1T}[\hat{P}_i; \bar{P}_i] + b^1) \\ 

r_i = \sigma (W^{2T}[\hat{P}_i; \bar{P}_i] + b^2) \\ 

f_i = \sigma(W^{3T} [\hat{P}_i; \bar{P}_i] + b^3) \\ 

\tilde{P}_i = r_i \odot \hat{P}_i + f_i \odot z_i
$$
- **Interaction层**：交互层对premise和hypothesis强化之后的编码进行交互。

$$
\mathbf{I}_{ij} = \beta(\tilde{P}_i, \tilde{H}_j)
$$

&emsp;交互的形式有很多种，我们发现$\beta(a, b) = a \odot b$是一种非常有效的形式。

- **Feature Extraction层**：使用DenseNet作为特征提取器，对上一步得到的交互张量，首先使用CNN对通道数进行衰减，不接ReLU；然后将feature map通过3组Dense Block，在经过衰减层衰减，使用池化对维度衰减。
- **Output层**：使用线性层对最终结果进行分类。

<div align=center><img src="http://ww1.sinaimg.cn/large/006Ejijoly1g9rptf2qa5j30gp0ii774.jpg"/></div>
---

> <font color=blue>**ESIM**</font>：[Enhanced LSTM for Natural Language Inference ](https://arxiv.org/pdf/1609.06038.pdf)[2017]

（1）模型结构

- 首先基于BiLSTM对输入的两个语句进行编码，得到编码了上下文的每个单词的输出向量；
- 基于每个单词的编码向量，计算相似度得到二维相似度矩阵；
- 用上面得到的相似度矩阵，结合第一步BiLSTM的输出，互相生成对对方各个单词向量加权之后的词向量序列；
- 经过上述local inference之后，进行拼接，并计算每个对应位置词向量的差和点积，拼接到后面；
- 对拼接后的词向量再次使用BiLSTM进行编码，再对编码后的向后使用Maxpooling和AvgPooling，拼接得到每个句子的句向量；
- 将两个句子的句向量进行拼接，经过一层线形成，得到最终的类别。

（2）图中左右表示ESIM，右侧是TreeLSTM

<div align=center><img src="http://ww1.sinaimg.cn/large/006Ejijoly1g9e0p6njraj30bx0fbdgz.jpg"/></div>
---

> <font color=blue>**BiMPM**</font>：[Bilateral Multi-Perspective Matching for Natural Language Sentences](https://arxiv.org/pdf/1702.03814.pdf)[2017]

（1）本文提出了一种双向多视角的匹配模型（BiMPM）。

（2）模型结构：

- **Word Representation Layer**：将输入两句话的每个单词进行向量表征，表征由词向量和Char Embedding经过LSTM编码之后的向量进行拼接；
- **Context Representation Layer**：使用一个BiLSTM对两个文本进行强化编码，得到每个time step的上下文表征向量；
- **Matching Layer**：匹配层的目的是将一个句子中单词的上下文表征向量和另一个句子中单词的上下文表征向量进行比对，使用4种匹配策略：
    - Full-Matching：每个前向上下文Embedding和另一个句子最后一个time-step向量匹配，每个后向上下文Embedding和另一个句子第一个上下文Embedding匹配；
    - Maxpooling-Matching：每个前向（后向）上下文Embedding和另一个句子每一个前向（后向）上下文Embedding匹配，只保留匹配得到的最大值；
    - Attentive-Matching：首先计算一个前向（后向）Embedding和另一个句子的前向（后向）Embedding的相似度，取相似度的值作为计算加权向量的权重得到Attentive向量，然后使用Attentive向量和两一个句子中每一个向量计算匹配向量；
    - Max-Attentive-Matching：取另一个句子每个维度的最大值作为Attentive向量，然后按照上面的方式进行计算。
> 注：两个向量计算得到$k$维向量的方法：

$$
\mathbf{m} = f_m (\mathbf{v}_1, \mathbf{v}_2; \mathbf{W}) \\ 

m_k = \text{cosine} (W_k \odot v_1, W_k \odot v_2)
$$
- **Aggregation Layer**：通过匹配层，得到每个句子强化之后的词向量表征，每个词向量为$k$维；对两个句子分别使用BiLSTM进行编码，得到两个固定长度的匹配向量，并拼接；
- **Prediction Layer**：经过MLP输出最终的分数。

<div align=center><img src="http://ww1.sinaimg.cn/large/006Ejijoly1g9rgzqocpbj30el0evdh7.jpg"/></div>
---

> <font color=blue>**Decomposable Attention**</font>：[A Decomposable Attention Model for Natural Language Inference](https://arxiv.org/pdf/1606.01933.pdf)[2016]

（1）模型结构

- 输入两个文本$\mathbf{a} = (a_1, \cdots, a_{l_a})$和$\mathbf{b} = (b_1, \cdots, b_{l_b})$，通过Embedding层得到每个单词对应的词向量；
- 使用Intra-Sentence Attention对每个句子进行加强编码；

$$
f_{ij} := F_{\text{intra}}(a_i)^T F_{\text{intra}}(a_j) 
$$
&emsp;$F_{\text{intra}}$是一个前向网络，然后构建自对齐的短语

$$
a_i' = \sum_{j=1}^{l_a} \frac{\exp(f_{ij} + d_{i-j})}{\sum_{k=1}^{l_a} \exp(f_{ik} + d_{i-k})} a_j
$$
&emsp;上面的偏置项$d_{i-j}$表示给模型提供的序列信息的最小量。

- 输入变成$\bar{a}_i := [a_i, a'_i]$，将$\bar{a}_i, \; \bar{b}_j$经过一层前向网络，计算每个单词的交互值，得到输入两个文本的交互attention矩阵；

$$
e_{ij} = F(\bar{a}_i)^T F(\bar{b}_j) \\ 

\beta_i := \sum_{j=1}^{l_b} \frac{\exp(e_{ij})}{\sum_{k=1}^{l_b} \exp(e_{ik})} \bar{b}_j \\ 

\alpha_j := \sum_{i=1}^{l_a} \frac{\exp(e_{ij})}{\sum_{k=1}^{l_a} \exp(e_{kj})} \bar{a}_i
$$
&emsp;其中$\beta_i$和$\bar{a}_i$对齐，$\alpha_j$和$\bar{b}_j$对齐。

- 经过上一步得到两个文本中每个单词的拼接向量$\{(\bar{a}_i, \beta_i)\}^{l_a}_{i=1}$以及$\{(\bar{b}_j, \alpha_j)\}^{l_b}_{j=1}$，然后将拼接的向量传入一个前向网络得到每一个单词的融合表征$\{v_{1, i}\}_{i=1}^{l_a}$和$\{v_{2, j}\}_{j=1}^{l_b}$；
- 通过计算每个句子中融合向量的和得到句子向量$\mathbf{v}_1, \;  \mathbf{v}_2$，将两个向量拼接，然后输入线性层得到最终的分类结果。

<div align=center><img src="http://ww1.sinaimg.cn/large/006Ejijoly1g9rfwwoggqj30di075dgg.jpg"/></div>
---

> <font color=blue>**Match-LSTM**</font>：[Learning Natural Language Inference with LSTM](https://www.aclweb.org/anthology/N16-1170.pdf)[2016]

（1）定义输入的premise为$X^s$，hypothesis为$X^t$，判断两者之间的关系。

（2）模型结构

- 首先，将输入经过Embedding进行编码，使用两层的LSTM网络分别对premise和hypothesis进行编码，分别使用$h_j^s, h_k^t$表示编码之后的隐层状态；
- 计算$h_k^t$相对于$h_j^s$的Attention向量：
$$
\mathbf{a}_k = \sum_{j=1}^M \alpha_{kj} h^s_j \\

s.t. \quad \alpha_{kj} = \frac{\exp(e_{kj})}{\sum_{j'} \exp(e_{kj'})} \\ 

e_{kj} = w^e \cdot \tanh(W^sh^s_j + W^th_k^t + W^m h_{k-1}^m)
$$
- 使用mLSTM网络计算上述公式中的$h_{k}^m$，每次mLSTM的输入为：

$$
m_k = \begin{bmatrix}\mathbf{a}_k \\ \mathbf{h}_k^t \end{bmatrix}
$$
- 最后，使用mLSTM最后一层隐层的输出$\mathbf{h}_N^m$经过一个MLP即可得到最终的输出。

<div align=center><img src="http://ww1.sinaimg.cn/large/006Ejijoly1g9qsxtcti3j30bz06a74n.jpg"/></div>
