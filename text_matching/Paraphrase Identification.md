&emsp;释义识别是判断两个句子是否有相同含义的任务，是自然语言理解的标准。

---

> <font color=blue>**MwAN**</font>：[Multiway Attention Networks for Modeling Sentence Pairs](https://www.ijcai.org/proceedings/2018/0613.pdf) [2018]

（1）模型结构

- 输入两个语句，首先将每个单词转化为对应的词向量以及上下文向量的拼接（预训练模型），然后使用BiGRU得到每个单词的双向表征；
- 基于4种不同的Attention函数分别得到两个语句中每个位置单词相对于另一个语句中所有单词的Attention表征 ；
- 将4种表征分别和每个位置的向量拼接，使用BiGRU得到基于每种Attention的上下文表征；
- 将4种不同的Attention上下文表征使用Attention得到每一个位置的融合表征，再经过一层BiGRU；
- 得到每个位置的强化表征向量之后，使用参数基于Attention计算分数，得到其中一个句子的表征；
- 将这个句子的表征作为参数，和另一个句子中的单词计算Attention分数，得到最终的融合表征；
- 将融合表征经过MLP得到最终的匹配分数。

<div align=center><img src="http://ww1.sinaimg.cn/large/006Ejijoly1g9o8odpk66j30o90gmdif.jpg"/></div>
---

> <font color=blue>**ABCNN**</font>：[ABCNN：Attention-Based Convolutional Neural Network for Modeling Sentence Pairs](https://arxiv.org/pdf/1512.05193.pdf) [2016]

（1）ABCNN提出了3种Attention机制将句子间的相互影响整合到CNN网络中，这样每个句子的表征可以同时兼顾另一个句子。

（2）ABCNN-1模型结构

- 输入两个文本，将每个单词转化为对应的词向量；
- 计算两个矩阵的匹配分数，得到匹配矩阵$A$，计算公式为$\frac{1}{1 + |x-y|}$，这里$|\cdot|$表示欧氏距离；
- 使用两个参数矩阵（共享权重）将匹配矩阵转化为每个句向量矩阵相同的形状；
- 将上面得到的两个矩阵，在Channel维度上了各自的原始矩阵进行拼接；
- 将得到的拼接后的矩阵输入多层的CNN-Pooling网络，最后经过MLP得到最终的输出。

<div align=center><img src="http://ww1.sinaimg.cn/large/006Ejijoly1g9o5kt5lz8j30nr09fabk.jpg"/></div>
（3）ABCNN-2模型结构

- 输入两个文本，将单词转化为对应的词向量；
- 将两个词向量矩阵经过一层CNN的结构，得到的CNN的输出；
- 比较两个输出矩阵中对应位置的短语向量，得到$s_0 \times s_1$的矩阵$A$；
- 将匹配矩阵的每一行的元素相加，得到一个长度为$s_0$的向量；同理，将每一列元素相加，得到一个长度为$s_1$的向量；
- 基于长度为$w$的窗口进行滑动，计算向量对应元素和CNN输出向量的加权值，最终得到两个和CNN输出向量大小相同的矩阵；
- 将矩阵再经过之后的CNN，使用相同的pooling方式；
- 将最终的输出经过MLP得到最终的结果。

<div align=center><img src="http://ww1.sinaimg.cn/large/006Ejijoly1g9o60vwtdcj30nu0b4jt0.jpg"/></div>
（4）ABCNN-3模型结构

- 将上面两种Attention的方式结合，在CNN前和CNN后同时使用Attention.

<div align=center><img src="http://ww1.sinaimg.cn/large/006Ejijoly1g9o62evcinj30ni0bxjts.jpg"/></div>
---

> <font color=blue>**Match-SRNN**</font>：[Match-SRNN: Modeling the Recursive Matching Structure with Spatial RNN](https://arxiv.org/pdf/1604.04378.pdf) [2016]

（1）本文提出了一种Spatial GRU的方法来对匹配矩阵进行处理，获得全局相关性表征。

（2）模型结构：

- 输入两个文本，基于每个单词进行相似度计算，得到高维的相关性表征张量；
- 对张量进行Spatial RNN，从左上到右下或者从右下到左上，得到全局相关性表征；
- 将得到的全局相关性表征，经过MLP得到最终的输出。

<div align=center><img src="http://ww1.sinaimg.cn/large/006Ejijoly1g9n1pcs9kwj30fc07omxm.jpg"/></div>
---

> <font color=blue>**MultiGranCNN**</font>：[MultiGranCNN: An Architecture for General Matching of Text Chunks on Multiple Levels of Granularity](https://www.aclweb.org/anthology/P15-1007.pdf) [2015]

（1）模型结构

- 首先，使用两个相同的gpCNN孪生网络，对于两个输入进行编码，分别得到不同级别的g-phrase表征；
- 然后基于3种不同的匹配特征模型（DirectSim，Concat或者InDireactSim），得到$s_1 \times s_2$的匹配矩阵，每个单元表示来自两个不同chunks的g-phrase的相似度；
- 通过2D动态池化的策略，将这个匹配特征转化成固定的大小；
- 将得到的固定大小的矩阵输入到mfCNN网络中，基于基础的交互特征提取更复杂的交互特征；
- 最终，将mfCNN的输出输入到MLP中计算最终的得分。

<div align=center><img src="http://ww1.sinaimg.cn/large/006Ejijoly1g9n6tvy6wkj30p407ywfh.jpg"/></div>
---

> <font color=blue>**MatchPyramid**</font>：[Text Matching as Image Recognition](https://arxiv.org/pdf/1602.06359.pdf) [2016]

（1）本文提出将Text Matching任务看做Image Recognition任务，通过多层的CNN和Max-Pooling，最后使用MLP得到匹配分数。

（2）模型结构

- 根据输入的两个文本，计算相似度匹配矩阵，可以使用的相似度计算函数包括**Indicator，Cosine，Dot**；
- 将相似度匹配矩阵输入CNN，经过动态池化得到特定size的特征图；
- 将上一步的结果再经过多次CNN和Max-Pooling，得到输出；
- 将输出经过MLP得到匹配分数。

<div align=center><img src="http://ww1.sinaimg.cn/large/006Ejijoly1g9mzbn6yqtj309i0fd75s.jpg"/></div>
---

> <font color=blue>**MV-LSTM**</font>：[A Deep Architecture for Semantic Matching with Multiple Positional Sentence Representations](https://arxiv.org/pdf/1511.08277.pdf) [2016]

（1）模型结构

- 使用BiLSTM对语句中的每个单词进行加强编码；
- 基于强化编码，使用特定的相似度计算函数，得到匹配矩阵；
- 对匹配矩阵使用k-Maxpooling，再经过MLP得到最终的匹配分数。

<div align=center><img src="http://ww1.sinaimg.cn/large/006Ejijoly1g9mvsp8vnuj30du09ujs7.jpg"/></div>
---

> <font color=blue>**ARC I 和ARC II**</font>：[Convolutional Neural Network Architectures for Matching Natural Language Sentences](https://arxiv.org/pdf/1503.03244.pdf) [2014]

（1）ARC I模型结构

- 输入两个语句$S_x$和$S_y$，获取语句中每个单词的Embedding，输出`[B, L, D]`；
- 经过多层的CNN和max-pooling的结果，将得到的结果展开并拼接`[B, *]`；
- 使用Dropout，再将结果经过MLP得到最终的输出。

<img src="http://ww1.sinaimg.cn/large/006Ejijoly1g9mtouox0jj30fq08vmy9.jpg"/>


（2）ARC II模型结构

- 输入两个语句$S_x$和$S_y$，获取语句中每个单词的Embedding，输出`[B, L, D]`；
- 经过一层的CNN得到两个句子的组合表征，输出`[B, L, F1]`；
- 计算匹配矩阵，输出`[B, L, R, F1]`；
- 经过多层二维的卷积，输出`[B, F2, L//P, R//P]`；
- 将上面的张量展开，经过Dropout之后输入到MLP中得到最终的输出。

<img src="http://ww1.sinaimg.cn/large/006Ejijoly1g9mtpy903rj30r40a8dha.jpg"/>