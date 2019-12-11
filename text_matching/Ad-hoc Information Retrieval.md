&emsp;信息检索是指**基于检索关键词或者关键句从一个固定的候选集合中返回匹配度较高的文档**。

---

> <font color=blue>**HiNT**</font>：[Modeling Diverse Relevance Patterns in Ad-hoc Retrieval](https://arxiv.org/pdf/1805.05737.pdf) [2018]

（1）本文属于对DeepRank的集成和发展，提出基于不同Passage的分层匹配模型。首先将文章划分为不同的Passage，基于不同的Passage计算局部相关性分数，对局部相关性分数进行融合，得到全局相关性分数。

（2）模型：

> **局部匹配层**：计算query和每一个passage的相关性向量：

- 基于term的query-passage对，计算精确匹配矩阵$M^{xor}$和相似度匹配矩阵$M^{cos}$；
- 进一步融合term importance，将每一个元素$M_{ij}$的最后一维，添加对应的query word和doc word的压缩语义表征；
- 基于两个query-passage的张量，使用spatial GRU得到相似性匹配向量；
- 使用双向的Spatial GRU并将两个Spatial GRU的输出进行concat，得到最终的局部表征。

> **全局决策层**：将Passage的局部表征进行组合。

- 独立决策模型：基于上面得到的query-passage对相关性表征，基于每个维度进行k-max pooling，拼接成一维向量，输入MLP；
- 累积决策模型：基于上面得到的query-passage对经过双向LSTM进行相关性向量的表达增强，基于每个维度进行k-max pooling，拼接成一维向量，输入MLP；
- 将独立决策模型和累积决策模型进行组合。

<div align=center><img src="http://ww1.sinaimg.cn/large/006Ejijoly1g9m6g6e7gzj30e107tdga.jpg"/></div>
---

> <font color=blue>**DeepRank**</font>：[DeepRank: A New Deep Architecture for Relevance Ranking in Information Retrieval](https://arxiv.org/pdf/1710.05649.pdf) [2017]

（1）本文描述了常规标注人员对文档相关性进行标注的流程：

- 相关位置检测；
- 确定局部相关性；
- 聚合局部相关性输出相关性标签。

&emsp;根据标注的流程，分别对应不同的处理过程，得到最终的网络结构。

（2）模型包含3个部分，分别是**检测策略，度量网络以及聚合网络**

- 针对每个单词在doc中出现的位置，设置一定的窗口大小，提取query-centric context，得到一系列单词序列，每一个表示局部相关位置信息；
- 给定一个query以及quer-centric context，通过相似度计算的方式，得到匹配矩阵；
- 为了进一步组合query/query-centric context，将每一个元素进行扩展，拼接对应位置的`query`和`query-centric context`单词的向量；
- 基于上面得到的张量，选择使用CNN或者2D的GRU计算局部相关性向量；
- 将上面得到的相关性向量按照query-term分组，加上中心词在document中的位置信息，使用RNN得到每个term的全局相关性表征；
- 基于term级别的相关性表征，使用term gating网络，将全局向量融合得到score。

<div align=center><img src="http://ww1.sinaimg.cn/large/006Ejijoly1g9m365qxetj30vm0duac4.jpg"/></div>
---

> <font color=blue>**DRMM-TKS**</font>：[A Deep Relevance Matching Model for Ad-hoc Retrieval (*A variation of DRMM)](https://link.springer.com/chapter/10.1007/978-3-030-01012-6_2) [2017]

（1）模型结构

- 输入query和document，对应的形状分别是`[B, L]`和`[B, R]`；通过embedding映射得到对应的词向量矩阵信息`[B, L, D]`和`[B, R, D]`；
- 计算匹配矩阵`[B, L, R]`，获取每一行中topk的值，得到`[B, L, K]`
- 根据query的词向量矩阵`[B, L, D]`以及对应的mask，得到每个位置的attention值；
- 将第二步得到的`[B, L, K]`经过全连接层得到`[B, L]`；
- 根据attention值和全连接的输出得到`[B, 1]`，再经过一个输出层得到最终的分数。

---

> <font color=blue>**Co-PACRR**</font>：[A Context-Aware Neural IR Model for Ad-hoc Retrieval](https://arxiv.org/pdf/1706.10192.pdf) [2017]

（1）本文提出了3种针对PACRR的改进方法：**通过context信息进行消歧，级联k-max pooling以及打乱组合顺序**。

（2）模型结构：

- 基于输入的query和document，经过word2vec的映射，得到`[B, L, D]`和`[B, R, D]`；
- 根据query和document计算相似度矩阵，得到`[B, L, R]`；根据document的滑动窗口context信息以及query的句向量，计算得到querysim，维度为`[B, R]`；
- 将相似度匹配矩阵经过几个2D的CNN结构，每个CNN的长度为$l_g \in [2, l_g]$得到多个CNN的输出，每个输出的维度是`[B, C, L, R]`；
- 基于每一种滤波器进行max-pooling，加上原始的sim矩阵，得到$l_g + 1$个矩阵，维度为`[B, 1, L, R]`；
- 对于每个matrix的每一行，获取前$n_s$个最大的值，执行一次得到`[B, L, NS]`；根据级联k-max pooling的方法，对不同长度的document进行ns-max pooling，假设定义了$n_c$中不同的长度，于是得到的输出维度为`[B, L, LG, NS*NC]`；与此同时，计算`querysim`向量中对应$n_c$位置的$n_s$个最大值，拼接在每一个张量的最后一个维度，于是得到的张量维度为`[B, L, LG, 2*NS*NC]`。
- 最后，将每一个query term经过归一化的IDF值拼接到后面；经过打乱rows之后，得到张量维度为`[B, L, LG, 2*NS*NC+1]`；
- 将上面的结果输入多层前向网络，输出得到最终的score。

<div align=center><img src="http://ww1.sinaimg.cn/large/006Ejijoly1g9luurbgm0j30tq09jgna.jpg"/></div>
---

> <font color=blue>**Duet**</font>：[Learning to Match using Local and Distributed Representations of Text for Web Search](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/wwwfp0192-mitra.pdf) [2017]

（1）本文总结了高效的检索模型需要考虑的3个问题：**精确的term匹配，匹配位置，不精确的term匹配**。本文特色是将传统的离散表征和分布式表征的结果结合。

（2）模型分为Local Model和Distributed Model两个并行的模型

> **Local Model**

- 限制query长度为10，限制document的长度为1000，计算每个位置的单词是否精确匹配，得到匹配矩阵`[B, R, L]`；
- 经过一个一维的CNN，得到`[B, C, L]`；
- 经过一个全连接层，得到`[B, C]`；再经过一个全连接层，形状保持不变；
- Dropout之后再经过全连接层得到Local Model的输出`[B, 1]`。

> **Distributed Model**

- 将query和document中每个单词转化为n-gram向量得到`[B, L, D]`；
- 分别经过一个一维的CNN层，得到`[B, D, C]`；
- query使用激活函数激活，并经过max-pooling，得到`[B, D]`；
- document使用激活函数激活，经过特定大小的二维pooling，得到`[B, D, E]`；
- 将query和document的结果按元素乘积，得到`[B, D, E]`；
- 展开，得到`[B, DxE]`，经过多层感知机，dropout之后再经过线性层得到最终的输出`[B, 1]`。

将上面得到的两个结果相加，再经过一个输出层，得到最终的score。

<div align=center><img src="http://ww1.sinaimg.cn/large/006Ejijoly1g9ln35wzj1j309r0hx3zj.jpg"/></div>
---

> <font color=blue>**Conv-KNRM**</font>：[Convolutional Neural Networks for So-Matching N-Grams in Ad-hoc Search](http://www.cs.cmu.edu/~zhuyund/papers/WSDM_2018_Dai.pdf) [2018]

（1）本文的思想和KNRM是一样的，只不过是使用CNN基于不同的卷积核得到多个特征向量，并将这些特征向量进行组合，通过线性层得到匹配分数。

（2）给定一个输入query为$q$，以及文档用$d$表示

- **Word Embedding**：将每个单词映射成一个$L$维的向量，分别query和document的矩阵表示为$T_q$和$T_d$；
- **CNN**：使用CNN组合n-gram的信息：
    - 对于每一个包含$h$个单词的窗口，滤波器将里面所有元素的值基于filter中的权重相加，得到一个值；
    - 使用$F$个不同的滤波器就可以得到$F$个不同的值，每个滤波器观察的维度不同；
    - 对于一个$\overset{\rightarrow}{g}_i^h$，其中第$f$个元素就表示第$f$个滤波器的分数，也就是说一个位置的向量经过同大小的多个滤波器得到了多维的向量；
    - 使用多个不同kenel的CNN滤波器，则将文档的词向量矩阵$T$，转化成了h-gram的Embedding$G^h$。
- **cross-match层**：将不同长度query的n-grams向量和document的n-gram向量进行组合，得到每个n-gram级别的匹配矩阵，匹配矩阵的每个元素是向量的余弦相似度。当然，bigram的query组合向量，也可以和trigram的document向量进行组合，最终得到了$h^2_{\max}$个匹配矩阵。
- **Kernel Pooling**：和KNRM一样，针对每个匹配矩阵都使用Kernel-Pooling的方法，于是得到了$h^2_{\max}$个维度为$K$的向量，将它们进行拼接，得到$K \times h^2_{\max}$维度的向量；
- **Learning-to-Rank**：使用线性层，以`tanh`为激活函数，得到最终的匹配分数。

<div align=center><img src="http://ww1.sinaimg.cn/large/006Ejijoly1g9kzyj5b3lj30f3093wg7.jpg"/></div>
---

> <font color=blue>**KNRM**</font>：[End-to-End Neural Ad-hoc Ranking with Kernel Pooling](https://arxiv.org/pdf/1706.06613.pdf) [2017]

（1）和上面的DRMM相似，仍然是针对相关性模型对精确匹配单词的关注度做文章，只不过采用了更加巧妙和泛化的方法。

（2）模型结构

- 首先，将query和document的单词映射成词向量；
- 计算词向量之间的相似度，得到$n \times m$的矩阵，每一行表示一个query单词与所有document单词的相似度；
- 使用不同的RBF Kernel对每一行的相似度进行统计，得到$n \times k$的张量；
- 计算每个维度上的对数和，得到k维向量；
- 经过线性层并使用tanh激活函数，得到最终的相似度值。

<img src="http://ww1.sinaimg.cn/large/006Ejijoly1g9ky763i02j30g30amdhj.jpg"/>

---

> <font color=blue>**DRMM**</font>：[A Deep Relevance Matching Model for Ad-hoc Retrieval](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf) [2016]

（1）传统的文本匹配模型都是基于语义相似度的匹配，而检索任务中的匹配通常是**相关性匹配**，于是提出了本文的模型。

（2）模型结构：

- 对`query`中的每一个词，和`document`中的所有词计算相似度，得到局部交互值；
- 将每个词的局部交互值映射到匹配直方图中，比如余弦相似度的取值范围是`[-1, 1]`，基于固定的`bin_size`划分多个bins，计算每个bins中的数量组成定长的向量；（除了Count，文中还有Normalized Count和LogCount两种方式）；
- 得到匹配直方图向量，经过多层前向网络得到每一个`term`和`document`的匹配分数；
- 通过Term Gating Network计算每一个单词的权重（文中使用了基于词向量和IDF两种权重计算方式），对上面得到的每个词的匹配分数进行加权，得到最终的匹配分数。

<div align=center><img src="http://ww1.sinaimg.cn/large/006Ejijoly1g9dzghyqgoj30ms0e2gox.jpg"/></div>
---

> <font color=blue>**CDSSM**</font>：[Learning Semantic Representations Using Convolutional Neural Networks for Web Search](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/www2014_cdssm_p07.pdf) [2014]

（1）模型结构：

- 首先是word-hashing层将每一个单词转化成基于tri-gram的字符集表示；
- 接着使用卷积层提取局部的语义表征；
- 基于每个特征维度进行最大池化，得到整个语句的全局特征向量；
- 最后经过一个线性层得到输入单词序列的高级语义特征向量；
- 基于得到的语义特征向量，使用余弦相似度计算`query`和`document`的距离。

<div align=center><img src="http://ww1.sinaimg.cn/large/006Ejijogy1g7t7wfk1k3j30f70bg76c.jpg"/></div>
---

> <font color=blue>**DSSM**</font>：[Learning Deep Structured Semantic Models for Web Search using Clickthrough Data](https://posenhuang.github.io/papers/cikm2013_DSSM_fullversion.pdf) [2013]

（1）模型结构：

- 先把`query`和`document`转换为BOW向量形式
- 然后通过`word hashing`变换做降维得到相对低维的向量（**除了降维之外，`word hashing`还可以很大程度上解决单词形态和OOV对匹配效果的影响**）
- 将降维得到的向量，喂给MLP网络，输出层对应的低维向量就是`query`和`document`的语义向量（设为`Q`和`D`）
- 计算`(D, Q)`的`cosine similarity`后，用`softmax`做归一化得到的概率值是整个模型的最终输出，将该值作为监督信号进行有监督训练。

<div align=center><img src="http://ww1.sinaimg.cn/large/006Ejijogy1g7s10rcxlej30mz0ad0u1.jpg"/></div>
