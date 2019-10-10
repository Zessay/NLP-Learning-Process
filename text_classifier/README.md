&emsp;该部分主要基于两个数据集进行实验，一个是[IMDB评论数据集](https://www.kaggle.com/oumaimahourrane/imdb-reviews)，一个是[Yelps评论数据集](https://www.kaggle.com/z5025122/yelp-csv#yelp_academic_dataset_review.csv)。这两个都是Kaggle上别人选取的部分数据集，也可以自己准备。



> <font size=3 color=red> **FastText**</font>

论文：[Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759)

&emsp;一种简单快速的文本分类方法，由Facebook在2016年提出，是一种适用于实际生产的方法。主要提出了将`n-gram`添加到词表，并结合之前提出的`n-char`的方法，比较好地解决了未登录词和生僻词的问题。为了防止词表过大导致OOM的问题，采用了Hash的方法将所有`n-gram`词Hash到固定大小的buckets中。



> <font size=3 color=red>**TextCNN**</font>

论文：[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

&emsp;发表于2014年，提出将CNN用于文本分类，卷积层通过采用不同大小的滤波器模拟`n-gram`的效果（原论文滤波器大小为`3, 4, 5`），论文中不同大小的滤波器个数各为`100`，一定程度上保留了单词的顺序，同时利用CNN并行化的优势可以加速训练。此外，论文中还提出了`two-channel`的CNN，原理是将Word2Vec预训练的词向量作为第一层，训练过程中冻结住；另外随机初始化一层词向量，训练过程中不断更新。这样做的原因是可以在一定程度上避免overfitting，提高模型的泛化能力。此外，预训练的词向量只关注词形、语义上的相似性，而更新的词向量则可以关注和分类相关的单词的相似性，向量被注入了分类特征。

![TextCNN](http://ww1.sinaimg.cn/large/006Ejijoly1g68i1saaclj310c0fwwh0.jpg)

> <font size=3 color=red>**CharCNN**</font>

论文：[Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626)

&emsp;发表于2016年，提出以`character`为基本单位对一句话进行分割，输入长度固定为`$l_0$`（原文中是1014），输入的字符表共有70个字符，包含26个英文字母，10个数字，33个其他字符以及换行符；在词表中的字符使用one-hot表示，不在词表中的单词以及空格用全零向量表示。模型分为large版和small版，都有9层，这里以small版为例。包含6层卷积层和3层全连接层，6层卷积层的`channel=256`，`kernel_size=7, 7, 3, 3, 3, 3`，`pool_size=3, 3, , , , 3 `；3层全连接层的大小为`1024, 1024, n_classes`。原文中还使用了同义词和短语对一些单词来进行替换，实现数据增强。

![CharCNN](http://ww1.sinaimg.cn/large/006Ejijoly1g68i0p0rzij30vc0963zx.jpg)

> <font size=3 color=red>**BiLSTM**</font>

博客： [Understanding LSTM](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

&emsp;LSTM论文发表于1997年，论文较长，相较原文，上面这篇博客的流传度反而更广，讲述比较清晰。文本可以理解为有序的数据，对于这种序列化的数据，LSTM这种类RNN可以可以通过隐状态保存一定的词序和语义上的相关性。BiLSTM则采用前向和后向两个方向，从两个方向理解语义关系，保留更多词序上的信息。代码实现采用了双层的BiLSTM，方法比较简单。输入word的最大长度为`200`，词向量长度为`200`，两层BiLSTM的隐层为`256`，最终经过一层线性层输出。


> <font size=3 color=red>**BiLSTM+Attention**</font>

论文：[Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Claasification](https://www.aclweb.org/anthology/P16-2034)

&emsp;论文发表于2016年，传统的BiLSTM只关注于最后一个输出向量，将其应用分类，这对于短文本来说效果还可以，但是如果文本较长，那么LSTM还是不容易捕获长期依赖。考虑到文本中的每个单词都有可能对文本分类有一定的贡献，而较长的语句容易出现信息丢失的情况。论文在BiLSTM输出向量后，增加一个Attention层，定义一个变量作为`query`，每个单词对应的输出作为`key`和`value`，计算`query`和所有`key`的点积，然后使用softmax归一化。用归一化的结果对不同单词的`value`进行加权求和。将最后得到的向量输入MLP中进行分类。

> <font size=3 color=red>**RCNN**</font>

论文：[Recurrent Convolutional Neural Networks for Text Classification](http://www.nlpr.ia.ac.cn/cip/~liukang/liukangPageFile/Recurrent%20Convolutional%20Neural%20Networks%20for%20Text%20Classification.pdf)

&emsp;论文发表于2015年，传统的机器学习文本分类方法（如BoW）丢失了文本的上下文语义和语境，`n-gram`方法能够一定程度上改进，但是使用了更多的表征的维度，消耗更大的内存。CNN4Text的方法通过使用不同的窗口捕捉文本的语序信息，不同的窗口表示不同大小`n-gram`的语义。本文使用BiLSTM来捕捉上下文的语义，当前单词$w_i$的前一个词$w_{i-1}$的前向输出$c_l$可以理解为单词$w_i$的上文语义，后一个词$w_{i+1}$的后向输出$c_r$可以理解为$w_i$的下文语义。将上下文语义和当前单词的词向量拼接，得到单词$w_i$的表征$[c_r(w_i); e(w_i); c_r(w_i) ]$。之后经过一个线性映射到低维，再在时间步维度方向求最大值，相当于`max-pooling`。最后，再经过一个输出层实现分类。

![RCNN](http://ww1.sinaimg.cn/large/006Ejijoly1g68hx4f00pj315c0hcgq8.jpg)


> <font size=3 color=red>**Adversarial LSTM**</font>

论文：[Adversarial Training Methods for Semi-Supervised Text Classification](https://arxiv.org/abs/1605.07725)

&emsp;论文发表于2017年，提出通过对原始词向量增加扰动的方式生成对抗样本，提升模型的泛化能力。传统使用`one-hot`编码单词的方式不适合增加扰动，因为词表征是高维稀疏向量，并且是离散的，增加扰动之后原始向量含义会发生比较大的变化。分布式词嵌入向量可以看做是连续的，虽然增加扰动之后大概率无法对应到某个单词，但是在一定程度上提升了模型的泛化能力。**以词频作为各个词的权重，计算词向量的均值和方差**，通过均值和方差对词向量做规范化得到新的词向量。根据正常计算的loss对词向量计算梯度，由梯度得到扰动值，将扰动的词向量作为输入，得到对抗损失。（**对抗损失时不参与梯度计算的，即LSTM层和Dense层的Weight和Bias是reuse之前的**）。将两个损失相加，对参数进行更新。

![Adversarial LSTM](http://ww1.sinaimg.cn/large/006Ejijoly1g68i39mnngj312k0bygmt.jpg)

> <font size=3 color=red>**HAN**</font>

论文：[Multilingual Hierarchical Attention Networks for Document Classification](https://arxiv.org/pdf/1707.00896.pdf)

&emsp;论文发表于2017年。由于传统的RNN网络的记忆能力优先，很难捕捉长距离的依赖。于是，本文提出了**用于分类的分层注意力网络**，采用层叠式的RNN，对文本的句子级别进行encoder，引入注意力机制，解决长句子的依赖问题。HAN包含三层，分别是**词汇层、句子层和输出层**，其中词汇曾和句子曾都包含一层Encoder和一层Attention层。在词汇层，HAN将文本分为`$K$`个句子，每个句子的长度为`$T$`，然后基于单词使用双向GRU进行Encoder，并使用Attention，得到句向量；在句子层，同样使用双向GRU进行Encoder，并使用Attention关注不同句子的贡献度，得到文本向量。最后，将文本向量用于输出层进行分类。

&emsp;此外，为了应对多语言训练容易出现参数量过大的问题，本文提出通过Encoder层和Attention层的参数共享，并且使用多标签训练的参数避免参数量过大的问题。为了避免在训练过程中，对某种语言存在偏向，在每个batch_size中进行采样，每种语言的采样数量的`batch_size / M`，这里`M`表示语言类别数。

<div align=center><img width="500" height="400" src="http://ww1.sinaimg.cn/large/006Ejijoly1g69tw798nmj30f00c83za.jpg"/></div>
> <font size=3 color=red>**DPCNN**</font>

论文：[Deep Pyramid Convolution Neurail Networks for Text Categorization](https://ai.tencent.com/ailab/media/publications/ACL3-Brady.pdf)

&emsp;论文发表于2017年。许多实验表明，浅层的CNN网络在文本分类中已经能够得到比较好的效果，这主要是因为深层CNN容易出现梯度消失的情况，并且由于feature map数量的增加，计算复杂度也会随之增加。本文提出了一种在不增加复杂度的前提下，能够得到更高分类精度的深层CNN网络。首先使用传统的word-embedding得到每个词的向量，然后使用[region embedding](http://riejohnson.com/paper/cnn-semi-nips15.pdf)得到区域文本向量（也就是采用固定核大小的卷积层）；紧接着使用堆叠的卷积块，买两个卷积之后进行一个shortcut，卷积块的最前面采用`size=3, strides=2`的池化层，减小计算复杂度。由于经过一层层赤化之后，文本的长度不断减小，但是通道数保持不变，所以称为`Pyramid`。文章采用了一些技巧，比如`pre-activation`以及固定滤波器的数量，这样保证了准确率，同时减小了复杂度。


<div align=center><img width="400" height="500" src="http://ww1.sinaimg.cn/large/006Ejijoly1g6dffxlhg1j30gc0jeac6.jpg"/></div>