&emsp;这个Repo主要记录自己学习NLP一些基础任务的历程以及代码实现，主要包括**文本分类、实体识别以及Aspect情感分析**，会随着学习的进程逐步更新代码。

---

# <font size=3>1. 文本分类任务</font>

&emsp;该部分主要基于两个数据集进行实验，一个是[IMDB评论数据集](https://www.kaggle.com/oumaimahourrane/imdb-reviews)，一个是[Yelps评论数据集](https://www.kaggle.com/z5025122/yelp-csv#yelp_academic_dataset_review.csv)。这两个都是Kaggle上别人选取的部分数据集，也可以自己准备。



> FastText

论文：[Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759)

&emsp;一种简单快速的文本分类方法，由Facebook在2016年提出，是一种适用于实际生产的方法。主要提出了将`n-gram`添加到词表，并结合之前提出的`n-char`的方法，比较好地解决了未登录词和生僻词的问题。为了防止词表过大导致OOM的问题，采用了Hash的方法将所有`n-gram`词Hash到固定大小的buckets中。



> TextCNN

论文：[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

&emsp;发表于2014年，提出将CNN用于文本分类，卷积层通过采用不同大小的滤波器模拟`n-gram`的效果（原论文滤波器大小为`3, 4, 5`），论文中不同大小的滤波器个数各为`100`，一定程度上保留了单词的顺序，同时利用CNN并行化的优势可以加速训练。此外，论文中还提出了`two-channel`的CNN，原理是将Word2Vec预训练的词向量作为第一层，训练过程中冻结住；另外随机初始化一层词向量，训练过程中不断更新。这样做的原因是可以在一定程度上避免overfitting，提高模型的泛化能力。此外，预训练的词向量只关注词形、语义上的相似性，而更新的词向量则可以关注和分类相关的单词的相似性，向量被注入了分类特征。



> CharCNN

论文：[Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626)

&emsp;发表于2016年，提出以`character`为基本单位对一句话进行分割，输入长度固定为$l_0$（原文中是1014），输入的字符表共有70个字符，包含26个英文字母，10个数字，33个其他字符以及换行符；在词表中的字符使用one-hot表示，不在词表中的单词以及空格用全零向量表示。模型分为large版和small版，都有9层，这里以small版为例。包含6层卷积层和3层全连接层，6层卷积层的`channel=256`，`kernel_size=7, 7, 3, 3, 3, 3`，`pool_size=3, 3, , , , 3 `；3层全连接层的大小为`1024, 1024, n_classes`。原文中还使用了同义词和短语对一些单词来进行替换，实现数据增强。



> BiLSTM

博客： [Understanding LSTM](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

&emsp;LSTM论文发表于1997年，论文较长，相较于文章，上面这篇博客的流传度反而更广，讲述比较清晰。文本可以理解为有序的数据，对于这种序列化的数据，LSTM这种类RNN可以可以通过隐状态保存一定的词序和语义上的相关性。BiLSTM则采用前向和后向两个方向，从两个方向理解语义关系，保留更多词序上的信息。代码实现采用了双层的BiLSTM，方法比较简单。输入word的最大长度为`200`，词向量长度为`200`，两层BiLSTM的隐层为`256`，最终经过一层线性层输出。