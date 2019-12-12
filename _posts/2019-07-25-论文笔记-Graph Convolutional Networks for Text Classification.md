---
layout:     post
title:      论文笔记
subtitle:   Graph Convolutional Networks for Text Classification
date:       2019-07-25
author:     dhx20150812
header-img: img/post-bg-ios9-web.jpg
catalog: true
mathjax: true
tags:
    - NLP
    - 文本分类
    - GCN
---

# Graph Convolutional Networks for Text Classification

## Introduction

文本分类是自然语言处理中一个很基础的问题。文本分类中的一个基本的中间步骤是文本表示，传统的方法使用人工提取的特征来表示文本，如词袋和n-gram等。深度学习的模型，如CNN，LSTM等考虑到了局部性和顺序性，他们可以很好地捕获局部连续的单词中的语义和语法信息，但可能会忽略语料库中的全局单词共现性。


然后作者引入了graph embedding的介绍。图形神经网络在被认为具有丰富关系结构的任务中是有效的，并且可以在图形嵌入中保留图形的全局结构信息。

> Graph neural networks have been effective at tasks thought to have rich relational structure and can preserve global structure information of a graph in graph embeddings. 

作者利用语料库构建了一个巨大的图，它将词和文档视为节点。作者使用了[Graph Convolutional Neural Networks](https://arxiv.org/pdf/1609.02907.pdf)来建模这个图。它可以捕获高阶的邻域信息。其中，两个单词节点之间的边由单词间的共生信息构建得到，单词和文档节点之间的边由词频和单词的文档频率得到。于是作者将文本分类的问题转化为了节点分类的问题。

这种方法可以使用一小部分的有标记文档实现不错的分类性能，并学习到可解释的单词和文档节点嵌入信息。

> The method can achieve strong classification performances with a small proportion of labeled documents and learn interpretable word and document node embeddings.


作者认为他们的工作有如下两点贡献：

* 提出了一种新的图神经网络方法做文本分类，据称这是第一个将整个语料库建模为异构图，并用图神经网络共同学习单词和文档嵌入的研究。
* 在多个基准数据集上的效果取得了state of the art，并且不用预训练的word embedding和外部知识。这一方法还可以自动学习预测词和文档的embedding向量。

## Method

### Graph Convolutional Networks

考虑一个图 $G = (V,E)$，其中 $V(\|V\| =n)$ 和 $E$ 分别表示节点和边的集合。每个节点都和自己相连。假设 $X \in \mathbb{R}^{n \times m}$ 是包含了 $n$ 个节点和它们的特征的矩阵，$m$ 是特征向量的维度，每一行 $x\_v \in \mathbb{R}^m$ 是 $v$ 的特征向量。然后引入一个邻接矩阵 $A$ 和它的度数矩阵 $D$ ，其中有 $D\_{ii}=\sum\_j A\_{ij}$。GCN可以通过一层的卷积操作捕获直接邻居的信息。当有多层GCN堆叠起来时，就可以整合更大邻域的信息。对于一层的GCN来说，新的 $k$ 维的节点特征矩阵 $L^{(1)} \in \mathbb{R}^{n \times k}$ 由如下得到：

$$
L^{(1)} = \rho(\tilde{A}XW_0)
$$

其中 $\tilde{A}=D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$ 是归一化的对称邻接矩阵，$W_0 \in \mathbb{R}^{m \times k}$ 是权重矩阵，$\rho$ 是激活函数。如上所述，通过堆叠多个GCN层，我们可以得到更高阶的邻域信息：

$$
L^{(j+1)} = \rho(\tilde{A}L^{(j)}W_j)
$$

其中 $j$ 代表了堆叠的层数，$L^{(0)}=X$。

### Text Graph Convolutional Networks(Text GCN)

作者建立了一个大型的异构文本图，它包括了词节点和文档节点，因此可以明确地建模词之间的共现性和调整图卷积。图的结构如下所示：

![图卷积结构](https://note.youdao.com/yws/api/personal/file/0C0F11000DBF4D689CEA3E3470A5D74E?method=download&shareKey=70880a9afaa08654e55de71cce691571)

文本图中的节点数目 $\|V\|$ 是文档数目（语料库大小）与单词数目（词表大小）之和。我们简单地将特征矩阵设置为单位矩阵 $X =I$，这意味着每个单词或文档都被表示为一个one-hot向量作为TextGCN的输入。我们基于文档中单词出现的频率和整个语料库中单词共现(word-word edges)在节点之间构建边。其中，单词和文档之间的边权重为该单词在该文档下的TF-IDF值；单词间的边权重为互信息PMI；最终，节点 $i$ 和节点 $j$ 之间的边权重定义为

$$
A_{ij}= \begin{cases} PMI(i,j) ~~~ i,j \text{ are words,}PMI(i,j)>0\\
TF-IDF_{ij}~~~ i\text{ is document,} j \text{ is word}\\ 1~~~ i = j\\0~~~ \text{otherwise}
\end{cases}
$$

单词对间的PMI值通过如下计算：

$$
PMI(i,j) = \log \frac{p(i,j)}{p(i)p(j)} \\
p(i,j) = \frac{\#W(i,j)}{\#W} \\
p(i) = \frac{\#W(i)}{\#W}
$$

其中，$\#W(i)$是滑动窗口中包含单词 $i$ 的数量，$\#W(i,j)$ 是滑动窗口中同时包含单词 $i$ 和单词 $j$ 的数量，$\#W$ 是语料库中滑动窗口的总数量。正的PMI值代表语料库中单词之间的高度语义相关性，而负的PMI则意味着极小的相关或不相关。因此，我们只添加那些具有PMI正值的单词节点。

在构建了文本图之后，我们将图送入到一个简单的两层GCN中，第二层的节点embedding与标签集大小相同。于是

$$
Z = softmax(\tilde{A}ReLU(\tilde{A}XW_0)W_1)
$$

损失函数定义为所有的标签文档上的交叉熵损失：

$$
L = -\sum_{d \in y_D}\sum_{f=1}^FY_{df}lnZ_{df}
$$

其中，$y\_D$ 是一组带有标签的文档索引，$F$ 是输出特征的维数，它等于类别的数目。$Y$是标签指示矩阵。

一个双层的GCN可以允许信息最多在节点间流动两步。因此，尽管图中文档间没有直接的边，但是双层的GCN允许信息在文档间流动。实验结果表明，双层的GCN表现比单层的好，但是更高的层数不会带来效果提升。
