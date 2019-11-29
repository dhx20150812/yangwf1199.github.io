# Seq2Emo for Multi-label Emotion Classification Based on Latent Variable Chains Transformation

> 来自AAAI 2020 [https://arxiv.org/pdf/1911.02147.pdf](https://arxiv.org/pdf/1911.02147.pdf)


## Motivation：

当前很多方法在做文本的情绪识别任务时，都将其建模为一个多标签分类(`Multi-label Classifiction`)问题，然后给网络最后一个输出层指定一个阈值，或者训练多个二分类器。

这样存在一些问题，首先，它忽略了情绪标签之间的关联性。比如，通常来说人们不会同时表达“高兴”和“愤怒”，但很可能“高兴”和“喜爱”会同时出现。

基于此，作者提出了隐变量链(`Latent Variable Chain`)的方法，它建模了情绪标签之间的关联性，同时提出了一个`Seq2Emo`的模型，它在考虑情绪标签之间关联性的同时预测了多个标签。

## Introduction

针对MLC问题的方法通常都含有大量的问题转换(`Problem Transformation`)，假设情绪集合为$\mathcal{L}=\left\{l_{1}, l_{2}, \cdots, l_{k}\right\}$，主要有如下三种：

- Binary Relevance，训练多个独立的二分类器，每个只用来判断一种情绪
- Classifier Chains，训练多个有关联的二分类器，考虑到了情绪标签之间的关联性
- Label Powerset，考虑了所有的标签组合的情况，因此共训练了$2^k$个分类器，当`k`很大时不适用

作者基于深度学习的隐层变量提出了`Latent Variable Chain`的方法，然后基于LVC提出了Seq2Emo这一模型。它可以捕获语义和情绪特征，然后使用双向的LVC来生成label。同时作者还收集了一个新的数据集`Balanced Multi-label Emotional Tweets`(BMLT)。

总结起来，本文的贡献有如下三点：
- 将多标签情绪分类建模为LVC问题
- 提出了新模型Seq2Emo
- 收集了新数据集BMLT


## Overview

本章将系统地介绍下BR和CC变换，同时如何在深度学习方法上使用这两种变换。

### MLC任务

MLC任务的形式化定义如下：给定一个文本$X=\{x_1,x_2,\dots,x_n\}$，其中$x_i$是一个词，$n$为文本的长度。每个文本都对应着一个目标$Y \subseteq \mathcal{L}$，符号$\subseteq$代表了每个$Y$都包含了$\mathcal{L}$中的多个元素，或为空集$Y=\emptyset$。

MLC模型的目标是学习到一个条件分布$\mathcal{P}(Y | X)$，$Y$是一个集合，其中的元素个数不一定是1

### BR Transformation

在BR的变换下，可以使用多个独立的二分类器来分别处理每种情绪。首先，目标$Y^{b}=\left(y_{1}, y_{2}, \cdots, y_{k}\right)$是一个0/1向量，当label set的大小为`k`时，需要训练`k`个独立的分类器。

设BR变换的分类器为$C_{j}^{B}(j \in[1 \cdots k])$。其中，分类器$C_{j}^{B}$只用作生成$y_i$。换言之，$C_j^B$需要学习到概率分布$\mathcal{P}\left(y_{i} | X\right)$，$Y^b$是通过所有的k个分类器共同得到的。

传统的分类器，如SVM、Naive Bayes等都可以用作BR变换的分类器，然而深度学习的方法不用`k`个分类器，它可以使用全连接层将隐层向量表示直接投影到标签层。全连接层的作用就是一个分类器，它采用编码器的输出作为输入，产生分类结果。那么`k`个分类器就可以共享编码器，然后使用不同的全连接层。

BR变换的分类器有两种变体，一种是在全连接层的输出有两个cell，另一种是只有一个cell。如下图所示

<img src="/Users/dinghanxing/Library/Application Support/typora-user-images/image-20191128154600703.png" alt="image-20191128154600703" style="zoom:50%;" />

两种变体的处理分别如下：

-   如上图左边所示，两个cell的输出经过softmax层，两个输出记作 $b_j^1$ 和 $b_j^2$，有 $y_{i} \triangleq \mathbb{1}\left(b_{j}^{0}<b_{j}^{1}\right)$

-   如上图右侧所示，引入阈值超参 $\tau$，输出经过sigmoid标准化，有$y_{i} \triangleq \mathbb{1}\left(b_{j}^{r}>\tau\right)$



### CC Transformation

同样的，CC变换也需要`k`个分类器。给定分类器 $C_j^C, j \in [1...k]$ ，原始的CC变换进行了`k`个二分类，每次的分类都基于上次的输出。使用二进制表示$Y^b$，这个变换可以被表示为如下的递归过程：
$$
y_j=C_j^C(X, y_{j-1})
$$
进入深度学习的时代之后，seq2seq模型被应用到各类任务上。它与CC变换的过程十分相似。它包括了两部分，一个Encoder和一个Decoder。

其中，Encoder将序列信息$X$压缩为向量或者向量表示：
$$
\boldsymbol{v}=Encoder(X)
$$
给定$\boldsymbol{v}$，decoder使用如下标签来预测目标$Y^b$：
$$
y_i=Decoder(\boldsymbol{v}, y_{j-1})
$$
CC变换有一个严重的问题。由于在测试阶段我们并不知道上一时刻的真实标签$y_{j-1}$，如果使用预测出的$\hat{y}_{j-1}$，则会导致训练阶段和测试阶段的不一致性。这就是`exposure bias`问题。



## Method

模型整体分为Encoder和Decoder，如图所示——

![image-20191128163021357](/Users/dinghanxing/Library/Application Support/typora-user-images/image-20191128163021357.png)

### Encoder

Encoder端使用了多层的双向LSTM，它将输入序列$X=[x_1, x_2,\cdots,x_n]$。作者结合使用了Glove和ELMo来更好地捕获上下文的语义信息。

#### Glove和ELMo

`Glv`是一个预训练的Glove模型，由Glove生成的词向量可以被表示为`Glv(x_t)`。`Glv`是一个矩阵，维度为$|V| \times D_{G}$。其中 $D_G$是Glove的Embedding size。$|V|$是词表大小。

同时，`ELMo`模型将句子$X$作为输入，生成一个矩阵，维度是$|n| \times D_{E}$。其中，$D_E$是 ELMo的embedding size，$|n|$是句子长度。

于是，$x_t$ 可表示分别为 $Glv(X_t)$ 和 $Elm_t^X$ 。

#### LSTM Encoder

在得到单词的向量表示之后，我们将两个词向量拼接，然后使用多层的双向LSTM来编码输入信息：
$$
h_{t}^{E}, c_{t}^{E}=\mathrm{bi}-\mathrm{LSTM}^{E}\left(\left[G l v\left(X_{t}\right) ; E l m_{t}^{X}\right],\left[h_{t-1}^{E} ; c_{t-1}^{E}\right]\right)
$$
$h_t^E$ 和 $c^E_t$ 分别代表了LSTM网络 $t$ 时刻的隐层状态和细胞状态。

#### Global Attention

将LSTM顶层的隐层状态记作$\overline{\boldsymbol{h}}^{E}=\left[\bar{h}_{1}^{E}, \bar{h}_{2}^{E}, \cdots, \bar{h}_{n}^{E}\right]$，作者在前向和后向解码时采用了一个单层单向的LSTM。记 $h_t^D$ 是解码器Decoder在$t$时刻的隐层状态。它会沿着如下过程更新：
$$
h_{t}^{D} \rightarrow \boldsymbol{\alpha}_{t} \rightarrow C T X_{t} \rightarrow \tilde{h}_{t}^{D}
$$
其中，$\alpha_{t}$ 是attention分数的向量，$CTX_t$ 是上下文向量，在每一时刻都会更新。它是编码器输出$\bar{h}^{E}$和attention分数$\alpha_t$的加权和。解码器隐层状态$h_t^D$的更新过程如下所示：
$$
\begin{aligned} \tilde{h}_{t}^{D} &=\tanh \left(W_{c}\left[C T X_{t} ; h_{t}^{D}\right]\right) \\ C T X_{t} &=\frac{\sum \alpha_{t} \bar{h}^{E}}{\sum \alpha_{t}} \\ \alpha_{t}(i) &=\frac{\exp \left(\text { score }\left(h_{t}^{D}, \bar{h}_{t}^{E}\right)\right)}{\sum_{j=1}^{n} \exp \left(\operatorname{score}\left(h_{t}^{D}, \bar{h}_{j}^{E}\right)\right)} \\ \text { score }\left(h_{t}^{D}, \bar{h}_{i}^{E}\right) &=h_{t}^{D \top} W_{a} \bar{h}_{i}^{E} \end{aligned}
$$

### Decoder




