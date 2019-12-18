---
layout:     post
title:      论文笔记
subtitle:   Label Specific Document Representation for Multi-Label Text Classification
date:       2019-12-18
author:     dhx20150812
header-img: img/post-bg-ios9-web.jpg
catalog: true
mathjax: true
tags:
    - NLP
    - 文本分类
    - 多标签分类
---

# Label-Specific Document Representation for Multi-Label Text Classification

>   来自EMNLP 2019，https://www.aclweb.org/anthology/D19-1044/

## Introduction

本文旨在解决文档级多标签分类的问题。作者认为，文档级多标签分类有如下难点：

（1）标签之间存在语义相关性，因为它们可能共享相同的文档子集

（2）文档可能很长，语义信息可能隐藏在嘈杂或冗余的内容中

（3） 大多数文档属于很少的标签，而大量的"尾部标签"只包含很少的正面文档

为了解决如上问题，作者认为应该重点关注如下问题：

（1）如何从原始文档中充分捕获语义模式

（2）如何从每个文档中提取与相应标签相关的判别信息

（3）如何准确地挖掘标签之间的关系

当然在MLTC任务上已经有很多的工作，但是他们大部分都没有考虑到标签之间的关系。最近也有很多工作关注了标签之间的关系，如DXML、SGM等。他们都取得了不错的效果，但是当标签之间的差异不大时，他们的效果还是不够好。

作者发现，在 MLTC 任务中，一个文档可以包含多个标签，并且每个标签可以作为文档的一个方面或部分，因此，整个文档的整体语义可以由多个部分组成。基于此，作者提出了一个全新的Label-Specific Attention Network（LSAN）模型。

为了从每个文档中捕获与标签相关的部分，作者采用 self-attention 机制来衡量每个词对每个标签的贡献。同时，LSAN 将标签文本表示为词向量，以便显式计算文档单词和标签之间的语义关系。之后，作者设计了一个辅助融合策略，从这两个方面提取信息，并构造每个文档的标签特定表示形式。

## Proposed Method

先给出模型的整体框架——

![image-20191218231615318](https://note.youdao.com/yws/api/personal/file/WEB4358d479ed563e2aac154362ed5156f1?method=download&shareKey=dc63a9310158f42bafa02750a9d80c34)

### 问题的形式化定义

记 $D=\{(x_i,y_i)\}\_{i=1}^N$ 为文档集合，它包含 $N$ 个文档，对应的标签集合为 $Y=\{y_i \in \{0,1\}^l\}$，$l$ 是标签的总个数。每个文档都是一个词的序列，而每个词都可以通过 word2vec 被表示为一个  $d$ 维的向量，令 $x_{i}=\left\{w_{1}, \cdots, w_{p}, \cdots, w_{n}\right\}$ 表示第 $i$ 个文档，$w_p \in \mathbb{R}^k$ 是文档中的第 $p$ 个词，$n$ 是文档中的词的数目。

与文档中的词相同，每个标签也可以被表示为词向量，那么标签的集合就可以被表示为矩阵 $C \in \mathbb{R}^{l \times k}$。给定输入文档及其关联的标签 $D$ ，MLTC的任务就是给每个文档分配最有可能的标签。

### 输入文档表示

作者使用biLSTM编码输入文档，$p$ 时刻的隐层状态为：

$$
\begin{array}{l}{\overrightarrow{h_{p}}=L S T M\left(\overrightarrow{h_{p-1}}, w_{p}\right)} \\ {\overleftarrow{h_{p}}=L S T M\left(\overleftarrow{h_{p-1}}, w_{p}\right)}\end{array}
$$

因此整个文档可以表示为：

$$
\begin{aligned} H &=(\vec{H}, \overleftarrow{H}) \\ \vec{H} &=(\overrightarrow{h_{1}}, \overrightarrow{h_{2}}, \cdots, \overrightarrow{h_{n}}) \\ \overleftarrow{H} &=(\overleftarrow{h_{1}}, \overleftarrow{h_{2}}, \cdots, \overleftarrow{h_{n}}) \end{aligned}
$$

整个文档集合被处理为一个矩阵 $H \in \mathbb{R}^{2k \times n}$。

### Label-Specific Attention Network

这部分的作用是从每个文档中确定与标签相关的部分。例如，对于句子“*June a fri- day, in the lawn, a war between the young boys of the football game starte*”，它的两个标签是“youth”和“sports”，显然“young boy”对应着“youth”，而“football game”对应着“sports”。
#### Self-Attention 机制

标签-词之间的注意力分数 $A^s \in \mathbb{R}^{l \times n}$ 由如下公式计算得到：

$$
A^{(s)}=\operatorname{softmax}\left(W_{2} \tanh \left(W_{1} H\right)\right)
$$

其中，$W_{1} \in \mathbb{R}^{d_{a} \times 2 k} $ 和 $W_{2} \in \mathbb{R}^{l \times d_{a}}$ 是需要训练的权重矩阵，$d_a$ 是超参。每一行 $A_j^{(s)}$ 代表了所有的词对第 $j$ 个标签的贡献。然后，通过这个注意力分数，可以获得每个标签的上下文词的线性组合：

$$
M_{j}^{(s)}=A_{j}^{(s)} H^{T}
$$

它可以作为输入文档关于第 $j$ 个标签的新的表示形式。所以在self-attention机制下的关于标签的文档表示矩阵是 $M^{(s)} \in \mathbb{R}^{l \times 2k}$。

#### Label-Attention 机制

在得到了标签和文档词的表达之后，我们可以显示地计算两者的关联性。最简单的做法就是计算 $\vec{h}_{p}$ 和 $C_j$ 之间的点积：

$$
\begin{array}{l}{\vec{A}^{(l)}=C \vec{H}} \\ {\overleftarrow{A}^{(l)}=C \overleftarrow{H}}\end{array}
$$

$\overrightarrow{A}^{(l)} \in \mathbb{R}^{l \times n}$ 和 $\overleftarrow{A}^{(l)} \in \mathbb{R}^{l \times n}$ 表示词与标签之间的前向和后向语义联系。与self-attention 机制相同，可以通过线性叠加得到关于标签的文档表示：

$$
\begin{array}{l}{\vec{M}^{(l)}=\vec{A}^{(l)} \vec{H}^{T}} \\ {\overleftarrow{M}^{(l)}=\overleftarrow{A}^{(l)} \overleftarrow{H}^{T}}\end{array}
$$

在Label-attention机制下关于标签的文档表示矩阵$M^{(l)}=\left(\vec{M}^{(l)}, \overleftarrow{M}^{(l)}\right) \in \mathbb{R}^{l \times 2 k}$

#### Adaptive Attention Fusion Strategy

作者采取了一个策略来利用得到的两种不同的文档表示，为了衡量两种注意力机制的重要性，作者计算了两个权重向量：

$$
\begin{array}{l}{\alpha=\operatorname{sigmoid}\left(M^{(s)} W_{3}\right)} \\ {\beta=\operatorname{sigmoid}\left(M^{(l)} W_{4}\right)}\end{array}
$$

$W_{3}, W_{4} \in \mathbb{R}^{2 k}$ 都是要训练的参数。$\alpha_j$ 和 $\beta_j$ 是代表了两种注意力机制在第 $j$ 个标签上的重要性。因此，需要给两个向量添加约束：

$$
\alpha_j + \beta_j = 1
$$

此时可以得到关于第 $j$ 个标签的最终的文档表示：

$$
M_{j} =\alpha_{j} M_{j}^{(s)}+\beta_{j} M_{j}^{(l)}
$$

关于所有标签的文档表示矩阵为 $M \in \mathbb{R}^{l \times 2k}$。

### 标签预测

每个文档的预测标签分布为：

$$
\hat{y}=\operatorname{sigmoid}\left(W_{6} f\left(W_{5} M^{T}\right)\right)
$$

其中，$W_{5} \in \mathbb{R}^{b \times 2 k}$，$W_{6} \in \mathbb{R}^{b}$ 是要训练的参数。

损失函数为：

$$
\mathcal{L}=-\sum_{i=1}^{N} \sum_{j=1}^{l}\left(y_{i j} \log \left(\hat{y}_{i j}\right)\right) +\left(1-y_{i j}\right) \log \left(1-\hat{y}_{i j}\right)
$$

