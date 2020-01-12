---
layout:     post
title:      论文笔记
subtitle:   SGM Sequence Generation Model for Multi-Label Classification
date:       2020-01-12
author:     dhx20150812
header-img: img/post-bg-ios9-web.jpg
catalog: true
mathjax: true
tags:
    - NLP
    - 文本分类
    - 多标签分类
---


# SGM: Sequence Generation Model for Multi-Label Classification


> 来自COLING 2018，[https://www.aclweb.org/anthology/C18-1330.pdf](https://www.aclweb.org/anthology/C18-1330.pdf)
>
> 代码已开源，[https://github.com/lancopku/SGM](https://github.com/lancopku/SGM)


## Motivation

作者认为现有的模型通常存在两点不足:

1.  忽略掉标签之间的关联；
2.  文本的不同部分对预测标签来说有不同的贡献

于是作者也将多标签分类建模为一个序列预测问题，同时引入了新的解码器结构。

我们先给出这篇文章的问题定义：

定义标签集合 $\mathcal{L}=\left\{l_{1}, l_{2}, \cdots, l_{L}\right\}$，它共有 $L$ 个标签。给定一个有 $m$ 个词的文本序列 $\boldsymbol{x}$，多标签分类的任务就是为文本 $\boldsymbol{x}$ 分配一个有 $n$ 个标签的子集 $\boldsymbol{y}$。于是从序列建模的角度来说，多标签分类的目标就是最大化如下的条件概率：


$$
p(\boldsymbol{y} | \boldsymbol{x})=\prod_{i=1}^{n} p\left(y_{i} | y_{1}, y_{2}, \cdots, y_{i-1}, \boldsymbol{x}\right)
$$

## Overview

SGM 模型的总体概况如下所示：

<img src="https://note.youdao.com/yws/api/personal/file/WEB70f1c0deebc2e4c2b018f62620441b95?method=download&amp;shareKey=a44df800c2283a6a0983eb3ac804d47a" alt="image-20200112230700925" style="zoom:67%;" />

首先，我们先将每个文本对应的标签按照他们在训练集中的频次排序。高频次的标签排在前面。同时将 $bos$ 和 $eos$ 插入到标签序列的头部和尾部中。

文本 $\boldsymbol{x}$ 首先被编码为隐层状态，然后通过注意力机制聚合到 $t$ 时刻的上下文向量中。解码器将上下文向量 $c_t$ 、上一时刻的隐层状态 $s_{t-1}$ 和上一时刻的输出的 embedding 向量 $\boldsymbol{y}_{t-1}$ 作为输入，产生这一时刻的隐层状态 $s_t$。

### Encoder

作者使用了双向 LSTM 来读取输入文本 $\boldsymbol{x}=\{\boldsymbol{x}_1,\boldsymbol{x}_2,\cdots,\boldsymbol{x}_m\}$，如下所示：

$$
\begin{array}{l}{\overrightarrow{\boldsymbol{h}}_{i}=\overrightarrow{\mathrm{LSTM}}\left(\overrightarrow{\boldsymbol{h}}_{i-1}, \boldsymbol{x}_{i}\right)} \\ {\overleftarrow{\boldsymbol{h}}_{i}=\overleftarrow{\operatorname{LSTM}}\left(\overleftarrow{\boldsymbol{h}}_{i+1}, \boldsymbol{x}_{i}\right)}\end{array}
$$

然后将前后向的隐层状态拼接起来得到 $t$ 时刻的隐层状态 $\boldsymbol{h}_{i}=\left[\overrightarrow{\boldsymbol{h}}_{i} ; \overleftarrow{\boldsymbol{h}}_{i}\right]$，它融合了第 $i$ 个词周围的序列信息。

### Attention

$t$ 时刻关于第 $i$ 个词的注意力权重 $\alpha_{ti}$ 由如下计算得到：

$$
\begin{aligned} e_{t i} &=\boldsymbol{v}_{a}^{T} \tanh \left(\boldsymbol{W}_{a} \boldsymbol{s}_{t}+\boldsymbol{U}_{a} \boldsymbol{h}_{i}\right) \\ \alpha_{t i} &=\frac{\exp \left(e_{t i}\right)}{\sum_{j=1}^{m} \exp \left(e_{t j}\right)} \end{aligned}
$$

而 $\boldsymbol{W}_{a}$、$\boldsymbol{U}_{a}$ 和 $\boldsymbol{v}_{a}$ 都是需要学习的参数。$\boldsymbol{s}_{t}$ 是解码器在当前时刻的隐层状态。这一时刻的上下文向量由如下计算得到：

$$
c_{t}=\sum_{i=1}^{m} \alpha_{t i} \boldsymbol{h}_{i}
$$

### Decoder

$t$ 时刻解码器的隐层状态由如下计算：

$$
s_{t}=\operatorname{LSTM}\left(s_{t-1},\left[g\left(\boldsymbol{y}_{t-1}\right) ; \boldsymbol{c}_{t-1}\right]\right)
$$

其中 $\left[g\left(\boldsymbol{y}_{t-1}\right) ; \boldsymbol{c}_{t-1}\right]$ 意味着将向量 $g\left(\boldsymbol{y}_{t-1}\right)$ 和 $\boldsymbol{c}_{t-1}$ 拼接起来。$g\left(\boldsymbol{y}_{t-1}\right)$ 是在分布$\boldsymbol{y}_{t-1}$ 下最大概率的标签的 embedding 向量。$\boldsymbol{y}_{t-1}$ 是 $t-1$ 时刻在标签集合 $\mathcal{L}$ 下的概率分布，它由如下计算得到：

$$
\begin{aligned} \boldsymbol{o}_{t} &=\boldsymbol{W}_{o} f\left(\boldsymbol{W}_{d} \boldsymbol{s}_{t}+\boldsymbol{V}_{d} \boldsymbol{c}_{t}\right) \\ \boldsymbol{y}_{t} &=\operatorname{softmax}\left(\boldsymbol{o}_{t}+\boldsymbol{I}_{t}\right) \end{aligned}
$$

其中，$\boldsymbol{W}_{o}$、$\boldsymbol{W}_{d}$ 和 $ \boldsymbol{V}_{d}$ 是需要学习的参数。$\boldsymbol{I}_{t}$ 是 mask 向量，它的作用是防止已经预测过的标签再次被预测到：

$$
\left(\boldsymbol{I}_{t}\right)_{i}=\left\{\begin{array}{ll}{-\infty} & {\text { if the label } l_{i} \text { has been predicted at previous } t-1 \text { time steps. }} \\ {0} & {\text { otherwise. }}\end{array}\right.
$$

### Global Embedding

上述解码器中的 embedding 向量 $g\left(\boldsymbol{y}_{t-1}\right)$  是在分布 $\boldsymbol{y}_{t-1}$ 下最大概率的标签的 embedding 向量。这样的做法是贪心的。可是，这个计算只是贪心的利用了 $\boldsymbol{y}_{t-1}$ 的最大值。在论文提出的 SGM 模型中，基于先前预测的标签来产生下一个标签。因此，如果在第 $t$ 时刻得到了错误的预测，然后就会在预测下一个标签的时候得到了一个错误的后继标签，这也叫做 exposure bias。

受到 Highway Netework 中 adaptive gate 的启发，作者引入一个全新的global embedding。定义 $\boldsymbol{e}$ 是在分布 $\boldsymbol{y}_{t-1}$ 下最大概率的标签的 embedding 向量，在之前的做法中，$g\left(\boldsymbol{y}_{t-1}\right)=\boldsymbol{e}$，而 $t$ 时刻的 weighted average embedding 向量如下：

$$
\bar{e}=\sum_{i=1}^{L} y_{t-1}^{(i)} e_{i}
$$

于是传入到 $t$ 时刻解码器的global embedding如下所示：

$$
g\left(\boldsymbol{y}_{t-1}\right)=(\mathbf{1}-\boldsymbol{H}) \odot \boldsymbol{e}+\boldsymbol{H} \odot \overline{\boldsymbol{e}}
$$

其中，$\boldsymbol{H}$ 是transform gate，它控制了weighted average embedding 的比例：

$$
\boldsymbol{H}=\boldsymbol{W}_{1} \boldsymbol{e}+\boldsymbol{W}_{2} \bar{e}
$$

$\boldsymbol{W}_{1}$ 和 $\boldsymbol{W}_{2}$ 是权重矩阵。此时 $\boldsymbol{y}_{t-1}$ 融合了所有可能的标签的信息，减小了之前的错误预测带来的损害。这使得模型更加鲁棒。

## Experiments

先介绍数据集——

| Dataset | total samples | label sets | words/samples | labels/samples |
| :-: | :-: | :-: | :-: | :-: |
| RCV1 | 804414 | 103 |123.94|3.24|
| AAPD | 55840 | 54 |163.42|2.41|

评价标准是Hamming-loss 和 Micro-F1。

作者设置的baseline由BR、CC、LP、CNN和CNN-RNN，在两个数据集上的结果分别如下：

![image-20200113002216410](https://note.youdao.com/yws/api/personal/file/WEB99988352bd4995c6a4a97793e9d574cf?method=download&shareKey=43fb7b3b011e336d0ce299931c7c89ff)

在 RCV1 数据集上，使用了 global embedding 的 SGM 模型比 BR 模型降低了约12.79%的Hamming loss，同时提高了2.33%的micro-F1。同时比其他的深度学习方法都降低了很大的 Hamming loss 和提高了 micro-F1。即使没有使用 global embedding 的 SGM 模型，也同样超越了baseline。同样可以在 AAPD 数据集上观察到相似的效果。


### Global Embedding 的影响

为了探索 global embedding 对于最终分类结果的影响，作者在 RCV1 数据集上进行了实验，此时将最终的 embedding 设置为：

$$
g\left(\boldsymbol{y}_{t-1}\right)=(1-\lambda) * e+\lambda * \bar{e}
$$

当增大 $\lambda$ 的值时，在 RCV1 数据集上的效果如下所示：

<img src="https://note.youdao.com/yws/api/personal/file/WEBadee1519c831ef2a576bec9b37c9669c?method=download&shareKey=b3020abe6e53777595e356a772057efd"  style="zoom:67%;" />

可见，使用 global embedding 的效果要优于不使用 global embedding的效果。在 $\lambda$ 增长时，模型的效果先提高，然后降低。

### Mask 和 排序对模型效果的影响

作者在两个数据集上

![image-20200113003951734](https://note.youdao.com/yws/api/personal/file/WEBa8d83aef88169486621d77c36e1aca27?method=download&shareKey=c60e8839cf52f2178394eea0e78405ca)
