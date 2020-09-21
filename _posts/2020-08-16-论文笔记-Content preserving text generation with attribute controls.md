---
layout:     post
title:      论文笔记
subtitle:   Content preserving text generation with attribute controls
date:       2020-08-16
author:     dhx20150812
header-img: img/post-bg-ios9-web.jpg
catalog: true
mathjax: true
tags:
    - NLP
    - 文本风格迁移
    - 可控文本生成
---

# Content preserving text generation with attribute controls

>   来自NIPS 2018 https://papers.nips.cc/paper/7757-content-preserving-text-generation-with-attribute-controls.pdf
>
>   代码已开源，第三方实现见 https://github.com/seanie12/CPTG

## 摘要

这篇文章想要解决的问题是修改句子中的属性。给定一个输入的句子和一系列标签，目标是生成与条件信息兼容的句子。为了保证内容的一致性，作者引入了重构损失，它融合了自编码损失和回译损失。同时作者还引入了对抗损失来保证生成的句子具有属性兼容性和真实性。

## 任务描述

假设共有 $K$ 个属性 $\{a\_{1}, \ldots, a\_{K}\}$。现有一个标记的句子集合 $D=\{\left(x^{n}, l^{n}\right)\}\_{n=1}^{N}$，$ l^n$是子集中的属性集合。给定一个句子 $x$ 和属性值 $l^{\prime}=\left(l\_{1}, \ldots, l\_{K}\right)$，我们的目标是生成一个句子，它有与 $x$ 相同的内容，但是却是由 $l^{\prime}$ 指定其属性值。在这个任务中，我们将内容定义为句子中未被属性捕获的信息。 我们使用属性向量来指代属性标签的二进制向量表示，它是由属性标签的one-hot向量表示拼接而来。

## 模型概述

将生成式模型定义为 $G$，$G$ 的目标是生成一个句子，它与输入句子的内容近似，然后属性一致。作者将其设计为一个编码-解码的模型 $G=\left(G_{\mathrm{enc}}, G_{\mathrm{dec}}\right)$。编码器是一个RNN，它以句子 $x$ 为输入，然后产生该句子的内容表示 $z_{x}=G_{\text {enc }}(x)$。给定一系列属性值 $l^{\prime}$，解码器RNN将基于$z_x$和$l^{\prime}$生成 $y \sim p_{G}\left(\cdot \mid z_{x}, l^{\prime}\right)$。

### 内容一致性

我们设计了两种重建损失，来鼓励内容一致性——

（1）自编码损失。$x$ 是一个句子，其对应的属性向量是 $l$。令 $z_{x}=G_{\mathrm{enc}}(x)$ 是句子 $x$ 的编码表示。由于在分布$G\left(\cdot \mid z_{x}, l\right)$下句子 $x$ 应该有很高的概率，因此我们使用自编码损失来强制执行此约束

$$
\mathcal{L}^{a e}(x, l)=-\log p_{G}\left(x \mid z_{x}, l\right)
$$

（2）回译损失。考虑 $l^{\prime}$，它是一个与 $l$ 不同的属性向量。令 $y \sim p_{G}\left(\cdot \mid z_{x}, l^{\prime}\right)$是以 $x$ 和 $l^{\prime}$ 为条件的生成的句子。假设有一个训练好的模型，采样得到的句子 $y$ 会保留句子 $x$ 的内容。因此，在分布 $p_{G}\left(\cdot \mid z_{y}, l\right)$ 下句子 $x$ 有很高的概率。其中，$z_{y}=G_{\mathrm{enc}}(y)$ 是句子 $y$ 的编码表示。因此回译损失中如下：

$$
\mathcal{L}^{b t}(x, l)=-\log p_{G}\left(x \mid z_{y}, l\right)
$$

自编码损失的缺陷是，在自回归模型中，模型只是简单地复制输入序列而不捕获任何潜在表示的信息特征。通常考虑使用降噪方法，通过删除，交换或重新排列单词将噪声引入输入序列。另一方面，在训练的早期阶段，生成的样本 $y$ 的内容可能与 $x$ 的内容不匹配，从而导致回译损失可能会误导生成器。

我们通过融合两个潜在表示来合并自编码损失 $z_{x}$ 和回译损失 $z_{y}$。令$z_{x y}=g \odot z_{x}+(1-g) \odot z_{y}$，其中 $g$ 是从伯努利分布中采样得到的二进制随机向量。然后我们定义了一个新的重建损失，它使用 $z_{xy}$ 重建原始句子：

$$
\mathcal{L}^{i n t}=\mathbb{E}_{(x, l) \sim p_{\text {data }}, y \sim p_{G}\left(\cdot \mid z_{x}, l^{\prime}\right)}\left[-\log p_{G}\left(x \mid z_{x y}, l\right)\right]
$$

值得注意的是，当 $g_i=1$ 时， $\mathcal{L}^{i n t}$ 退化为 $\mathcal{L}^{a e}$，当 $g_i=0$ 时，$\mathcal{L}^{i n t}$ 退化为 $\mathcal{L}^{bt}$。

### 属性一致性

对抗损失使得模型生成真实且属性兼容的句子，它的目标是匹配句子和属性向量对 $(s,a)$ 之间的分布。这个句子可以是真实或者生成的句子。令 $h_x$ 和 $h_y$ 分别是句子 $x$ 和 $y$ 的解码器隐层状态序列，$D$ 是判别器，序列 $h_x$ 保持不变，且 $l^{\prime} \neq l$：

$$
\mathcal{L}^{\mathrm{adv}}=\min _{G} \max _{D} \mathbb{E}_{(x, l) \sim p_{\text {data }}, y \sim p_{G}\left(\cdot \mid z_{x}, l^{\prime}\right)}\left[\log D\left(h_{x}, l\right)+\log \left(1-D\left(h_{y}, l^{\prime}\right)\right)\right]
$$

判别器可能会忽视属性，然后基于隐层状态做出真假判断，或者反过来。因此，作者新加了一个假样本对 $(x,l^{\prime})$，以此鼓励判别器区分真实的句子和不匹配的标签。更新后的目标如下：

$$
\mathcal{L}^{\mathrm{adv}}=\min _{G} \max _{D} \mathbb{E}_{(x, l) \sim p_{\text {data }}, y \sim p_{G}\left(\cdot \mid z_{x}, l^{\prime}\right)}\left[2 \log D\left(h_{x}, l\right)+\log \left(1-D\left(h_{y}, l^{\prime}\right)\right)+\log \left(1-D\left(h_{x}, l^{\prime}\right)\right)\right]
$$

判别器使用如下的结构：

$$
D(s, l)=\sigma\left(l_{v}^{T} W \phi(s)+v^{T} \phi(s)\right)
$$

其中，$l_v$ 代表了与 $l$ 对应的二进制属性向量，$\phi$ 是一个双向的RNN编码器。

总体的loss函数是 $\mathcal{L}^{\mathrm{int}}+\lambda \mathcal{L}^{\mathrm{adv}}$，$\lambda$ 是超参。

## 实验设置

编码器使用 GRU，隐层大小为500，属性标签表示为二进制向量，通过线性映射得到embedding，大小为200。解码器使用编码器和属性标签的拼接来初始化，它也是GRU，隐层大小为700。判别器将RNN隐层状态序列和属性向量为输入，编码序列用大小为700的双向RNN得到。

### 评价准则

前人的工作只关注了评价生成文本的属性兼容性，但忽视了评价文本的内容一致性。目前，多数的属性都缺乏平行语料。因此作者在没有标记数据的情形下定义了评价指标。尽管这些指标各自都有其不足之处，但综合起来它们有助于客观地比较不同的模型并在不同的工作中进行一致的评估。

**属性准确率**：作者先使用CNN网络预训练一个情感分类器，然后以此来评估生成文本的情感准确性。

**内容一致性**：作者从没有平行语料的无监督机器翻译评价中获得启发。给定两个非平行数据集 $D_{src}$ 和 $D_{tgt}$，和两个翻译模型 $M_{s r c \rightarrow t g t}$、$M_{t g t \rightarrow s r c}^{\prime}$，评价指标如下：

$$
f_{\text {content }}\left(M, M^{\prime}\right)=0.5\left[\mathbb{E}_{x \sim D_{\text {src }}} \operatorname{BLEU}\left(x, M^{\prime} \circ M(x)\right)+\mathbb{E}_{x \sim D_{\text {tgt }}} \operatorname{BLEU}\left(x, M \circ M^{\prime}(x)\right)\right]
$$

其中，$M \circ M^{\prime}(x)$ 代表将 $x \in D_{src}$ 翻译到 $D_{tgt}$，然后再翻译回 $D_{src}$。作者假设 $D_{src}$ 和 $D_{tgt}$ 分别是积极、消极情感的测试集，然后 $M$ 和 $M^{\prime}$ 分别代表以积极和消极情感为条件的生成模型。

**流畅性**：作者以困惑度来评价生成文本的流畅性。
