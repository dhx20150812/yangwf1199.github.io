---
layout:     post
title:      论文笔记
subtitle:    LexicalAT: Lexical-Based Adversarial Reinforcement Training for Robust Sentiment Classiﬁcation
date:       2019-04-13
author:     dhx20150812
header-img: img/post-bg-ios9-web.jpg
catalog: true
mathjax: true
tags:
    - NLP
    - 情感分类
    - 对抗攻击
---

# LexicalAT: Lexical-Based Adversarial Reinforcement Training for Robust Sentiment Classiﬁcation

>   来自EMNLP 2019，https://www.aclweb.org/anthology/D19-1554.pdf

## Introduction

最近的研究工作发现，情感分类模型容易受到对抗样本的攻击。这表明这些模型并没有学习到决定正确标签的潜在模式。过拟合的问题仍需探索。

近年来，针对攻击问题已经提出了几种方法。 这些研究可以大致分为两类，基于数据增强的方法和基于对抗训练的方法。前者的核心思想是通过预先设计的样本来扩展训练数据来辅助分类器的训练。后者通过在词嵌入中添加随机噪声来提高泛化能力。

在这篇文章中，作者提出了一个基于词汇的对抗增强训练框架。这个框架包括了一个生成器和一个分类器。生成器提供对抗样本来攻击分类器，分类器学习防御攻击。

为了生成多样性的样本，作者使用了WordNet，通过将某些单词替换为同义词来增加对扰动的干扰。特别地，生成器的输出是一个行为的序列，它决定了哪些词应该被替换以及他们的替换词。

由于生成器中有离散的生成步骤，基于梯度的优化方法无法反向传播误差。作者使用策略梯度方法来训练生成器。以分类器的反馈作为奖励，鼓励生成器为分类器生成更严格的样本。 反过来，随着越来越多的困难的训练样本，分类器变得越来越强大。

总的来说，本文有三点贡献：

1.  提出了一个基于词汇的对抗增强训练框架，可以增强情感分类模型的鲁棒性。
2.  将知识库和对抗学习结合，知识库有助于生成各种样本，而对抗学习则可以制定攻击策略。
3.  实验表明LexicalAT超越了baseline，在许多模型上提升了效果。

## Approach

### Overview

下图展示了LexicalAT框架的总体结构——

![image-20191217202250336](https://note.youdao.com/yws/api/personal/file/WEB00f025bf1de6a98706d50626ea50b02c?method=download&shareKey=445ab2afe08cfe94c3778b8750c5015d)

给定一个句子，生成器首先生成一个行为序列，它决定替换哪个词，然后构建新样本。然后新样本被送到分类器中获得行为的reward，如果生成的样本成功地迷惑了分类器，降低了真实样本的概率，我们就认为他是一个好的样本，给他一个高的reward。通过这种方式，鼓励生成器为分类器生成困难的示例，并在分类器的培训过程中使用具有挑战性的生成示例。通过交替训练生成器和分类器，能够训练分类器更加健壮和强大。

### 生成器

生成器通过使用WordNet在真实样本中添加噪音来生成攻击样本。生成过程可以简化为序列建模问题。它将文本作为输入，输出行为序列。作者定义了五种行为，如下所示：

<img src="https://note.youdao.com/yws/api/personal/file/WEBf8fbb74d02678ae8ea5ff1cfc95bbdb1?method=download&shareKey=85fd3c41b704c06354c95578cb2409aa" alt="image-20191217204212918" style="zoom: 50%;" />

对于一个单词序列 $\boldsymbol{x}=\left\{x_{1}, x_{2}, \cdots, x_{n}\right\}$，$n$ 是输入文本的长度。模型会生成一个替换行为的序列 $\boldsymbol{a}=\left\{a_{1}, a_{2}, \cdots, a_{n}\right\}$，然后由 $\boldsymbol{x}$ 和 $\boldsymbol{a}$ 生成新的句子。由于作者提出的框架与生成器的结构无关，为了简化起见，作者使用了传统的biLSTM。

### 分类器

这篇论文重点关注单分类。输入是一个词序列，输出是一个预定义的标签集合中的一个标签 $Y=\left\{y_{1}, y_{2}, \cdots, y_{k}\right\}$。

作者使用了三种经典的情感分类模型——CNN、RNN和BERT。

### 对抗增强训练

由于生成步骤中的选择是离散的，作者使用策略梯度方法来对抗性地训练这两个模块。可以将生成器看作agent，将分类器看作环境，生成器通过最大化从环境里得到的reward来提升自己。

给定一个真实样本 $(\boldsymbol{x},y)$，生成器基于如下的概率分布采样出一个行为序列：

$$
\hat{\boldsymbol{a}} \sim p_{\mathbf{G}}(\boldsymbol{a} | \boldsymbol{x}, \boldsymbol{\theta})
$$

其中 $\boldsymbol{\theta}$ 是生成器的参数。基于这个行为序列，我们可以通过单词置换后得到一个新的样本 $\left(\boldsymbol{x}^{\prime}, y\right)$。

然后将真实的样本和生成的样本同时送入分类器，得到替换的 reward $r(\hat{\boldsymbol{a}})$。我们将 reward 定义为真实样本和生成样本之间的概率 $y$ 的绝对值之差：

$$
r(\hat{\boldsymbol{a}})=\log p_{\mathbf{C}}(y | \boldsymbol{x} ; \boldsymbol{\phi})-\log p_{\mathbf{C}}(y | \hat{\boldsymbol{x}} ; \boldsymbol{\phi})
$$

其中 $\boldsymbol{\phi}$ 是分类器的参数。在实际中，作者使用了如下公式来获得 reward，稳定地训练生成器：

$$
r^{\prime}(\hat{\boldsymbol{a}})=r(\hat{\boldsymbol{a}})-\mathbb{E}_{P_{G}(\boldsymbol{a} | \boldsymbol{x}, \boldsymbol{\theta})}(r(\boldsymbol{a}))
$$

其中 $\mathbb{E}_{P_{G}(\boldsymbol{a} | \boldsymbol{x}, \boldsymbol{\theta})}(r(\boldsymbol{a}))$ 是 $r(\boldsymbol{a})$ 的期望。

然后，reward 被返回给生成器，参数 $\boldsymbol{\theta}$ 的期望梯度可以近似为：

$$
\nabla_{\boldsymbol{\theta} \mathcal{L}} \approx -r^{\prime}(\hat{\boldsymbol{a}}) \nabla_{\boldsymbol{\theta}} \log p_{\mathbf{G}}(\hat{\boldsymbol{a}} | \boldsymbol{x} ; \boldsymbol{\theta}) -\mathbb{E}_{p_{\mathbf{G}}(\hat{a} | \boldsymbol{x} ; \boldsymbol{\theta})} r^{\prime}(\hat{\boldsymbol{a}})
$$

第二项表示在实践中我们通过采样来估计期望值。

生成的样本通过最小化交叉熵损失来训练分类器：

$$
\mathcal{L}=-\log p_{\mathbf{C}}\left(y | \boldsymbol{x}^{\prime}, \boldsymbol{\phi}\right)
$$

此外，为了防止分类器忘记训练数据的知识，我们还使用了 teacher forcing 的方法。在每一步的增强训练之后，直接使用真实样本来训练分类器：

$$
\mathcal{L}=-\log p_{\mathbf{C}}\left(y | \boldsymbol{x}, \boldsymbol{\phi}\right)
$$



## 实验

作者选择了三个鲁邦增强的方法作为baseline，在多个数据集（SST-2、SST-5、RT和Yelp）上进行了实验。结果如下：

<img src="https://note.youdao.com/yws/api/personal/file/WEBc74396befaf763fd96af4e533f0c0001?method=download&shareKey=6321b382dc14a245a00f01238ad06bf7" alt="image-20191217232300896" style="zoom:50%;" />

显然，作者提出的LexicalAT方法超越了基础的baseline模型CNN、RNN和BERT。相比于BERT，LexicalAT在四个数据集上分别超越了0.43、0.31、0.11和0.74。

同时可以发现，SynDA方法相比于baseline并没有超越太多。LexicalAT和SysDA的主要差异在于，SynDa通过随机替换的策略生成样本，而LexicalAT通过从对抗增强训练框架中学到的动态替换策略来生成样本。这效果的差异表明，对抗强化训练框架对于学习针对分类器弱点的攻击策略非常有效。

此外，可以发现LexicalAT在多个数据集上打败了VAT。VAT只在输入词的词嵌入上添加随机噪声，并没有改变输入文本或表达，这限制了模型的鲁棒性的提升。

作者同时实验了模型在防御攻击时的表现，结果如下图。简而言之，可见通过对抗增强训练的模型拥有更好的防御能力。

![image-20191217233223507](https://note.youdao.com/yws/api/personal/file/WEB3245eb93ace24767f5ac19d7452e78a7?method=download&shareKey=859ad9108d7eca69f32eb9730198ce12)
