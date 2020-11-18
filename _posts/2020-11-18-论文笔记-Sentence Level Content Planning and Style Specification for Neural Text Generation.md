---
layout:     post
title:      论文笔记
subtitle:   Sentence Level Content Planning and Style Specification for Neural Text Generation
date:       2020-11-18
author:     dhx20150812
header-img: img/post-bg-ios9-web.jpg
catalog: true
mathjax: true
tags:
    - NLP
    - Argument Generation
    - Content Planning
---

# Sentence-Level Content Planning and Style Specification for Neural Text Generation

> 来自EMNLP 2019，<https://www.aclweb.org/anthology/D19-1055.pdf>
>
> 代码已开源：<http://xinyuhua.github.io/Resources/emnlp19/>

##  Introduction

文本生成需要解决三个关键的问题：

（1）`content selection`——识别要呈现的相关信息

（2）`text planning`——将内容组织成有序的句子

（3）`surface realization`——决定能够提供连贯输出的单词和句法结构

传统的文本生成系统通常单独处理每个组件，因此需要在数据采集和系统工程方面进行大量的工作。端到端训练的神经网络模型可以生成流畅的文本，但受限于模型的结构和训练目标，他们往往缺乏可解释性，且生成的文本不连贯。

为了解决这个问题，作者认为，神经网络模型必须对 `content planning` 进行充分控制，以便产生一致的输出，尤其是对于长文本生成。同时，为了达到预期的语义目标，通过明确地建模和指定适当的语言风格，有助于实现风格控制的 `surface realization`。

<img src="https://note.youdao.com/yws/api/personal/file/WEB6e1dcad63165cd7dbb797c0fc3cfef43?method=download&shareKey=f638258f272c5d62e3cb0e11b3f18e81" alt="image-20201118195849944" style="zoom:50%;" />

例如，在“美国是否应该完全切断对外援助”这个话题上，上图展示了人类如何挑选一系列论点和适当的语言风格。辩论以“对外援助可以作为政治谈判筹码”为命题，然后是涵盖了几个关键概念的具体例子。最后是以辩论风格的语言结束。

因此，作者提出了一个端到端训练的神经文本生成框架，包括对传统生成组件的建模，以促进对生成文本内容和语言风格的控制。作者提出的模型使用了句子级别的 content planning 来进行信息的选择和排序，然后使用风格可控的surface realization来产生最终的输出。

![image-20201118195833139](https://note.youdao.com/yws/api/personal/file/WEB692f57f44765614bf437edfda740ff33?method=download&shareKey=846ddcfefc142200b0a34322df1af374)

如上图所示，模型的输入包括一个主题语句和一组关键短语。输出是一个相关且连贯的段落，以反映输入中的要点。作者使用了两个独立的解码器，对于每个句子：（1）`text planning` 解码器根据先前的选择来选择相关的关键短语和所需的风格（2）`surface realization` 解码器以指定的样式生成文本。

作者在三个任务上进行了实验：（1）Reddit CMV的论点生成任务（2）维基百科的引言段落生成（3）AGENDA数据集上的科研论文摘要生成。实验结果表明，在三个数据集上，作者提出的模型比非检索的方式取得了更高的BLEU、ROUGH和METEOR。

## Model

模型的输入包括两部分：（1）主题陈述 $\mathbf{x}=\\{x_{i}\\}$。它可以是论点，可以是维基百科的文章标题，也可以是论文标题。（2）关键词记忆单元，$\mathcal{M}$，它包含话题要点列表，在 `content planning` 和风格选择中起着关键作用。目标输出是一个序列 $\mathbf{y}=\\{y_{t}\\}$，它可以是一个反驳的论点，维基百科文章中的一段话，或者一篇论文摘要。

### 输入编码

输入的文本 $\mathbf{x}$ 通过双向LSTM，它最后一层的隐藏层用于 `content planning` 解码器和  `surface realization` 解码器的初始状态。为了编码记忆单元 $\mathcal{M}$ 中的关键词组，每个关键词首先将它的所有单词的GLOVE词向量中求和，转换成一个向量 $e_{k}$。基于双向LSTM的关键词读取器采用隐藏状态 $\boldsymbol{h}_{k}^{e}$ 对 $\mathcal{M}$ 中的所有关键短语进行编码。作者还在 $\mathcal{M}$ 中插入`<START>`和`<END>`标记，以便于学习何时开始和完成选择。

### Content Planning

`Content planning` 基于前面的句子中已选择的关键词短语，从 $\mathcal{M}$ 中为每个句子（以 $j$ 为索引）选择一组关键词短语，从而实现主题连贯性和避免内容重复。这个选择器表示为一个选择向量 $\boldsymbol{v}\_{j} \in \mathbb{R}^{\|\mathcal{M}\|}$，它的每一维 $\boldsymbol{v}\_{j, k} \in\{0,1\}$ 表示第 $k$ 个短语是否被选择为生成第 $j$ 个句子。

作者使用了句子级别的 LSTM 网络 $f$，根据选择的短语词向量之和 $\boldsymbol{m}_{j}$，产生第 $j$ 个句子的隐层状态：

$$
\begin{array}{l}
\boldsymbol{s}_{j}=f\left(\boldsymbol{s}_{j-1}, \boldsymbol{m}_{j}\right) \\
\boldsymbol{m}_{j}=\sum_{k=1}^{|\mathcal{M}|} \boldsymbol{v}_{j, k} \boldsymbol{h}_{k}^{e}
\end{array}
$$

在实际中，多次使用的短语应该避免被再次选中，因此，作者提出了一个向量 $\boldsymbol{q}_{j}$，它用来追踪到第 $j$ 个句子时哪些短语已经被选择过：

$$
\boldsymbol{q}_{j}=\left(\sum_{r=0}^{j} \boldsymbol{v}_{r}\right)^{T} \times \mathbb{E}
$$

其中，$\mathbb{E}=\left[\boldsymbol{h}\_{1}^{e}, \boldsymbol{h}\_{2}^{e}, \ldots \boldsymbol{h}\_{\|\mathcal{M}\|}^{e}\right]^{T} \in \mathbb{R}^{|\mathcal{M}| \times H}$ 是短语表示的矩阵。$H$ 是 LSTM 的隐层维度。因此，下一时刻的短语选择向量 $\boldsymbol{v}_{j+1}$ 有下面的式子得到：

$$
P\left(\boldsymbol{v}_{j+1, k}=1 \mid \boldsymbol{v}_{1: j}\right)=\sigma\left(\mathbf{w}_{v}^{T} \boldsymbol{s}_{j}+\boldsymbol{q}_{j} \mathbf{W}^{c} \boldsymbol{h}_{k}^{e}\right)
$$

作为学习目标的一部分，作者采用了交叉熵来作为评价准则：

$$
\mathcal{L}_{\mathrm{sel}}=-\sum_{(\mathbf{x}, \mathbf{y}) \in D} \sum_{j=1}^{J}\left(\sum_{k=1}^{|\mathcal{M}|} \log \left(P\left(\boldsymbol{v}_{j, k}^{*}\right)\right)\right)
$$

### Style Specification


