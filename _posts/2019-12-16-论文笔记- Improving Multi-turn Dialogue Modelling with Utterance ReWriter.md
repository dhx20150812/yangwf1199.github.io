---
layout:     post
title:      论文笔记
subtitle:   Improving Multi-turn Dialogue Modelling with Utterance ReWriter
date:       2019-12-16
author:     dhx20150812
header-img: img/post-bg-ios9-web.jpg
catalog: true
mathjax: true
tags:
    - NLP
    - 对话系统
---

# Improving Multi-turn Dialogue Modelling with Utterance ReWriter

>来自ACL 2019，https://arxiv.org/pdf/1906.07004.pdf

## Introduction

单轮对话建模已经取得令人瞩目的进展，但是多轮对话的表现还远远不能令人满意。作者认为，主要的原因是我们的日常对话中存在着大量的指代和信息遗漏的问题。作者分析了2000条中文多轮对话的样本，经过统计，70%的utterance中存在不同程度的指代和信息遗漏的问题。下面是一个简单的多轮对话的例子——

<img src="https://note.youdao.com/yws/api/personal/file/WEB0359bf9b63981716a269e1e98f6694a2?method=download&shareKey=f6193241573552c5b168720d2bdf2893" alt="image-20191213104237827" style="zoom:50%;" />

在第一个对话中的“他”指代是“梅西”。第二个对话中的“为什么呢？”省略了原始的问题“为什么最喜欢泰坦尼克？”。如果不扩展出这些指代和信息遗漏，机器没法继续下面的对话。

为了解决这样的问题，作者提出可以重写当前的utterance，将多轮对话简化为单轮。关于utterance的重写要做到（1）指代消解，（2）信息补齐。在上图的例子中，每一个utterance 3 都被重写成了 utterance $3^{'}$ 。然后，对话系统只考虑utterance $3^{'}$，而不考虑utterance 1和2，来产生后续的回复。这样的简化在保留了必要的信息的同时，缩短了上下文的长度，它降低了多轮对话的复杂度。

训练好的utterance重写器与模型无关，它可以轻松地集成到其他黑盒对话系统中。 由于对话历史记录的信息以重写后的单轮对话的形式反映出来，因此它的存储效率也更高。

为了获得utterance重写的有监督训练数据，作者构建了一个包含了20k条中文多轮对话的数据集。每一个utterance都带有一个手工标注的重写的utterance。改写的utterance是根据注意力机制通过复制对话历史记录或当前utterance中的单词而生成的。

本文中作者的贡献主要有三点：

1.  收集了高质量的标注数据集，以在多回合对话中实现指代消解和信息补齐，这可能会有利于未来的相关研究。
2.  提出了一种高效的基于Transformer的utterance重写器，它的性能优于其他几个强大的baseline模型。
3.  训练好的utterance重写器集成到两个在线聊天机器人中之后，展示出了对原始系统的重大改进。

## 数据集

为了获得句子重写的平行语料，作者从社交网站上爬取了200k条多轮对话数据。在开始标注之前，作者随机抽取了2k条样本来分析其中指代和信息遗漏发生的频率。作者发现约30%的utterance既没有指代也没有遗漏，很少的一部分utterance两者都有。这进一步证明了在多轮对话中强调这两个问题的重要性。

<img src="https://note.youdao.com/yws/api/personal/file/WEBf2a64d9f3e8be2f00a2f2874fdc393ae?method=download&shareKey=910e749c33cc9b9ab35e68506b7b3e50" alt="image-20191213110605322" style="zoom: is 30%;" />

在标注过程中，手工标注者需要识别这两种情况，然后重写utterance来包含所有被隐藏的信息。标注者需要在给定原始的utterance 1 和 2 的前提下，改写utterance 3 来获得 utterance $3^{'}$。为了保证标注质量，每位标注者每天标注量的10%将被抽查。仅当检查结果的准确性超过95％时，才认为标注有效。

除准确性检查外，还要求管理人员：

（1）选择在日常会话中更可能谈论的主题

（2）尝试涵盖更广泛的领域

（3）平衡指代和信息遗漏的比例

整个标注过程共计花费了4个月才完成。最终，作者获得了40k条高质量的标注数据。其中的一半是负样本，不需要改写，还有一半是正样本，需要改写。重写后的utterance平均含有10.5个词语，相比原本的上下文长度减少了80%。

## 模型

模型的整体结构如下图所示：

<img src="https://note.youdao.com/yws/api/personal/file/WEBe15da8d2ca9e01ae1a9a37f8201b34a5?method=download&amp;shareKey=0071f2b59510a3606795b4fe84e038f5" alt="image-20191216102244247" style="zoom:67%;" />

### 问题形式化

我们将一个训练样本记为 $(H,U_n \rightarrow R)$，其中 $H=\{U_1,U_2,\cdots,U_{n-1}\}$ 代表了前 $n-1$ 时间的对话历史，$U_n$ 是第 $n$ 轮的utterance，也就是需要改写的utterance。$R$ 是将 $U_n$ 改写后得到的包含了所有指代的和遗漏的信息的utterance。如果 $U_n$ 中没有指代或是遗漏的信息，那么 $U_n$ 将会和 $R$ 相同（也就是负样本）。我们的目标是学习到一个映射函数  $p(R \mid H,U_n)$，它可以基于对话历史 $H$ 自动重写 $U_n$。

### Encoder



我们先将 $(H,U_n)$ 中的所有token全部展开为 $(w_1,w_2,\cdots,w_m)$，$m$ 是整个对话中token的数量。每两轮对话的token之间插入一个终止符。作者将整个token序列输入到Transformer中，对于每个token而言，输入的embedding向量是词嵌入、位置嵌入和轮数嵌入三者之和：

$$
I\left(w_{i}\right)=W E\left(w_{i}\right)+P E\left(w_{i}\right)+T E\left(w_{i}\right)
$$

词嵌入 $WE(w_i)$ 和位置嵌入 $PE(w_i)$ 与Transformer结构中的一致。作者加入了一个轮数嵌入 $TE(w_i)$ 来指示当前这个token来自哪一轮。因此同一轮的token会共享同样的轮数嵌入。

然后输入的embedding会被输入到 $L$ 层堆叠的Encoder中来获得最终的编码表示，每一个Encoder都包含了self-attention层和前向神经网络层：

$$
\begin{aligned} \mathbf{E}^{(0)} &=\left[I\left(w_{1}\right), I\left(w_{2}\right), \ldots, I\left(w_{m}\right)\right] \\ \mathbf{E}^{(l)}=& \text { FNN }\left(\text { MultiHead }\left(\mathbf{E}^{(l-1)}, \mathbf{E}^{(l-1)}, \mathbf{E}^{(l-1)}\right)\right) \end{aligned}
$$

最终的编码来自第 $L$ 层编码器的输出。

### Decoder

Decoder同样包含了 $L$ 层，每层由三个子层组成。第一层是multi-head self-attention层：

$$
\mathbf{M}^{l}=\text { MultiHead }\left(\mathbf{D}^{(l-1)}, \mathbf{D}^{(l-1)}, \mathbf{D}^{(l-1)}\right)
$$

而初始时刻有 $\mathbf{D}^{(0)}=R$。

第二层是encoder-decoder attention，它的作用是将encoder的信息融入到decoder中。由于在这个任务中，对话历史 $H$ 和当前utterance $U_n$ 作用是不同的。因此，作者对其分别构造了key-value矩阵。Encoder最后部分的输出 $\mathbf{E}^{(L)}$ 被分为两个部分 $\mathbf{E}^{(L)}\_{(H)}$ 和 $\mathbf{E}^{(L)}\_{(U_n)}$，于是encoder-decoder向量由下面计算得到：

$$
\begin{aligned} \mathbf{C}(H)^{l} &=\text { MultiHead }\left(\mathbf{M}^{(l)}, \mathbf{E}_{H}^{(L)}, \mathbf{E}_{H}^{(L)}\right) \\ \mathbf{C}\left(U_{n}\right)^{l} &=\text { MultiHead }\left(\mathbf{M}^{(l)}, \mathbf{E}_{U_{n}}^{(L)}, \mathbf{E}_{U_{n}}^{(L)}\right) \end{aligned}
$$

第三层是一个全连接网络：

$$
\mathbf{D}^{(l)}=\operatorname{FNN}\left(\left[\mathbf{C}(H)^{l} \circ \mathbf{C}\left(U_{n}\right)^{l}\right]\right)
$$

$\circ$ 表示向量拼接。

### 输出分布

在解码阶段，我们希望模型能够学习到在不同的时刻如何从 $H$ 或 $U_n$ 中复制单词。因此，作者引入了一个soft gating权重 $\lambda$ 来做决策。解码概率的计算结合了最后一层的attention分布：

$$
p\left(R_{t}=w | H, U_{n}, R_{<t}\right)=\lambda  \sum_{i:\left(w_{i}=w\right) \wedge\left(w_{i} \in \mathrm{H}\right)} a_{t, i} \\+(1-\lambda)  \sum_{j:\left(w_{j}=w\right) \backslash\left(w_{j} \in U_{n}\right)} a_{t, j}^{\prime} \\ a= \text { Attention }\left(\mathbf{M}^{(L)}, \mathbf{E}_{U_{n}}^{(L)}\right) \\ a^{\prime}= \text { Attention }\left(\mathbf{M}^{(L)}, \mathbf{E}_{H}^{(L)}\right) \\ \lambda=\sigma\left(\boldsymbol{w}_{d}^{\top} \mathbf{D}_{t}^{L}+\boldsymbol{w}_{H}^{\top} \mathbf{C}(H)_{t}^{L}+\boldsymbol{w}_{U}^{\top} \mathbf{C}\left(U_{n}\right)_{t}^{L}\right)
$$

$a$ 和 $a^{\prime}$ 分别是 $H$ 和 $U_n$ 中关于token的attention分布，$\boldsymbol{w}\_{d}$、$\boldsymbol{w}\_{H}$和$\boldsymbol{w}\_{U}$ 是需要学习的参数，$\sigma$ 是sigmoid函数。

gating权重 λ 就像哨兵一样，告诉解码器是从对话历史记录 $H$ 中提取信息还是直接从 $U_n$ 中复制。如果 $U_n$ 中既不包含指代也不包含信息遗漏，那么 $\lambda$ 一直为1，直接从 $U_n$ 中拷贝；否则 $\lambda$ 为0，attention机制就起作用了，选择何时从原句子拷贝，何时从对话历史拷贝。整个模型的训练目标是最大化 $p(R \mid H，U_n)$。


