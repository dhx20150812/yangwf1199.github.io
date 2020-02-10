---
layout:     post
title:      论文笔记
subtitle:   Toward Controlled Generation of Text
date:       2020-02-11
author:     dhx20150812
header-img: img/post-bg-ios9-web.jpg
catalog: true
mathjax: true
tags:
    - NLP
    - 文本生成
    - 可控文本生成
---

# Toward Controlled Generation of Text

>   来自ICML 2017，https://arxiv.org/pdf/1703.00955.pdf
>
>   代码已开源：https://github.com/asyml/texar/tree/master/examples/text_style_transfer

## Introduction

近些年来，研究人员对深度生成模型的兴趣激增，例如VAE、GAN和自回归模型。他们在计算机视觉领域取得了不错的成绩，但在自然语言处理领域的应用还少有研究。甚至生成逼真的句子也是具有挑战性的，因为需要生成模型来捕获句子潜在的复杂语义结构。

在本文中，作者旨在解决可控文本生成的问题。作者专注于生成逼真的句子，它们的属性可以通过学习潜在的语义表示来控制。为了能够操纵生成的句子，作者认为，需要先解决一些挑战。

1.  文本样本的离散性。由此产生的不可微性阻碍了使用全局判别器来评估生成的样本，同时也阻碍了梯度的反向传播以一种整体的方式指导生成器的优化。而这些方法已经被证明在连续图像生成和表示建模中非常有效。
2.  学习潜在表示。可解释性要求潜在表示的每个部分都只关注样本的一个方面。

针对上述问题，作者提出了一种新的文本生成模型，该模型允许使用指定的语义结构进行高度分离的表示，并生成具有动态指定属性的句子。该模型的生成器以VAE为基础，并结合了属性的判别器来有效地识别潜在编码的结构。通过可微的softmax近似实现了end-to-end的优化，该近似平滑地退火到离散情况，并有助于快速收敛。VAE的概率编码器还可以作为一个附加的鉴别器来捕获隐式建模方面的变化，并指导生成器避免在属性编码操作期间发生纠缠。

## 可控文本生成

该模型旨在生成以具有指定语义结构的向量为条件的句子。与前人的工作相比，该生成模型具有以下的特点：

1.  通过使用全局判别器来指导离散文本生成器的学习，促进对潜在编码语义的有效实施
2.  通过显式地加强独立属性上的约束来改进模型的可解释性
3.  通过使用定制的 wake-sleep方法结合变分自编码器，实现高效的半监督学习和自举

### Overview

模型的概述如下：

![image-20200204140054252](https://note.youdao.com/yws/api/personal/file/WEB3ed2a56d5ba6b24b4116d168e83679b3?method=download&shareKey=fcba44959dbac43aa41e6051e3ec9568)

作者从VAE入手开始构建自己的文本生成模型。传统的VAE使用一个无结构的向量 $\boldsymbol{z}$ ，它的维度之间都是混乱的。为了以一种可解释的方式对感兴趣的属性进行建模和控制，我们用一组结构化变量 $\boldsymbol{c}$ 扩充了非结构化变量 $\boldsymbol{z}$，每个结构化变量 $\boldsymbol{c}$  均对应句子里的一个显著且独立的语义特征。

作者希望句子生成器可以以组合向量 $(\boldsymbol{z}, \boldsymbol{c})$ 为条件，生成满足结构化代码  $\boldsymbol{c}$  中指定的属性的文本样本。与在数据空间中计算元素级距离相比，在特征空间中计算距离可以使不变性分散注意力，并提供更好的整体度量。因此，对 $\boldsymbol{c}$ 中的每个属性编码，作者设置一个单独的判别器来衡量生成样本与期望属性的匹配程度。

在这个框架中使用判别器的难点在于，文本样本是离散和不可微分的，这会阻断梯度从判别器到生成器的传播。我们使用基于softmax 的连续逼近方法，并采用逐渐降低的温度，随着训练的进行，该近似值会逐渐离散化。这个简单但有效地办法具有较低的方差和快速的收敛性。

直观地讲，具有可解释的表示形式意味着 $\boldsymbol{c}$ 中的每个结构化编码都可以独立控制其目标特征，而不会与其他属性（尤其是未明确建模的属性）纠缠不清。我们通过强制将那些无关的属性完全捕获在非结构化代码 $\boldsymbol{z}$ 中，从而将它们与 $\boldsymbol{c}$ 分开（我们将对其进行操作）来鼓励独立性。为此，我们重用VAE编码器作为识别 $\boldsymbol{z}$ 中建模属性的附加鉴别器，并对生成器进行训练，以便可以从生成的样本中恢复这些非结构化属性。只要 $\boldsymbol{z}$ 不变，改变不同的属性编码将使非结构化属性保持不变。

因此，如上图所示，完整的模型综合了 VAE 和属性判别器。其中，VAE的部分训练生成器识别真实的句子来生成逼真的句子，判别器强迫生成器来产生与条件编码一致的属性。生成器和判别器形成了一组协作学习者，并互相反馈信号。作者表示，这个 VAE 和 wake-sleep算法组合的学习方法实现了一个高效的半监督学习框架，它只需少量的监督就可以获得可解释性表示和生成。

### 生成器学习

生成器 $G$ 是一个 LSTM-RNN 网络，它以潜在编码 $(\boldsymbol{z}, \boldsymbol{c})$ 为条件，产生 token 序列 $\hat{\boldsymbol{x}}=\\{\hat{x}_{1}, \ldots, \hat{x}_{T}\\}$，它描述了生成式分布：

$$
\begin{aligned} \hat{\boldsymbol{x}} \sim G(\boldsymbol{z}, \boldsymbol{c}) &=p_{G}(\hat{\boldsymbol{x}} | \boldsymbol{z}, \boldsymbol{c}) \\ &=\prod_{t} p\left(\hat{x}_{t} | \hat{\boldsymbol{x}}^{<t}, \boldsymbol{z}, \boldsymbol{c}\right) \end{aligned}
$$

其中，$\hat{\boldsymbol{x}}^{<t}$ 表示在 $\hat{\boldsymbol{x}}$ 之前的 token。因此，生成过程涉及到一个离散决策序列，该序列在每个时间步长 $t$ 时，从一个多元正态分布参数化的 softmax 函数中抽取一个 token：

$$
\hat{x}_{t} \sim \operatorname{softmax}\left(\boldsymbol{o}_{t} / \tau\right)
$$

其中，$\boldsymbol{o}_{t}$ 是 logit 向量，它是 softmax 函数的输入，$\tau > 0$ 是温度，通常设为1。

表示的无结构部分 $\boldsymbol{z}$ 被建模为标准高斯先验 $p(\boldsymbol{z})$ 的连续变量。而结构化编码 $\boldsymbol{c}$ 可以包含连续和离散的变量来根据合适的先验 $p(\boldsymbol{c})$ 建模不同属性。给定观测 $\boldsymbol{x}$，基础的 VAE 使用一个条件概率编码器 $E$ 来推断潜在 $\boldsymbol{z}$：

$$
\boldsymbol{z} \sim E(\boldsymbol{x})=q_{E}(\boldsymbol{z} | \boldsymbol{x})
$$

令 $\theta_{G}$ 和 $\theta_{E}$ 分别是生成器 $G$ 和编码器 $E$ 的参数。VAE 的优化目标是最小化重构观测句子的误差，同时使得编码器更接近于先验 $p(\boldsymbol{z})$：

$$
\mathcal{L}_{\mathrm{VAE}}\left(\boldsymbol{\theta}_{G}, \boldsymbol{\theta}_{E} ; \boldsymbol{x}\right)= \mathrm{KL}\left(q_{E}(\boldsymbol{z} | \boldsymbol{x}) \| p(\boldsymbol{z})\right) -\mathbb{E}_{q_{E}(\boldsymbol{z} | \boldsymbol{x}) q_{D}(\boldsymbol{c} | \boldsymbol{x})}\left[\log p_{G}(\boldsymbol{x} | \boldsymbol{z}, \boldsymbol{c})\right]
$$

其中，$\mathrm{KL}(\cdot \| \cdot)$ 是 KL 散度，$q_{D}(\boldsymbol{c} | \boldsymbol{x})$ 是由判别器 $D$ 为 $\boldsymbol{c}$ 中各结构化变量定义的条件分布：
$$
D(\boldsymbol{x})=q_{D}(\boldsymbol{c} | \boldsymbol{x})
$$
同时，为了强制生成器生成与 $\boldsymbol{c}$ 中的结构化编码相匹配的一致的属性，判别器提供了额外的学习信号。然而，由于不可能从判别器通过离散样本传播梯度，作者求助于连续近似的方式。这个近似指的是用 softmax 之后的概率向量来替换采样的token。这样得到的 soft 生成的句子，记为 $\widetilde{G}_{\tau}(\boldsymbol{z}, \boldsymbol{c})$，输入判别器来衡量与目标属性的匹配程度，因此有如下的损失函数：

$$
\mathcal{L}_{\text {Attr, } c}\left(\boldsymbol{\theta}_{G}\right)=-\mathbb{E}_{p(\boldsymbol{z}) p(c)}\left[\log q_{D}\left(\boldsymbol{c} | \widetilde{G}_{\tau}(\boldsymbol{z}, \boldsymbol{c})\right)\right]
$$

为了保证属性之间的独立性，使其不产生相互干扰，即 $D$ 只优化目标属性而不会对其他属性进行优化，又特意的强制其他不相干的属性要在隐变量 $\boldsymbol{z}$ 中被完全的表示出来。而且这个任务作者并没有去训练一个新的判别器，而是直接将 VAE 中的编码器 $E$ 拿过来使用的。文中提到，因为 $E$ 对于 $\boldsymbol{z}$ 的作用和 $D$ 对于 $\boldsymbol{c}$ 的作用可以视为一样的，都是将文本中的属性抽出来得到其分布。因此，第三项的 loss 表示为：

$$
\mathcal{L}_{\text {Attr }, z}\left(\boldsymbol{\theta}_{G}\right)=-\mathbb{E}_{p(\boldsymbol{z}) p(\boldsymbol{c})}\left[\log q_{E}\left(\boldsymbol{z} | \widetilde{G}_{\tau}(\boldsymbol{z}, \boldsymbol{c})\right)\right]
$$

综合以上三个 Loss，训练生成器的目标是：

$$
\min _{\boldsymbol{\theta}_{G}} \mathcal{L}_{G}=\mathcal{L}_{\mathrm{VAE}}+\lambda_{c} \mathcal{L}_{\mathrm{Attr}, c}+\lambda_{z} \mathcal{L}_{\mathrm{Attr}, z}
$$

其中 $\lambda_c$ 和 $\lambda_z$ 是平衡因子。训练 VAE 的目标是最小化VAE loss，也就是 $\mathcal{L}_{\mathrm{VAE}}$。

### 判别器学习

判别器的学习是半监督的方式。记 $\boldsymbol{\theta}\_{D}$ 为判别器的参数。为了学习到特定语义，作者设置了一系列有标签样本 $ \mathcal{X}\_{L}=\\{\left(\boldsymbol{x}\_{L}, \boldsymbol{c}\_{L}\right)\\}$ 来训练判别器 $D$ ，目标如下：

$$
\mathcal{L}_{s}\left(\boldsymbol{\theta}_{D}\right)=-\mathbb{E}_{\mathcal{X}_{L}}\left[\log q_{D}\left(\boldsymbol{c}_{L} | \boldsymbol{x}_{L}\right)\right]
$$

此外，生成器 $G$ 还能够合成噪声句子属性对 $(\hat{\boldsymbol{x}}, \boldsymbol{c})$，该属性对可用于增强半监督学习的训练数据。为了缓解噪声数据的问题并确保模型优化的鲁棒性，作者引入了最小熵正则化项。因此训练目标是：

$$
\mathcal{L}_{u}\left(\boldsymbol{\theta}_{D}\right)=-\mathbb{E}_{p_{G}(\hat{\boldsymbol{x}} \mid z, c) p(\boldsymbol{z}) p(\boldsymbol{c})}\left[\log q_{D}(\boldsymbol{c} \mid \hat{\boldsymbol{x}})+\beta \mathcal{H}\left(q_{D}\left(\boldsymbol{c}^{\prime} \mid \hat{\boldsymbol{x}}\right)\right)\right]
$$

其中，$\mathcal{H}\left(q_{D}\left(\boldsymbol{c}^{\prime} \mid \hat{\boldsymbol{x}}\right)\right)$ 是生成的句子 $\hat{\boldsymbol{x}}$ 上分布 $q_{D}$ 的经验香农熵。直觉来说，最小熵正则项使得模型在预测标签时有更高的置信度。

于是判别器的联合训练目标由如下得到：

$$
\min _{\boldsymbol{\theta}_{D}} \mathcal{L}_{D}=\mathcal{L}_{s}+\lambda_{u} \mathcal{L}_{u}
$$

### 联合学习

生成器和判别器联合学习的算法流程如下图所示：

<img src="https://note.youdao.com/yws/api/personal/file/WEB2926c12a234fc3ddc53f26929d275d5b?method=download&shareKey=09f0819c4293ec80de99a1c7ac662362" alt="image-20200207161628194" style="zoom:67%;" />

首先，在一个很大的无标签的句子集合上通过训练 VAE 来初始化生成器，目的是使得此时从先验分布 $p(\boldsymbol{c})$ 上采样的潜在编码 $\boldsymbol{c}$ 最小化VAE loss。然后交替着优化生成器和判别器来训练整个模型。

## 实验

作者应用该模型从情感和时态两方面来生成短文本。

关于大规模无标记文本，作者选用了 IMDB 电影评论数据集，从中挑选了少于15个词的句子，并将不常用的词替换为 \<unk\> token，最终得到了约1.4M个句子，词表大小为16K。

关于情感数据集，作者选用了如下三个数据集——

1.  **SST-full** 训练集、验证集和测试集的大小分别为6920、872和1821。作者使用了2837个长度小于15的训练样本，然后在原始的测试集上进行了评估
2.  **SST-small** 为了研究在半监督学习中进行准确属性控制所需的标记数据的大小，作者从 **SST-full** 中抽取了250个句子用于训练
3.  **Lexicon** 为了验证模型的有效性，即使是使用单词级的标签也可以进行句子级的控制。词汇表包含了2700个带有情感标签的词。
4.  **IMDB** 作者随机从IMDB数据集中挑选了积极和消极的电影评论，训练集、验证集和测试集的大小分别是5K、1K和10K。

第二个属性是句子中主语的时态。关于时态，目前没有公开的数据集，因此作者从 TimeBank 数据集中进行训练，并获得一个由5250个单词和短语组成的词典，这些单词和短语用 {“ past”，“ present”，“ future”} 中的一个元素进行标记。

### 生成属性的准确度

作者通过评估生成指定情感的准确性来定量衡量句子属性的控制。作者将自己的模型与另一个半监督 VAE（S-VAE）进行对比。S-VAE也是通过属性编码来重构源文本，但是没有使用判别器。作者使用了SOTA的情感分类器（在SST数据集取得了90%的准确率）来自动评估情感生成的准确度。

下图展示了 S-VAE 和作者提出的模型在 30K 个句子上采用 SST-full、SST-small 和 Lexicon 等情感数据集训练得到的结果。

![image-20200210123250966](https://note.youdao.com/yws/api/personal/file/WEB5c0494d1f2c6ae74c605023cb11a3b3b?method=download&shareKey=aa2f85da537c1753b87a0a1b2548247d)

可见作者提出的模型在所有的数据集上都超越了 S-VAE。特别地是，如果在 SST-small 数据集上只使用250个标记样本学习的话，作者的模型也能取得还不错的准确度，这证明了展示了模型在很少的监督下学习纠缠表示的能力。

更重要的是，在 Lexicon 中仅给出单词级的标记，我们的模型成功地将知识转移到句子级，并合理地生成了所需的情感。 与我们通过直接评估生成的句子来驱动学习的方法相比，S-VAE 仅尝试通过重构标记的单词来捕获情感语义，这效率较低且性能较差。

下图展示了在四个数据集上训练的分类器的准确性。其中，“Std” 是一个在原始数据集上的 ConvNet，使用了和情感判别器一样的网络结构；“H-reg"额外地对生成的句子使用了最小熵正则化；“ours”结合了最小熵正则化和情感编码 $\boldsymbol{c}$。结果可见，我们的模型在四个数据集上都取得了最好的效果。“ H-Reg”相对于“ Std”的改进表明，最小熵正则化对生成句子的积极影响。如“ Ours”和“ S-VAE”中所述，结合生成样本的情感代码，可提供额外的性能提升，表明条件生成可自动创建标记数据。与上述实验一致，我们的模型优于S-VAE。

<img src="https://note.youdao.com/yws/api/personal/file/WEBeae05a9ba02d75b615a7d8b6881b822c?method=download&shareKey=5c3fdcb03777bd8e70f09890b1fabb03" alt="image-20200210133331187" style="zoom:67%;" />

### 无纠缠表示

作者同时还研究了生成结果的可解释性和独立性约束。下图展示了模型在使用独立性约束和不使用约束时的生成样本。

![image-20200210140031257](https://note.youdao.com/yws/api/personal/file/WEB695e98c968d96da575a085d800e9dc12?method=download&shareKey=580c13ee1d7cbbb0a495e3fc3f857041)

左侧的生成过程使用了独立性约束，在控制情感时保持其他属性不变，可见生成的句子对中，主语、语气和措辞都高度相关。反之，右侧生成的句子则显得较为混乱。

## 讨论

这篇论文最大的贡献在于将 VAE 中的属性相互交错在一起隐变量 $\boldsymbol{z}$，通过加入特定判别器的方法，成功的将 $\boldsymbol{z}$ 中的特定属性 $\boldsymbol{c}$ 抽出来，从而生成具有特定属性的文本。并且各个属性之间是可以单独进行训练，然后随意组合的，而且半监督学习要求的样本数量很少，形式也没有什么要求，甚至可以是单词和短语，这就使得训练对样本的需求要求大大的降低了，并且可以并行的训练很多个属性。

而整篇论文从结构的搭建，到连续逼近离散的处理，再到模型的优化算法，包括施加独立性的强制约束和通过判别器 $D$ 的性能来隐性的反应 $G$ 的性能，都是闪光之处，也是十分值得去借鉴和学习的。

这篇论文现在还存在的问题就是句子长度太短，只有15个，并且没有给出如何将独立的不同属性组合起来的具体方法。而想要将这篇论文中的结构运用到其他 NLP 问题上，比如对话系统等等，则需要进一步的实验，以及对于模型架构的新的改进。

