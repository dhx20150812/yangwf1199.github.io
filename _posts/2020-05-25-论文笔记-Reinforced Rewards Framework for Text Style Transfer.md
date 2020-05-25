# Reinforced Rewards Framework for Text Style Transfer

>   来自ECIR 2020

## Introduction

文本风格迁移是在保留文本的核心内容的前提下，将给定文本的风格迁移为另一种目标风格的一个任务。现有的方法在训练时主要使用了word-level的目标函数，而它与任务所期望的指标（内容保留和迁移强度）不一致。任务期望的指标通常在句子级别进行计算，因此word-level的目标函数是不够的。而且，这些指标的离散性使得模型无法对其直接优化。

在本文中，作者提出了一种基于强化学习的框架。它采用了句子级别的目标函数来完成风格迁移。该框架基于Seq2seq模型和注意力机制来产生文本。由该模型生成的句子与ground-truth的句子一起传递到内容模块和风格分类器，该风格分类器计算度量分数以获得最终的reward。 然后，这些reward将以损失项的形式传播回Seq2seq模型。

## Reinforced Rewards Framework

### Problem Formalization

问题的输入是一个句子 $x=x_1 \cdots x_l$，它具有风格 $s_1$。我们需要将其转换为目标句子 $y=y_1 \cdots y_m$，它具有目标风格 $s_2$。$x$ 包含两个部分，内容 $c_1$ 和风格 $s_1$，目标是生成句子 $y$，它由相同的内容 $c_1$，但有不同的风格 $s_2$。

### Generation Process

我们的方法基于一种 copy-enriched seq2seq 框架。训练时的输入是源文本 $x$ 和风格 $c_1$，输入句子中的单词通过一个 LSTM 编码器映射到隐层空间，网络通过注意力机制得到一个上下文向量，解码器综合了一个RNN网络和一个PTR网络，前者预测词表上的概率分布，后者预测输入句子上的概率分布。两个分布的加权和得到了最后的预测输出：
$$
P_{t}(w)=\delta P_{t}^{R N N}(w)+(1-\delta) P_{t}^{P T R}(w)
$$

其中，$\delta$ 由编码器的输出和解码器上一时刻的隐层状态计算得到。模型的目标是优化如下的损失函数：

$$
L_{m l}=-\sum_{t=1}^{m} \log \left(p\left(P_{t}\left(y_{t}^{*}\right)\right)\right)
$$

其中，$m$ 是输出句子的最大长度。$y^*_t$ 是ground truth的词。

![image-20200517145153444](https://note.youdao.com/yws/api/personal/file/WEB5c0de01b3b3f8893144cd908e3dbe065?method=download&shareKey=e38990a7ec56f4ab24ab0a67eb254070)

值得注意的是，当模型的优化目标是生成与ground truth接近的句子时，并没有保证生成的句子保留了原来的内容，同时迁移到了新的目标风格。

为了达到这两点目标，我们使用了一个风格分类器和一个内容模块。模型的整体框架如上图所示。他们以生成的句子和ground truth的句子为输入，获得reward。我们以BLEU评分来衡量模型保留了原内容的reward，然后以分类器的分数作为迁移强度的reward，这些reward作为损失项回传，对网络产生的不正确输出进行惩罚。

### Content Module

为了在迁移风格的同时保留文本的内容，作者使用了Self-Critic Sequence Training（SCST）的方法，并使用BLEU作为reward分数。SCST是一种强化学习中的策略梯度的方法，它可以在不可微的评估指标下直接训练端到端的模型。BLEU评分衡量了ground truth和生成句子之间的重叠。

作者产生了两个输出的句子，一个是从概率分布 $p\left(y_{t}^{s} | y_{1: t-1}^{s}, x\right)$ 中采样得到的 $y^s$，另一个是在每一时刻贪心的选择概率最大的输出 $y^{\prime}$。因此这一部分的loss计算为

$$
L_{c p}=\left(r\left(y^{\prime}\right)-r\left(y^{s}\right)\right) \sum_{t=1}^{m} \log \left(p\left(y_{t}^{s} | y_{1: t-1}^{s}, x\right)\right)
$$

注意到这个公式并不要求评价指标是可微的，因为reward是作为loss的权重项。最小化 $L_{cp}$ 等价于鼓励模型生成更高reward的句子。Content modula的作用是鼓励模型保留文本中的内容。

### Style Classifier

为了鼓励模型生成属于目标风格的句子，作者训练了一个风格分类器。它的作用是对输入的文本进行判别，预测它属于目标风格的似然性。因此，可以再增加一个如下的 loss 项：

$$
L_{t s}=\left\{\begin{array}{ll}-\log \left(1-s\left(y^{\prime}\right)\right), & \text { high to low level } \\ -\log \left(s\left(y^{\prime}\right)\right), & \text { low to high level }\end{array}\right.
$$

在这个公式中，$y^{\prime}$ 是从解码器输出的概率分布中贪心得到的，$s(y^{\prime})$ 是风格分类器对于 $y^{\prime}$ 的似然性分数。

### Training and Inference

整体的loss如下三项构成：

$$
Loss =\alpha L_{m l}+\beta L_{c p}+\gamma L_{t s}
$$

经过超参调整后，合适的参数设置为 $\alpha=1.0, \beta=0.125,\gamma=1.0$。

