---
layout:     post
title:      论文笔记
subtitle:   A Deep Reinforced Sequence-to-Set Model for Multi-Label Text Classification
author:     dhx20150812
header-img: img/post-bg-ios9-web.jpg
catalog: true
mathjax: true
tags:
    - NLP
    - 多标签分类
---

# A Deep Reinforced Sequence-to-Set Model for Multi-Label Text Classification


>   来自AAAI 2019，https://arxiv.org/pdf/1809.03118.pdf

## Introduction

多标签文本分类任务的目标是为数据集中的每个文本分配多个标签。这些标签之间通常都有一些内在的联系，传统的方法都忽略了这些联系。而 Seq2Seq 模型将多标签分类问题建模为序列生成任务，预测产生一个多标签的序列。

作者认为，Seq2Seq 模型不适合这一任务。因为 Seq2Seq 模型通常需要人们预先定义输出标签的顺序，而多数情况下，输出的标签更是一个集合，而不是有序的序列。[前人的工作](https://arxiv.org/pdf/1511.06391.pdf)已经证明了输入和输出的顺序对 Seq2Seq 模型的效果有很大的影响。而我们无法提前知道预测的输出标签的最佳顺序。

因此，将输出标签定义为无序的集合更适合这一任务。无序的集合有一个很重要的性质——交换不变性。我们交换集合中的任意两个元素，集合保持不变。

## Proposed Approach

给定一段文本 $x$ 包含 $m$ 个单词，多标签分类的任务是为 $x$ 分配标签集合 $\mathcal{L}$ 的一个子集 $\mathcal{y}$ ——包含 $n$ 个 label。作者提出的 Seq2Set 模型如下所示——

![image-20191202165111911](https://note.youdao.com/yws/api/personal/file/WEB120050cd87b594eb43aa75b1be41ee59?method=download&shareKey=428cbe9227625d2e7766c357ff3ce9bc)

如图所示，整个模型有一个编码器 $\mathcal{E}$ 和两个解码器$\mathcal{D}_1$和$\mathcal{D}_2$。给定输入文本序列 $x$，编码器$\mathcal{E}$和序列解码器$\mathcal{D}_1$联合工作，将其编码为隐层表示$\hat{s}=\left(\hat{s}_{0},\hat{s}_{1},\cdots,\hat{s}_{n}\right)$，它的目的是学习符合可能的人类先验知识的初步标签顺序。然后集合解码器$\mathcal{D}_2$将编码器$\mathcal{E}$和序列解码器$\mathcal{D}_1$的隐层状态作为输入，产生最终的预测标签集合。在这过程中使用带有 self-critical 训练的策略梯度方法来减少模型对标签顺序的依赖性。

### Encoder $\mathcal{E}$

作者在此处采用了一个双向的 LSTM，它从两个方向读取输入序列，然后每个词对应的隐层状态，如下所示：

$$
\begin{array}{l}{\vec{h}_{i}=\overrightarrow{\operatorname{LSTM}}\left(\overrightarrow{h}_{i-1}, x_{i}\right)} \\ {\overleftarrow{h}_{i}=\overleftarrow{\operatorname{LSTM}}\left(\overleftarrow{h}_{i+1}, x_{i}\right)}\end{array}
$$

最终第 $i$ 个词的隐层状态是将这两者拼接在一起。h_i=\left[\overrightarrow{h}_{i-1}; \overleftarrow{h}_{i-1}\right]。

### Sequence Decoder $\mathcal{D}_1$

给定编码器 $\mathcal{E}$ 的隐层状态向量 $(h_1,\cdots,h_m)$，序列解码器 $\mathcal{D_1}$ 像标准的 Seq2Seq 模型一样解码。此处作者使用了注意力机制，在$t+1$时刻的隐层状态$\hat{h}_{t+1}$ 通过如下的计算得到：

$$
\begin{aligned} e_{t, i} &=\boldsymbol{v}_{a}^{T} \tanh \left(\boldsymbol{W}_{a} \hat{s}_{t}+\boldsymbol{U}_{a} h_{i}\right) \\ \alpha_{t, i} &=\frac{\exp \left(e_{t, i}\right)}{\sum_{j=1}^{m} \exp \left(e_{t, j}\right)} \\ c_{t} &=\sum_{i=1}^{m} \alpha_{t, i} h_{i} \\ \hat{s}_{t+1} &=\mathrm{LSTM}\left(\hat{s}_{t},\left[e\left(y_{t}\right) ; c_{t}\right]\right) \end{aligned}
$$

其中，$[e(y_t);c_t]$ 代表了向量 $e(y_t)$ 和向量 $c_t$ 的拼接，$\boldsymbol{v}_{a}$、$\boldsymbol{W}_{a}$ 和  $\boldsymbol{U}_{a}$ 是权重矩阵，$e(y_t)$ 是最后一个时刻的最高概率的标签对应的 embedding 向量，关于标签空间的概率分布有如下计算：

$$
\begin{array}{l}{o_{t}=\boldsymbol{W}_{o} f\left(\boldsymbol{W}_{d} \hat{s}_{t}+\boldsymbol{V}_{d} c_{t}\right)} \\ {y_{t} \sim \operatorname{softmax}\left(o_{t}+I_{t}\right)}\end{array}
$$

其中，$\boldsymbol{W}_{o}$、$\boldsymbol{W}_{d}$ 和 $\boldsymbol{V}_{d}$ 是权重参数。$I_t \in \mathbb{R}^L$ 是 mask 向量，它的作用是防止序列解码器 $\mathcal{D}_1$ 生成重复的标签，$f$ 是一个非线性函数。

$$
\left(I_{t}\right)_{i}=\left\{\begin{array}{l}{-\infty} ~~~~~~~~~~~~\text{if the $i$-th label has been predicted}\\ {0} ~~~~~~~~~~~~~~~~~\text{otherwise}\end{array}\right.
$$


### Set decoder $\mathcal{D}_2$

集合解码器 $\mathcal{D}_2$ 是整个模型的核心部分，它的目标是捕获标签之间的联系，并且避免标签顺序之间的强烈依赖。在序列解码器 $\mathcal{D}_1$ 生成对应的隐层状态序列之后，集合解码器 $\mathcal{D}_2$ 根据 $\mathcal{D}_1$ 的隐层状态序列来生成更准确的标签。

给定编码器 $\mathcal{E}$ 的隐层状态序列向量 $(h_1,\cdots,h_m)$ 和序列解码器 $\mathcal{D}_1$ 的隐层状态序列 $\left(\hat{s}_{0}, \cdots, \hat{s}_{n}\right)$，分别对其使用注意力机制计算得到两个上下文向量——分别是编码器上下文向量 $c_t^e$ 和序列解码器上下文向量 $c_t^d$ 。集合解码器 $\mathcal{D}_2$ 在 $t+1$ 时刻的隐层状态由如下计算得到：

$$
s_{t+1}=\operatorname{LSTM}\left(s_{t},\left[e\left(y_{t}\right) ; c_{t}^{e} ; c_{t}^{d}\right]\right)
$$

同时这里也使用了与序列解码器 $\mathcal{D}_1$ 相似的 mask 向量和 softmax 层。

## Training and Testing

为了更好地使模型摆脱对标签顺序的依赖，作者从强化学习的角度对多标签分类任务进行建模。此时，模型中的集合解码器 $\mathcal{D}_2$ 是 agent，在 $t$ 时刻的 state 是已经生成的标签 $(y_0,\cdots,y_{t-1})$，action 是下一时刻生成的标签。当生成完一个完整的标签子集 $\boldsymbol{y}$ 时，agent $\mathcal{D}_2$ 会得到一个reward $r$，训练的目标是最小化负期望reward——

$$
L(\theta)=-\mathbb{E}_{\boldsymbol{y} \sim p_{\theta}}[r(\boldsymbol{y})]
$$

作者此处使用了self-critical策略梯度训练算法，如图所示——

<img src="https://note.youdao.com/yws/api/personal/file/WEBc9f2e6dd6737fa895e0ffa67c235ab26?method=download&shareKey=ed7dec7f4aa18d84b22359b70f71423b" alt="image-20191203150729504" style="zoom:67%;" />

对每个minibatch里的每个样本，上述reward的期望值可以近似为

$$
\nabla_{\theta} L(\theta) \approx-\left[r\left(\boldsymbol{y}^{s}\right)-r\left(\boldsymbol{y}^{g}\right)\right] \nabla_{\theta} \log \left(p_{\theta}\left(\boldsymbol{y}^{s}\right)\right)
$$

其中，$\boldsymbol{y}^s$ 是从概率分布 $p_{\theta}$ 中采样得到的标签序列，$\boldsymbol{y}^g$ 是使用贪心算法得到的标签序列。

### Reward Design

作者采用了 $F_1$ 值作为 reward——

$$
r(\boldsymbol{y})=\mathrm{F}_{1}\left(\boldsymbol{y}, \boldsymbol{y}^{*}\right)
$$

其中，$\boldsymbol{y}$ 和 $\boldsymbol{y}^*$ 分别是生成的标签序列和生成的标签序列。

### Training Objective

序列解码器 $\mathcal{D}_1$ 的训练目标是最大化标签序列的条件概率——

$$
L(\phi)=-\sum_{t=0}^{n} \log \left(p\left(y_{t}^{*} | \boldsymbol{y}_{<t}^{*}, \boldsymbol{x}\right)\right)
$$

其中，$\phi$ 代表 $\mathcal{D}_1$ 的参数，$\boldsymbol{y}_{<t}^{*}$ 代表了序列 $\left(y_{0}^{*}, \cdots, y_{t-1}^{*}\right)$。最终的目标函数如下：

$$
L_{t o t a l}=(1-\lambda) L(\phi)+\lambda L(\theta)
$$

$L(\phi)$旨在将标签顺序的先验知识融合到模型中，$L(\theta)$ 旨在降低标签顺序的依赖性。

## Results and Discussion

作者在两个数据集（RCV1-V2、AAPD）上分别进行了实验，最终的结果如下图所示——

<img src="https://note.youdao.com/yws/api/personal/file/WEB5c316d4231992c4f3f1d1d56ef5b7a32?method=download&shareKey=17e926adce038f7718a3482ecceb3132" alt="image-20191203160555024" style="zoom:50%;" />

<img src="https://note.youdao.com/yws/api/personal/file/WEBe8d74288eedcb6d380ca64191abd0e96?method=download&shareKey=b148fa4f8b38dac7d95a13c0cefc5643" alt="image-20191203160607003" style="zoom:50%;" />
