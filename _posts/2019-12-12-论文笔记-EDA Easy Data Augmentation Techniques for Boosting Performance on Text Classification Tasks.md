---
layout:     post
title:      论文笔记
subtitle:   EDA Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks
date:       2019-12-12
author:     dhx20150812
header-img: img/post-bg-ios9-web.jpg
catalog: true
mathjax: true
tags:
    - NLP
    - 数据增强
---



# EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks


>   来自EMNLP 2019 short paper，https://arxiv.org/abs/1901.11196.pdf

## Introduction

作者提出了一个简单实用的文本数据增强手段——Easy Data Augmentation(EDA)。尽管已经有人提出了一些数据增强的手段，比如将英文文本翻译到法语，然后再翻译回英文。但是这样的手段在实践中并不是很常用，于是作者提出了自己的简单实用的文本数据增强手段——EDA。代码已开源在[Github](http://github.com/jasonwei20/eda_nlp)。

## EDA

详细介绍下作者提出的EDA，对于数据集中一个给定的句子，主要有四种操作——

1.  **Synonym Replacement (SR)** ——随机从句子中选择 $n$ 个词(非停用词)，然后从它们的同义词中随机挑选一个替换掉原词。
2.  **Random Insertion(RI)** ——随机地从句子中选择一个词（非停用词），再随机地选择它的一个同义词，插入句子的任意一个位置，重复这一操作 $n$ 次。
3.  **Random Swap** —— 随机从句子中选择两个词，交换他们的位置。重复这一操作 $n$ 次。
4.  **Random Deletion** —— 以概率 $p$ 随机删除句子里的每个词。

由于长的句子有更多的单词，因为在面对扰动时更能保持自己的标签不变。于是根据句子长度，作者制定了一个比例 $\alpha$ 来控制被改变的单词数目。于是有

$$
n=\alpha l
$$

同时，对每个原句子，作者平均生成了 $n\_{avg}$ 个经过EDA处理的句子。

##  实验与分析

### EDA Makes Gains

作者在CNN和RNN模型上实验了EDA的效果，如图所示，在使用EDA后，500个训练样本上的效果提升了3.0%，而完整的数据集上则提升了0.8%。

<img src="https://note.youdao.com/yws/api/personal/file/WEBb982f33b2cb7f25d9cd1790e421bacde?method=download&shareKey=783cf5487fc6cf6999c13254396837ac" alt="image-20191211180808445" style="zoom:50%;" />

### Training Set Sizing

作者也做了对比试验，验证了不同的训练集大小对于效果提升的影响。训练集大小分别选用了【1%，5%，10%，20%，30%，40%，50%，60%，70%，80%，90%，100%】，在五个数据集的效果和平均效果如下图所示：

![image-20191211181119178](https://note.youdao.com/yws/api/personal/file/WEB25be1d34633c3014a7e5bdf2d508f1d0?method=download&shareKey=738ce827ed03a1f5caf8865838eb198f)

图(a)-(e)分别对应着SST-2、CR、SUBJ、TREC和PC五种不同的数据集。图(f)是五种数据集上的平均值。可见EDA技术在所有的数据集上都取得了比Baseline更好的效果，但同时我们发现，随着数据集的增大，EDA的效果提升变得不明显。我们注意到，在使用了EDA手段后，模型平均只需要50%的训练数据就可以得到大约88.6%的准确率。

### Does EDA conserve true labels?

在做数据增强时，我们只改变了文本，保留了标签不变。但是如果文本经过了较大的改变，会使得标签不正确。为了探究EDA处理后的语句是否改变了语义，作者对其做了可视化处理。做法是在没有EDA处理时训练RNN网络，然后将测试集的句子输入网络，取最后一个全连接层的输出做t-SNE聚类，结果如下：

<img src="https://note.youdao.com/yws/api/personal/file/WEBcae138ab4fe11083b85cefd527306408?method=download&shareKey=64637458556586c8a121a07a3638e117" alt="image-20191211182530641" style="zoom:50%;" />

结果可见，经过EDA处理后的句子与原句子距离还是很近。大部分经过EDA处理的句子还是保留了正确的标签。

### EDA Decomposed

作者探究了不同的操作对于提升分类效果的影响。作者在不同的 $\alpha$ 下分别实验了SR、RI、RS和RD四种操作的影响。结果如下图所示：

![image-20191211192404236](https://note.youdao.com/yws/api/personal/file/WEB8a807b8858218c8ba5df70a9853f06ec?method=download&shareKey=dd26986d1f6336228648d7bce7649bd3)

可见，四种操作都会提升分类效果。其中，在 $\alpha$ 值较小时，SR的效果提升较显著，反而在 $\alpha$ 值较大时效果不显著，可能是因为替换了太多的词导致原句子的语义被改变。其他几种操作也有类似的现象。最终，作者发现 $\alpha=0.1$ 是最温和的点。

### How much augmentation

作者实验了每个句子平均产生的增强句子数目对效果的影响。结果如下：

<img src="https://note.youdao.com/yws/api/personal/file/WEB6059f2a572689f27ffd4bcaa6d68019e?method=download&shareKey=f9bba6634fae8b14a16d32a52c16333d" alt="image-20191211192929577" style="zoom: 33%;" />

对小的训练集来说，更容易发生过拟合，因此生成增强句子会显著提升效果。而对于大的训练集来说，数据量足够训练，产生较多的增强句子反而会影响效果。因此，作者推荐的可用参数如下图所示：

<img src="https://note.youdao.com/yws/api/personal/file/WEB4820c70dbfc47b7ef59c44e27f357774?method=download&shareKey=5075c7dcc69015c8883211d83779afbb" alt="image-20191211193216465" style="zoom:67%;" />
