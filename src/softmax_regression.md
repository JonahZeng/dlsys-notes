# softmax线性回归
pdf地址：[这里](https://dlsyscourse.org/slides/2-softmax_regression.pdf)

这一章使用的数据集是MINIST手写数字，60000张28*28 pixel，10个类别（0~9）。
深度学习三个基本元素：
- 网络模型
- 损失函数
- 优化方法

## 模型
MINIST手写数字分类问题本质上是一个降维问题，每一张图片有28*28个像素，每个像素是`0~255`范围内的值，把它展开到1维上就是一个维度等于784的向量，最后我们要降维到10，每个维度表示一个分类的概率；

模型的任务就是做这个映射：

$$
\begin{aligned}
h : \mathbb{R}^{n} \rightarrow \mathbb{R}^{k} \\
n = 28 \times 28 = 784 \\
k = 10
\end{aligned}
$$


这一节使用的网络模型是一个纯线性层，我们知道线性代数里面矩阵可以改变输入维度：

$$
\begin{aligned}
\theta \in \mathbb{R}^{n \times k} \\
x \in \mathbb{R}^{1 \times n}  \\
h_{\theta}(x) = x * \theta = out \in \mathbb{R}^{1 \times k}
\end{aligned}
$$

这里的$\theta$就是我们这一节的模型。
实际训练过程中我们会用批量输入进行密集运算，批量的意思是：

$$
\begin{aligned}
\theta \in \mathbb{R}^{n \times k} \\
x \in \mathbb{R}^{m \times n}  \\
h_{\theta}(x) = x * \theta = out \in \mathbb{R}^{m \times k} \\
m = 60000
\end{aligned}
$$

我们只要把输入也组成一个矩阵，第一个维度是batch数量`m`，第二个维度`n`不变，就可以一次计算获取多个输入的结果。

## 损失函数
损失描述的是模型输出结果和理想结果之间的距离，比如MINIST里面的一张数字`4`的图片，它经过$h_\theta(x)$之后产生一个维度为10的向量，我们不妨称之为$h(x)$好了：

$$
h(x) \in \mathbb{R}^{1 \times k} \\
k = 10
$$

我们暂时先假定它的值是`[1, 3, 4, 2, 8, 2, 4, 1, 0, 5]`。
如果我们使用最简单直观的损失函数：

$$
    l_{err}(h(x), y) =
    \begin{cases}
        0, & if argmax_{i}h_i(x) = y,\\
        1, & otherwise.
    \end{cases}
$$

如果使用这个损失函数，问题就来了，`[1, 3, 4, 2, 8, 2, 4, 1, 0, 5]`和`[0, 0, 0, 0, 8, 0, 0, 0, 0, 0]`这两个输出的损失都是0，但明显`[0, 0, 0, 0, 8, 0, 0, 0, 0, 0]`这个结果更好。其次这个分段函数不可微，导致优化非常困难。
