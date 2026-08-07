# softmax线性回归
pdf地址：[这里](https://dlsyscourse.org/slides/2-softmax_regression.pdf)<br>
这一章使用的数据集是MINIST手写数字，60000张28*28 pixel，10个类别（0~9）。
深度学习三个基本元素：
- 网络模型
- 损失函数
- 优化方法

## 模型
MINIST手写数字分类问题本质上是一个降维问题，每一张图片有28*28个像素，每个像素是`0~255`范围内的值，把它展开到1维上就是一个维度等于784的向量，最后我们要降维到10，每个维度表示一个分类的概率；

模型的任务就是做这个映射：

$$
h : \mathbb{R}^{n} \rightarrow \mathbb{R}^{k} \\\\
n = 28 \times 28 = 784 \\\\
k = 10
$$

这一节使用的网络模型是一个纯线性层，我们知道线性代数里面矩阵可以改变输入维度：

$$
\theta \in \mathbb{R}^{n \times k} \\\\
x \in \mathbb{R}^{n \times 1}  \\\\
h_{\theta}(x) =  = out \in \mathbb{R}^{k \times 1}
$$

这里的\\(\theta\\)就是我们这一节的模型。
实际训练过程中我们会用批量输入进行密集运算，批量的意思是：

$$
\theta \in \mathbb{R}^{n \times k} \\\\
x \in \mathbb{R}^{n \times m}  \\\\
h_{\theta}(x) = \theta^{T} * x = out \in \mathbb{R}^{k \times m} \\\\
m = 60000
$$

我们只要把输入也组成一个矩阵，第一个维度是batch数量`m`，第二个维度`n`不变，就可以一次计算获取多个输入的结果。

## 损失函数
损失描述的是模型输出结果和理想结果之间的距离，比如MINIST里面的一张数字`4`的图片，它经过\\(h_\theta(x)\\)之后产生一个维度为10的向量，我们不妨称之为\\(h(x)\\)好了：

$$
h(x) \in \mathbb{R}^{1 \times k} \\\\
k = 10
$$

我们暂时先假定\\(h(x)\\)的值是`[1, 3, 4, 2, 8, 2, 4, 1, 0, 5]`。
如果我们使用最简单直观的损失函数：

$$
    l_{err}(h(x), y) =
    \begin{cases}
        0, & if\text{ }argmax_{i}h_i(x) = y,\\\\
        1, & otherwise.
    \end{cases} \\\\
    y  = 4, \in \mathbb{R}^{1}
$$

如果使用这个损失函数，问题就来了，`[1, 3, 4, 2, 8, 2, 4, 1, 0, 5]`和`[0, 0, 0, 0, 8, 0, 0, 0, 0, 0]`这两个输出的损失都是0，但明显`[0, 0, 0, 0, 8, 0, 0, 0, 0, 0]`这个结果更好。其次这个分段函数不可微，导致优化非常困难。

### softmax/corss-entropy loss
这个loss函数分两部分组成，第一步是对\\(h(x)\\)进行指数运算然后归一化：

$$
p(lable=i) = softmax(h(x_{i})) = \frac{\exp(h_{i}(x))}{\sum_{j=i}^{k}\exp(h_{j}(x))}
$$

这样做的好处有几个，第一就是把\\(h(x)\\)内的负值转为了正值，而且不改变原来的大小顺序，第二个就是这个过程完全可微（不涉及分段函数），且最后的输出累加和为1；举例来说：

$$
h(x) = [-1, 3, 4, 2, 8, 2, -4, 1, 0, 5] \\\\
softmax(h(x)) = [1.1414e-04, 6.2321e-03, 1.6940e-02, 2.2926e-03, 9.2492e-01, 2.2926e-03, \\\\
 5.6829e-06, 8.4342e-04, 3.1028e-04, 4.6049e-02]
$$

第二部分，真正的loss来了：

$$
l_{err}(h(x), y) = -\log(p(label=y)) = -\log(\frac{\exp(h_{y}(x))}{\sum_{j=i}^{k}\exp(h_{j}(x))}) = -h_{y}(x) + \log(\sum_{j=i}^{k}\exp(h_{j}(x)))
$$

注：这里用到了\\(\log(\frac{a}{b}) = \log(a) - \log(b)\\)。

## 优化
最后，优化问题变成了，找到一个\\(\theta\\)，使得：
$$
\frac{1}{m}\sum_{i=1}^{m}l_{err}(h_{\theta}(x^{i}), y^{i})  = \frac{1}{m}\sum_{i=1}^{m}l_{err}(\theta^{T}(x^{i}), y^{i})
$$
最小。

### 梯度下降
模型目前只有一个线性层，需优化的参数就是\\(\theta \in \mathbb{R}^{n \times k}\\)，梯度下降的优化步骤就是：
$$
\theta = \theta - \alpha * \nabla_{\theta}f(\theta)
$$

这里直接给出最终导数：
$$
\nabla_{\theta}f(\theta) = \nabla_{\theta}l_{err}({\theta}^{T}x, y) = x(z - e_{y})^{T} \\\\
z = softmax({\theta}^{T}x), z \in \mathbb{R}^{k \times 1} \\\\
e_{y} = 1\\{i=y\\}, e_{y} \in \mathbb{R}^{k \times 1} \\\\
x \in \mathbb{R}^{n \times 1}
$$

### 链式求导证明过程
$$
l_{err}(h(x), y) = -h_{y}(x) + \log(\sum_{j=i}^{k}\exp(h_{j}(x)))
$$
首先对\\(h\\)求导：
$$
\frac{\partial l_{err}(h, y)}{\partial h_{i}} = \frac{\partial}{\partial h_{i}}(-h_{y} + \log(\sum_{j=i}^{k}\exp(h_{j}))) \\\\
= -1\\{i=y\\} + \frac{\exp(h_{i})}{\sum_{j=1}^{k}\exp(h_{j})} \\\\
= - e_{y} + z
$$
然后求\\(h\\)对\\(\theta\\)的导数：
$$
\frac{\partial h(x)}{\partial \theta} = \frac{\partial}{\partial \theta}(\theta^{T}x) = x
$$
然后把维度组合起来得到：
$$
\nabla_{\theta}f(\theta)  = x(z - e_{y})^{T}
$$

### trick
这里矩阵求导用到了一个trick，假设：
$$
Y = WX \\\\
W \in \mathbb{R}^{m \times n} \\\\
X \in \mathbb{R}^{n \times 1} \\\\
Y \in \mathbb{R}^{m \times 1} \\\\
L = L(Y)
$$
我们关心的是：
$$
\frac{\partial L}{\partial W}
$$
而不是
$$
\frac{\partial Y}{\partial W}
$$
链式法则：
$$
\frac{\partial L}{\partial W_{ij}} = \sum_{k=1}^{m}\frac{\partial L}{\partial y_{k}} \frac{\partial y_{k}}{\partial  W_{ij}} 
$$
注意到：
$$
y_{k} = \sum_{t=1}^{n} W_{kt}X_{t}
$$
当\\(k \neq i\\)时，\\(y_{k}不依赖W_{ij}\\)，此时导数为0。所以有：
$$
y_{i} = \sum_{t=1}^{n} W_{it}X_{t} \\\\
\frac{\partial y_{i}}{\partial  W_{ij}} = X_{j}
$$
因此：
$$
\frac{\partial L}{\partial W_{ij}} = \frac{\partial L}{\partial y_{i}} X_{j}
$$
如果记：
$$
g:= \frac{\partial L}{\partial Y} \in \mathbb{R}^{m \times 1}
$$
那么就有：
$$
\frac{\partial L}{\partial W} = g X^{T}
$$