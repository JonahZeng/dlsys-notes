# 算子

## add
$$
c = a + b \\\\
\frac{\partial c}{\partial a} = 1 \\\\
\frac{\partial c}{\partial b} = 1 \\\\
\frac{\partial Loss}{\partial a} = \frac{\partial Loss}{\partial c} * 1 \\\\
\frac{\partial Loss}{\partial b} = \frac{\partial Loss}{\partial c} * 1
$$

## sub
$$
c = a - b \\\\
\frac{\partial c}{\partial a} = 1 \\\\
\frac{\partial c}{\partial b} = -1 \\\\
\frac{\partial Loss}{\partial a} = \frac{\partial Loss}{\partial c} * 1 \\\\
\frac{\partial Loss}{\partial b} = \frac{\partial Loss}{\partial c} * (-1)
$$

## neg
$$
c = -a \\\\
\frac{\partial c}{\partial a} = -1 \\\\
\frac{\partial Loss}{\partial a} = \frac{\partial Loss}{\partial c} * (-1)
$$

## element-wise mul
$$
c[k] = a[k] * b[k] \\\\
\frac{\partial c[k]}{\partial a[k]} = b[k] \\\\
\frac{\partial c[k]}{\partial b[k]} = a[k] \\\\
\frac{\partial Loss}{\partial a} = \frac{\partial Loss}{\partial c} * b \\\\
\frac{\partial Loss}{\partial b} = \frac{\partial Loss}{\partial c} * a 
$$
这里 \\(c, a, b\\) 的维度相同即可。

## relu
$$
dout =
\begin{cases}
0 & \text{if } din < 0 \\\\
din & \text{if } din \geq 0
\end{cases}
$$

$$
\frac{\partial dout}{\partial din} = 
\begin{cases}
0 &\quad din < 0 \\\\
1 &\quad din \geq 0
\end{cases}
$$

## sigmod
$$
dout = \frac{1}{1+e^{-din}} \\\\
dout = {(1+e^{-din})}^{-1} \\\\
\frac{\partial dout}{\partial din} = -1 * {(1+e^{-din})}^{-2} * (-1) * e^{-din} \\\\
\frac{\partial dout}{\partial din} = {(1+e^{-din})}^{-2} * e^{-din} = \frac{1}{1+e^{-din}} * \frac{e^{din}}{1+e^{-din}} \\\\
\frac{\partial dout}{\partial din} = dout * (1-dout)
$$

## tanh
$$
dout = \tanh(din) \\\\
\tanh(din) = \frac{\sinh(din)}{\cosh(din)} \\\\
\sinh(din) = \frac{e^{din}-e^{-din}}{2} \\\\
\cosh(din) = \frac{e^{din}+e^{-din}}{2} \\\\
\tanh(din) = \frac{e^{din}-e^{-din}}{e^{din}+e^{-din}} \\\\
\frac{\partial dout}{\partial din} = \frac{(e^{din}+e^{-din})(e^{din}+e^{-din}) - (e^{din}-e^{-din})(e^{din}-e^{-din})}{(e^{din}+e^{-din})^2} \\\\
\frac{\partial dout}{\partial din} = \frac{2*e^{din} * 2 e^{-din}}{(e^{din}+e^{-din})^2} = \frac{4}{(e^{din}+e^{-din})^2} = \frac{4}{4 * \cosh^2(din)} = \frac{1}{\cosh^2(din)} \\\\
\frac{\partial dout}{\partial din} = \frac{\cosh^2(din) - \sinh^2(din)}{\cosh^2(din)} = 1 - \tanh^2(din) = 1 - dout^2
$$

## matmul
$$
𝐴_{𝑚×k}𝐵_{k×n} = 𝐶_{m×n}
$$

展开表示：
$$
\underset{m \times k}{\begin{bmatrix}
a_{11} & \cdots & a_{1k} \\\\
\vdots &        & \vdots \\\\
a_{m1} & \cdots & a_{mk}
\end{bmatrix}}
\*
\underset{k \times n}{\begin{bmatrix}
b_{11} & \cdots & b_{1n} \\\\
\vdots &        & \vdots \\\\
b_{k1} & \cdots & b_{kn}
\end{bmatrix}}
\=
\underset{m \times n}{\begin{bmatrix}
c_{11} & \cdots & c_{1n} \\\\
\vdots &        & \vdots \\\\
c_{m1} & \cdots & c_{mn}
\end{bmatrix}}
$$

已知：
$$
c_{ij} = \sum_{p=1}^{k} a_{ip} * b_{pj}
$$

对于 \\(a_{11}\\) 来说，它对结果 \\(c\\) 的贡献仅仅局限在 \\(c_{11} \cdots c_{1n}\\) 这一行，所以它对于最终 \\(loss\\) 的梯度就是：
$$
\sum_{p=1}^{n} \frac {\partial {c_{1p}}}{\partial a_{11}} * c_{1p}^{'} = \sum_{p=1}^{n} b_{p1} * c_{1p}^{'}
$$

其中 \\(c_{1p}^{'}\\) 表示上游梯度。
对于 \\(a_{12}\\) 来说，它对结果 \\(c\\) 的贡献也是仅仅局限在 \\(c_{11} \cdots c_{1n}\\) 这一行，所以它对于最终 \\(loss\\) 的梯度就是：
$$
\sum_{p=1}^{n} \frac {\partial {c_{1p}}}{\partial a_{12}} * c_{1p}^{'} = \sum_{p=1}^{n} b_{p2} * c_{1p}^{'}
$$

推广到所有的 \\(a_{ij}\\) 上，得到：
$$
\sum_{p=1}^{n} \frac {\partial {c_{ip}}}{\partial a_{ij}} * c_{ip}^{'} = \sum_{p=1}^{n} b_{pj} * c_{ip}^{'}
$$

所以得到：
$$
\frac {\partial {Loss}}{\partial A} = \frac {\partial {Loss}}{\partial C} * B^{T} \\\\
A \in \mathbb{R}^{m \times k}, \frac {\partial {Loss}}{\partial A} \in \mathbb{R}^{m \times k} \\\\
\frac {\partial {Loss}}{\partial C} \in \mathbb{R}^{m \times n}, B^{T} \in \mathbb{R}^{n \times k}
$$

类似的，得到：

对于 \\(b_{11}\\) 来说，它对结果 \\(c\\) 的贡献仅仅局限在 \\(c_{11} \cdots c_{m1}\\) 这一列，所以它对于最终 \\(loss\\) 的梯度就是：
$$
\sum_{p=1}^{m} \frac {\partial {c_{p1}}}{\partial b_{11}} * c_{p1}^{'} = \sum_{p=1}^{m} a_{1p} * c_{p1}^{'}
$$
对于 \\(b_{21}\\) 来说，它对结果 \\(c\\) 的贡献仅仅局限在 \\(c_{11} \cdots c_{m1}\\) 这一列，所以它对于最终 \\(loss\\) 的梯度就是：
$$
\sum_{p=1}^{m} \frac {\partial {c_{p1}}}{\partial b_{21}} * c_{p1}^{'} = \sum_{p=1}^{m} a_{2p} * c_{p1}^{'}
$$

推广到所有的 \\(b_{ij}\\) 上，得到：
$$
\sum_{p=1}^{m} \frac {\partial {c_{pj}}}{\partial b_{ij}} * c_{pj}^{'} = \sum_{p=1}^{m} a_{ip} * c_{pj}^{'}
$$

所以得到：
$$
\frac {\partial {Loss}}{\partial B} = A^{T} * \frac {\partial {Loss}}{\partial C}  \\\\
B \in \mathbb{R}^{k \times n}, \frac {\partial {Loss}}{\partial B} \in \mathbb{R}^{k \times n} \\\\
A^{T} \in \mathbb{R}^{k \times m}, \frac {\partial {Loss}}{\partial C} \in \mathbb{R}^{m \times n}
$$

## sum
$$
dout = \sum_{i=0}^{K} din_{i} \\\\
\frac{\partial dout}{\partial din_{i}} = 1
$$

## cross_entropy
cross_entropy 损失函数计算分为两步计算，第一步计算每一个类别的 \\(p\\) :
$$
p(lable=i) = softmax(h_{i}) = \frac{\exp(h_{i})}{\sum_{j=0}^{k}\exp(h_{j})}
$$
这里为了防止分母太大或者太小，做一个等比例缩放：
$$
h_{max} = max([h_{0}...h_{k}]) \\\\
p(lable=i) = softmax(h_{i}) = \frac{\exp(h_{i} - h_{max})}{\sum_{j=0}^{k}\exp(h_{j} - h_{max})}
$$
第二步计算 \\(loss\\) :
$$
loss = -\log(p(lable=y)) = -\log(\frac{\exp(h_{y} - h_{max})}{\sum_{j=0}^{k}\exp(h_{j} - h_{max})}) = -(h_{y} - h_{max}) + \log(\sum_{j=0}^{k}\exp(h_{j} - h_{max}))
$$

然后它的导数就很明了了，我们需要对所有的 \\(h_{i}\\) 求导：
$$
\frac{\partial loss}{\partial h_{i}} = \begin{cases}
-1 + \frac{\exp(h_{i} - h_{max})}{\sum_{j=0}^{k}\exp(h_{j} - h_{max})} &\quad i = y \\\\
\frac{\exp(h_{i} - h_{max})}{\sum_{j=0}^{k}\exp(h_{j} - h_{max})} &\quad i \neq y
\end{cases}
$$
